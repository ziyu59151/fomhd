#include "fomhd/verification.hpp"
#include "fomhd/diagnostics.hpp"
#include "fomhd/backend.hpp"
#include "fomhd/physics.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numbers>
#include <set>
#include <sstream>

namespace fomhd {
namespace {
constexpr double tau = 2 * std::numbers::pi;
std::string norm_name(Norm n) {
  return n == Norm::linf          ? "linf"
         : n == Norm::weighted_l2 ? "weighted_L2"
                                  : "value";
}
std::string budget_key(const std::string &name, const std::string &pop,
                       Norm n) {
  return name + (pop.empty() ? "" : "." + pop) + "." + norm_name(n);
}
std::string check_key(const Check &r) {
  return r.name + "|" + r.population + "|" + r.component + "|" + r.producer +
         "|" + r.region + "|" + norm_name(r.norm) + "|" + r.tag.stage + "|" +
         std::to_string(r.tag.step) + "|" + std::to_string(r.tag.generation) +
         "|" + std::to_string(r.tag.trial) + "|" +
         std::to_string(r.tag.spectral_context) + "|" +
         std::to_string(std::bit_cast<std::uint64_t>(r.tag.time));
}
void truth(std::vector<Check> &checks, const std::string &name, bool ok,
           const StageTag &t) {
  Check x;
  x.name = name;
  x.tag = t;
  x.unit = x.value_unit = "1";
  x.norm = Norm::value;
  x.raw = ok ? 0 : 1;
  x.passed = ok;
  x.hard = true;
  x.reference = 1;
  checks.push_back(x);
}
} // namespace
const Budget &budget(const Context &c, const std::string &key) {
  const auto it = c.p.budgets.find(key);
  if (it == c.p.budgets.end())
    throw Failure("MissingBudget", "missing diagnostic budget: " + key);
  return it->second;
}
void measure(std::vector<Check> &out, const std::string &name,
             const Vec &residual, const RealVec &w, const std::string &unit,
             const StageTag &t, const Context &c, const std::string &pop,
             bool samples, const std::string &component) {
  // The local periodic verification adapter has no boundary entities.
  if (w.empty() && residual.empty() && name.starts_with("wall_") &&
      c.geometry.walls.empty() && c.op.T_full.rows == 0 && c.op.T.rows == 0) {
    for (const auto *region : {"all", "inner", "wall"})
      for (Norm norm : {Norm::linf, Norm::weighted_l2}) {
        const auto &b = budget(c, budget_key(name, pop, norm));
        if (b.unit != unit || b.norm != norm)
          throw Failure("BudgetUnit", "wrong unit/norm for " + name);
        Check record;
        record.name = name;
        record.population = pop;
        record.component = component;
        record.region = region;
        record.tag = t;
        record.unit = record.value_unit = unit;
        record.norm = norm;
        record.available = record.passed = record.hard = false;
        record.reason = "not applicable: no boundary entities";
        out.push_back(std::move(record));
      }
    return;
  }
  if (w.empty() || residual.empty() || residual.size() % w.size())
    throw Failure("DiagnosticShape", name);
  RealVec distance;
  std::string measure_unit = "m^3";
  auto append_distance = [&](const EntityTable &entities) {
    for (std::size_t i = 0; i < entities.keys.size(); ++i) {
      double d = std::numeric_limits<double>::infinity();
      for (const auto &wall : c.geometry.walls) {
        const auto a = wall.vertices[0], b = wall.vertices[1];
        d = std::min(d,
                     point_segment_distance(
                         {entities.R[i], entities.Z[i]},
                         {c.geometry.vertices.R[a], c.geometry.vertices.Z[a]},
                         {c.geometry.vertices.R[b], c.geometry.vertices.Z[b]}));
      }
      distance.push_back(d);
    }
  };
  const bool wall = name.starts_with("wall_");
  if (wall) {
    measure_unit = "1";
    distance.assign(w.size(), 0);
  } else if ((name == "alias_B" || name == "equilibrium_B") &&
             !component.empty()) {
    if (component == "R")
      append_distance(c.geometry.z_faces);
    else if (component == "phi")
      append_distance(c.geometry.vertices);
    else if (component == "Z")
      append_distance(c.geometry.r_faces);
    else
      throw Failure("DiagnosticComponent", component);
  } else if (w == c.op.M0)
    distance = c.geometry.cell_wall_distance;
  else if (w == c.op.M1) {
    append_distance(c.geometry.r_faces);
    append_distance(c.geometry.cells);
    append_distance(c.geometry.z_faces);
  } else if (w == c.op.MB) {
    append_distance(c.geometry.z_faces);
    append_distance(c.geometry.vertices);
    append_distance(c.geometry.r_faces);
  } else if (w == c.geometry.wv)
    append_distance(c.geometry.vertices);
  else if (w.size() == 3 * c.op.Nc) {
    for (int d = 0; d < 3; ++d)
      distance.insert(distance.end(), c.geometry.cell_wall_distance.begin(),
                      c.geometry.cell_wall_distance.end());
  } else
    throw Failure("DiagnosticLocation", "unknown spatial weights for " + name);
  if (distance.size() != w.size())
    throw Failure("DiagnosticShape", "region weights mismatch");
  for (double weight : w)
    if (!(weight > 0) || !std::isfinite(weight))
      throw Failure("DiagnosticWeight", name);
  for (const auto &region : std::vector<std::string>{"all", "inner", "wall"})
    for (Norm n : {Norm::linf, Norm::weighted_l2}) {
      const auto &b = budget(c, budget_key(name, pop, n));
      if (b.unit != unit || b.norm != n)
        throw Failure("BudgetUnit", "wrong unit/norm for " + name);
      Check x;
      x.name = name;
      x.population = pop;
      x.component = component;
      x.region = region;
      x.tag = t;
      x.unit = unit;
      x.norm = n;
      x.hard = b.hard;
      x.value_unit = n == Norm::weighted_l2 && measure_unit != "1"
                         ? unit + " * sqrt(" + measure_unit + ")"
                         : unit;
      double raw = 0, volume = 0, largest = 0;
      for (std::size_t i = 0; i < residual.size(); ++i) {
        bool inner = distance[i % w.size()] >= c.p.inner_distance;
        if ((region == "inner" && !inner) || (region == "wall" && inner))
          continue;
        ++x.selected;
        const double a = std::abs(residual[i]);
        if (!std::isfinite(a)) {
          x.available = false;
          x.reason = "nonfinite residual";
          x.worst = i;
          break;
        }
        if (a > largest) {
          largest = a;
          x.worst = i;
        }
        if (n == Norm::weighted_l2) {
          raw = std::hypot(raw, std::sqrt(w[i % w.size()]) * a);
          volume += w[i % w.size()];
        } else
          raw = std::max(raw, a);
      }
      if (n == Norm::weighted_l2) {
        const double angular = samples ? tau / c.N : tau;
        raw *= std::sqrt(angular);
        // A constant physical reference has only mode zero, not M repeated
        // modes.
        const double modes = samples ? 1 : double(residual.size() / w.size());
        x.reference = b.reference * std::sqrt(angular * volume / modes);
      } else
        x.reference = b.reference;
      if (!x.selected) {
        x.available = false;
        x.reason = "empty region";
        x.hard = false;
      }
      x.raw = raw;
      x.threshold = b.atol + b.rtol * x.reference;
      x.passed = x.available && std::isfinite(raw) && raw <= x.threshold;
      out.push_back(std::move(x));
    }
}
void scalar_check(std::vector<Check> &out, const std::string &name, double raw,
                  const std::string &unit, const StageTag &t, const Context &c,
                  const std::string &pop) {
  const auto key = budget_key(name, pop, Norm::value);
  const auto &b = budget(c, key);
  if (b.unit != unit || b.norm != Norm::value)
    throw Failure("BudgetUnit", "wrong scalar unit for " + name);
  Check x;
  x.name = name;
  x.population = pop;
  x.tag = t;
  x.unit = x.value_unit = unit;
  x.norm = Norm::value;
  x.hard = b.hard;
  x.raw = raw;
  x.reference = b.reference;
  x.threshold = b.atol + b.rtol * b.reference;
  x.available = std::isfinite(raw);
  x.passed = x.available && std::abs(raw) <= x.threshold;
  out.push_back(x);
}
void gate(const std::vector<Check> &checks,
          const std::vector<std::string> &required) {
  std::set<std::string> keys, names;
  for (const auto &x : checks) {
    if (!keys.insert(check_key(x)).second)
      throw Failure("DuplicateDiagnostic", check_key(x));
    names.insert(x.name);
    if (x.hard && (!x.available || !x.passed)) {
      std::ostringstream s;
      s.precision(17);
      s << "hard check " << x.name << "/" << x.population << "/"
        << norm_name(x.norm) << " = " << x.raw << ", threshold " << x.threshold;
      Failure f("HardGate", s.str());
      f.checks = checks;
      throw f;
    }
  }
  for (const auto &n : required)
    if (!names.contains(n))
      throw Failure("MissingDiagnostic", n);
}

void gate_records(const std::vector<Check> &records,
                  const std::vector<Check> &required) {
  gate(records);
  std::set<std::string> present;
  for (const auto &record : records)
    present.insert(check_key(record));
  for (const auto &record : required)
    if (!present.contains(check_key(record)))
      throw Failure("MissingDiagnostic", check_key(record));
}
std::vector<Check> stage_requirements(const Context &c, const StageTag &tag) {
  std::vector<Check> out;
  auto scalar = [&](const std::string &name, const std::string &pop = "") {
    Check x;
    x.name = name;
    x.population = pop;
    x.norm = Norm::value;
    x.tag = tag;
    out.push_back(x);
  };
  auto array = [&](const std::string &name, const std::string &pop = "") {
    for (auto region : {"all", "inner", "wall"})
      for (auto norm : {Norm::linf, Norm::weighted_l2}) {
        Check x;
        x.name = name;
        x.population = pop;
        x.region = region;
        x.norm = norm;
        x.tag = tag;
        out.push_back(x);
      }
  };
  for (auto name : {"finite", "hermitian_rhs", "source_identity",
                    "memory_bound", "minimum_B"})
    scalar(name);
  for (auto name :
       {"poisson_linear", "darwin_dynamic", "longitudinal_projection", "gauss",
        "transverse_divergence", "current_divB", "current_ampere",
        "CT_increment", "current_response"})
    array(name);
  for (int a = 0; a < 2; ++a) {
    auto id = c.p.species[a].id;
    for (auto name :
         {"minimum_n", "minimum_face_n", "minimum_perp", "electric_pairing",
          "pressure_pairing", "magnetic_work", "wall_work",
          "convection_pairing", "spectral_transfer_defect"})
      scalar(name, id);
    for (auto name :
         {"stage_continuity", "wall_state", "wall_rhs", "wall_multiplier",
          "theta_identity", "wall_number_flux", "wall_perp_flux"})
      array(name, id);
    if (a) {
      scalar("minimum_parallel", id);
      array("cgl_source", id);
      array("wall_parallel_flux", id);
    }
  }
  return out;
}

Totals state_totals(const State &u, const StageResult &s, const Context &c) {
  Totals t;
  const auto nc = c.op.Nc;
  for (int a = 0; a < 2; ++a)
    for (int l = 0; l < c.N; ++l)
      for (std::size_t j = 0; j < nc; ++j) {
        const auto &f = s.fluid[a];
        const double w = tau * c.op.M0[j] / c.N;
        t.number[a] += w * f.n[l * nc + j].real();
        for (int d = 0; d < 3; ++d)
          t.kinetic[a] += w * std::norm(f.mom[l * 3 * nc + d * nc + j]) /
                          (2 * c.p.species[a].mass * f.n[l * nc + j].real());
        if (a == 0)
          t.ion_internal +=
              w * f.perp[l * nc + j].real() / (c.p.species[0].gamma - 1);
        else {
          t.electron_perp += w * f.perp[l * nc + j].real();
          t.electron_parallel += w * f.parallel[l * nc + j].real() / 2;
        }
      }
  for (std::size_t i = 0; i < u.B.size(); ++i)
    t.magnetic +=
        tau * c.op.MB[i % c.op.MB.size()] * std::norm(u.B[i]) / (2 * c.p.mu0);
  for (std::size_t i = 0; i < s.cache.EL.size(); ++i)
    t.longitudinal += tau * c.op.M1[i % c.op.M1.size()] *
                      std::norm(s.cache.EL[i]) * c.p.epsilon0 / 2;
  t.energy = t.kinetic[0] + t.kinetic[1] + t.ion_internal + t.electron_perp +
             t.electron_parallel + t.magnetic + t.longitudinal;
  const auto b0 = slice(u.B, c.p.K * c.op.MB.size(), c.op.MB.size());
  t.topology = c.compute.mv(c.op.topology_flux_row, b0)[0].real();
  for (std::size_t v = 0; v < c.geometry.av.size(); ++v)
    t.physical_flux +=
        c.geometry.av[v] * b0[c.geometry.z_faces.keys.size() + v].real();
  return t;
}

Momentum momentum(const State &, const StageResult &s, const Context &c) {
  Momentum out;
  const auto nc = c.op.Nc, nw = c.geometry.walls.size();
  const auto ELnode = sample_apply(
      c.op.H, inverse(s.cache.EL, c.op.M1.size(), c, true, c.p.ref.electric),
      c.N, c.compute);
  const int N = std::max(c.N, 3);
  out.angular_samples = N;
  const auto cart = [](const std::array<double, 3> &v, double phi) {
    return std::array<double, 3>{v[0] * std::cos(phi) - v[1] * std::sin(phi),
                                 v[0] * std::sin(phi) + v[1] * std::cos(phi),
                                 v[2]};
  };
  for (int q = 0; q < N; ++q) {
    const int l = c.N == 1 ? 0 : q;
    const double phi = tau * q / N;
    for (std::size_t j = 0; j < nc; ++j) {
      std::array<double, 3> p{}, wall{}, E{}, B{};
      for (int d = 0; d < 3; ++d) {
        E[d] = ELnode[l * 3 * nc + d * nc + j].real();
        B[d] = s.BN[l * 3 * nc + d * nc + j].real();
      }
      for (int d = 0; d < 3; ++d) {
        const int a = (d + 1) % 3, b = (d + 2) % 3;
        p[d] = s.fluid[0].mom[l * 3 * nc + d * nc + j].real() +
               s.fluid[1].mom[l * 3 * nc + d * nc + j].real() +
               c.p.epsilon0 * (E[a] * B[b] - E[b] * B[a]);
        wall[d] = s.terms[0].reaction[l * 3 * nc + d * nc + j].real() +
                  s.terms[1].reaction[l * 3 * nc + d * nc + j].real();
      }
      const auto pc = cart(p, phi), wc = cart(wall, phi);
      const double weight = tau * c.op.M0[j] / N;
      for (int d = 0; d < 3; ++d) {
        out.value[d] += weight * pc[d];
        out.wall_reaction[d] += weight * wc[d];
      }
      out.angular += weight * c.geometry.cells.R[j] * p[1];
      out.wall_torque += weight * c.geometry.cells.R[j] * wall[1];
    }
    for (std::size_t w = 0; w < nw; ++w) {
      const auto &wall = c.geometry.walls[w];
      const auto &tr = s.boundary.traces[l * nw + w];
      const int dir = wall.component == WallComponent::r_face ? 0 : 2;
      const double sigma = wall.sign;
      double b2 = 0, el2 = 0, elt = 0;
      for (int d = 0; d < 3; ++d) {
        b2 += tr.B[d] * tr.B[d];
        el2 += tr.EL[d] * tr.EL[d];
        elt += tr.EL[d] * tr.ET[d];
      }
      std::array<double, 3> traction{};
      for (int d = 0; d < 3; ++d) {
        const double unit = d == dir ? 1 : 0;
        double stress =
            (tr.B[d] * tr.B[dir] - 0.5 * b2 * unit) / c.p.mu0 +
            c.p.epsilon0 *
                (tr.EL[d] * tr.EL[dir] - 0.5 * el2 * unit +
                 tr.EL[d] * tr.ET[dir] + tr.ET[d] * tr.EL[dir] - elt * unit);
        traction[d] = -sigma * stress;
        for (int a = 0; a < 2; ++a)
          traction[d] +=
              c.p.species[a].mass * tr.density[a] * tr.velocity[a][d] * sigma *
                  tr.velocity[a][dir] +
              sigma * (tr.perp[a] * unit + (tr.parallel[a] - tr.perp[a]) *
                                               tr.B[d] * tr.B[dir] / b2);
      }
      const auto tc = cart(traction, phi);
      const double weight = tau * wall.area_per_radian / N;
      for (int d = 0; d < 3; ++d)
        out.outward[d] += weight * tc[d];
      const double R = dir == 0 ? c.geometry.r_faces.R[wall.face]
                                : c.geometry.z_faces.R[wall.face];
      out.outward_torque += weight * R * traction[1];
    }
  }
  return out;
}

void check_stage(const State &u, StageResult &s, const Context &c) {
  auto &out = s.checks;
  const auto &t = s.tag;
  const auto nc = c.op.Nc;
  bool finite = is_finite(u.B) && is_finite(s.rhs.B) && is_finite(s.cache.E) &&
                is_finite(s.cache.phi) && is_finite(s.current);
  bool hermitian = true;
  const auto physical = [&](const Vec &values, std::size_t E, double reference,
                            const std::string &component,
                            const std::string &unit,
                            const std::string &population = "") {
    Check x;
    x.name = "hermitian_rhs_component";
    x.component = component;
    x.population = population;
    x.tag = t;
    x.norm = Norm::value;
    x.unit = x.value_unit = unit;
    x.available = is_finite(values);
    x.reference = reference;
    x.selected = values.size();
    if (x.available) {
      x.raw = Fourier(c.p.K, c.N).hermitian_defect(values, E);
      x.threshold = 512 * std::numeric_limits<double>::epsilon() *
                    std::max(max_abs(values), reference);
    }
    x.passed = x.available && x.raw <= x.threshold;
    out.push_back(x);
    return x.passed;
  };
  for (int a = 0; a < 2; ++a) {
    const auto &r = s.rhs.fluid[a];
    const auto &v = u.fluid[a];
    finite &= is_finite(v.n) && is_finite(v.mom) && is_finite(v.perp) &&
              is_finite(v.parallel) && is_finite(r.n) && is_finite(r.mom) &&
              is_finite(r.perp) && is_finite(r.parallel);
    const auto id = c.p.species[a].id;
    const auto &ref = c.p.ref;
    hermitian &= physical(r.n, nc, (a ? ref.ne : ref.ni) / ref.time, "n",
                          "1/(m^3*s)", id);
    hermitian &= physical(r.mom, 3 * nc,
                          c.p.species[a].mass * (a ? ref.ne : ref.ni) *
                              ref.velocity / ref.time,
                          "momentum", "kg/(m^2*s^2)", id);
    hermitian &=
        physical(r.perp, nc, ref.pressure / ref.time, "perp", "Pa/s", id);
    if (a)
      hermitian &= physical(r.parallel, nc, ref.pressure / ref.time, "parallel",
                            "Pa/s", id);
  }
  hermitian &=
      physical(s.rhs.B, c.op.Nb, c.p.ref.magnetic / c.p.ref.time, "B", "T/s");
  truth(out, "finite", finite, t);
  truth(out, "hermitian_rhs", hermitian, t);
  truth(out, "source_identity",
        c.p.species[0].id != c.p.species[1].id && s.source_id != 0 &&
            s.cache.tag == t && s.fluid[0].wall.tag == t &&
            s.fluid[1].wall.tag == t && u.step == t.step &&
            u.generation == t.generation && u.time == t.time,
        t);
  truth(out, "memory_bound",
        c.p.memory_limit > 0 && estimated_peak_bytes(c) <= c.p.memory_limit, t);
  const auto minimum = [&](const std::string &name, const Vec &values,
                           double floor, const std::string &unit,
                           const std::string &pop = "") {
    Check x;
    x.name = name;
    x.population = pop;
    x.tag = t;
    x.unit = x.value_unit = unit;
    x.norm = Norm::value;
    x.raw = std::numeric_limits<double>::infinity();
    x.threshold = floor;
    x.reference = floor;
    x.selected = values.size();
    for (std::size_t i = 0; i < values.size(); ++i)
      if (values[i].real() < x.raw) {
        x.raw = values[i].real();
        x.worst = i;
      }
    x.available = !values.empty() && is_finite(values);
    x.passed = x.available && x.raw > floor;
    out.push_back(x);
  };
  Vec magnetic(c.N * nc);
  for (std::size_t i = 0; i < magnetic.size(); ++i) {
    auto l = i / nc, j = i % nc;
    magnetic[i] = std::hypot(s.BN[l * 3 * nc + j].real(),
                             s.BN[l * 3 * nc + nc + j].real(),
                             s.BN[l * 3 * nc + 2 * nc + j].real());
  }
  minimum("minimum_B", magnetic, c.p.B_min, "T");
  const auto remember = [&](const std::string &name, Vec r, const RealVec &w,
                            const std::string &unit,
                            const std::string &pop = "", bool samples = false) {
    measure(out, name, r, w, unit, t, c, pop, samples);
    s.residuals[name + (pop.empty() ? "" : "." + pop)] = std::move(r);
  };
  remember("poisson_linear", s.cache.poisson_residual, c.op.M0, "V/m^2");
  remember("darwin_dynamic", s.residuals.at("darwin_dynamic"), c.op.M1,
           "A/(m^2*s)");
  remember("longitudinal_projection", s.residuals.at("longitudinal_projection"),
           c.op.M0, "A/m^3");
  remember("gauss",
           subtract(mode_apply(c.op.DJ, s.cache.EL, c.compute),
                    scaled(s.rho, 1 / c.p.epsilon0)),
           c.op.M0, "V/m^2");
  remember("transverse_divergence", mode_apply(c.op.DJ, s.cache.ET, c.compute),
           c.op.M0, "V/m^2");
  remember("current_divB", mode_apply(c.op.DB, u.B, c.compute), c.geometry.wv,
           "T/m");
  remember("current_ampere", s.ampere, c.op.M1, "T/m");
  remember("CT_increment", mode_apply(c.op.DB, s.rhs.B, c.compute),
           c.geometry.wv, "T/(m*s)");
  Vec actual(s.current.size());
  for (int a = 0; a < 2; ++a) {
    const auto &f = s.fluid[a];
    const auto &st = s.terms[a];
    const auto &sp = c.p.species[a];
    const auto &L = s.ledger[a];
    const auto id = sp.id;
    minimum("minimum_n", f.n, sp.n_min, "1/m^3", id);
    minimum("minimum_face_n", f.nf, sp.face_min, "1/m^3", id);
    minimum("minimum_perp", f.perp, sp.p_min, "Pa", id);
    if (a)
      minimum("minimum_parallel", f.parallel, sp.parallel_min, "Pa", id);
    remember("stage_continuity", add(s.rhs.fluid[a].n, f.divergence_hat),
             c.op.M0, "1/(m^3*s)", id);
    remember("wall_state",
             sample_apply(c.op.T_full, inverse(u.fluid[a].mom, 3 * nc, c), c.N,
                          c.compute),
             RealVec(c.op.T_full.rows, 1), "kg/(m^2*s)", id, true);
    remember("wall_rhs", sample_apply(c.op.T_full, st.force, c.N, c.compute),
             RealVec(c.op.T_full.rows, 1), "kg/(m^2*s^2)", id, true);
    remember("wall_multiplier", st.multiplier_residual, RealVec(c.op.T.rows, 1),
             "kg/(m^2*s^2)", id, true);
    Vec wall_n(c.N * c.geometry.walls.size()), wall_perp(wall_n.size()),
        wall_parallel(wall_n.size());
    for (int l = 0; l < c.N; ++l)
      for (std::size_t j = 0; j < c.geometry.walls.size(); ++j) {
        const auto &wall = c.geometry.walls[j];
        const auto edge = wall.component == WallComponent::r_face
                              ? wall.face
                              : c.geometry.r_faces.keys.size() + nc + wall.face;
        const auto i = l * c.geometry.walls.size() + j,
                   index = l * c.op.Ne + edge;
        wall_n[i] = static_cast<double>(wall.sign) * f.flux[index];
        wall_perp[i] = static_cast<double>(wall.sign) * f.pressure_flux[index];
        if (a)
          wall_parallel[i] =
              static_cast<double>(wall.sign) * f.parallel_flux[index];
      }
    const RealVec wall_weights(c.geometry.walls.size(), 1);
    remember("wall_number_flux", wall_n, wall_weights, "1/(m^2*s)", id, true);
    remember("wall_perp_flux", wall_perp, wall_weights, "Pa*m/s", id, true);
    if (a)
      remember("wall_parallel_flux", wall_parallel, wall_weights, "Pa*m/s", id,
               true);
    scalar_check(out, "electric_pairing", L.electric_node - L.electric_edge,
                 "W", t, c, id);
    scalar_check(out, "pressure_pairing",
                 L.pressure_kinetic + L.pressure_internal, "W", t, c, id);
    scalar_check(out, "magnetic_work", L.magnetic, "W", t, c, id);
    scalar_check(out, "wall_work", L.wall, "W", t, c, id);
    scalar_check(out, "convection_pairing", L.convection, "W", t, c, id);
    scalar_check(out, "spectral_transfer_defect",
                 L.spectral_kinetic + L.spectral_perp + L.spectral_parallel,
                 "W", t, c, id);
    Vec vu = sample_apply(c.op.I, f.u, c.N, c.compute);
    // V(n) consumes n*u; interpolation of u alone would break theta.
    Vec nu = f.u;
    for (int l = 0; l < c.N; ++l)
      for (std::size_t j = 0; j < 3 * nc; ++j)
        nu[l * 3 * nc + j] *= f.n[l * nc + j % nc];
    vu = sample_apply(c.op.I, nu, c.N, c.compute);
    for (std::size_t j = 0; j < vu.size(); ++j)
      vu[j] /= f.nf[j];
    remember("theta_identity", subtract(f.theta, divergence(vu, c)), c.op.M0,
             "1/s", id, true);
    if (a) {
      Vec r(f.theta.size());
      for (std::size_t i = 0; i < r.size(); ++i) {
        Complex pq{};
        const auto l = i / nc, j = i % nc;
        for (int ab = 0; ab < 9; ++ab)
          pq += f.tensor[l * 9 * nc + ab * nc + j] *
                f.gradient[l * 9 * nc + ab * nc + j];
        r[i] = f.perp[i] * (f.strain[i] - f.theta[i]) -
               f.parallel[i] * f.strain[i] + pq;
      }
      remember("cgl_source", r, c.op.M0, "Pa/s", id, true);
    }
    const auto fluxdot =
        sample_apply(c.op.I, s.rhs.fluid[a].mom, 2 * c.p.K + 1, c.compute);
    actual = add(actual, scaled(fluxdot, sp.charge / sp.mass));
  }
  remember("current_response",
           subtract(actual, add(s.fJ0, c.compute.mv(s.AE, s.cache.E))), c.op.M1,
           "A/(m^2*s)");
  gate_records(out, stage_requirements(c, t));
}

StepResult check_step(const State &left, const StageResult &base,
                      const State &midU, const StageResult &mid,
                      const State &right, const StageResult &end,
                      const Cumulative &cumulative, const Context &c,
                      const State *prescribed_source) {
  StepResult result;
  auto &out = result.checks;
  const auto &tag = end.tag;
  const double dt = c.p.dt;
  auto interval_tag = tag;
  interval_tag.stage = "interval";
  const auto m = [&](const std::string &name, const Vec &r, const RealVec &w,
                     const std::string &unit, const std::string &pop = "") {
    measure(out, name, r, w, unit, interval_tag, c, pop);
  };
  const auto scalar = [&](const std::string &name, double v,
                          const std::string &unit,
                          const std::string &pop = "") {
    scalar_check(out, name, v, unit, interval_tag, c, pop);
  };
  result.interval = {};
  result.interval.source_id = mid.source_id;
  result.interval.trial = mid.tag.trial;
  result.interval.left_step = left.step;
  result.interval.right_step = right.step;
  result.interval.weight = dt;
  result.interval.left_time = left.time;
  result.interval.right_time = right.time;
  Vec charge_change(base.rho.size()), flux_div(base.rho.size());
  const auto &TL = base.totals;
  const auto &TR = end.totals;
  const auto sources = prescribed_source
      ? verification::source_rates(*prescribed_source, midU, mid, c)
      : verification::SourceRates{};
  for (int a = 0; a < 2; ++a) {
    const auto &sp = c.p.species[a];
    const auto &L = mid.ledger[a];
    const auto id = sp.id;
    auto ndiff = scaled(subtract(right.fluid[a].n, left.fluid[a].n), 1 / dt);
    if (prescribed_source) ndiff = subtract(ndiff, prescribed_source->fluid[a].n);
    m("species_continuity", add(ndiff, mid.fluid[a].divergence_hat), c.op.M0,
      "1/(m^3*s)", id);
    charge_change = add(charge_change, scaled(ndiff, sp.charge));
    flux_div = add(flux_div, scaled(mid.fluid[a].divergence_hat, sp.charge));
    result.interval.number_flux[a] = scaled(mid.fluid[a].flux_hat, dt);
    result.interval.current[a] =
        scaled(result.interval.number_flux[a], sp.charge);
    scalar("total_number_balance",
           TR.number[a] - TL.number[a] + dt * (mid.boundary.number[a] - sources.totals.number[a]), "1", id);
    scalar("kinetic_energy",
           TR.kinetic[a] - TL.kinetic[a] + dt * mid.boundary.kinetic[a] -
               dt *
                   (L.electric_edge + L.pressure_kinetic + L.wall + L.magnetic + sources.totals.kinetic[a]),
           "J", id);
  }
  m("RC_flux", add(charge_change, flux_div), c.op.M0, "C/(m^3*s)");
  const auto djJ = mode_apply(c.op.DJ, mid.current, c.compute);
  m("RC_mom", add(charge_change, djJ), c.op.M0, "C/(m^3*s)");
  m("deltaC", subtract(djJ, flux_div), c.op.M0, "C/(m^3*s)");
  auto elchange = scaled(subtract(end.cache.EL, base.cache.EL), 1 / dt);
  if (prescribed_source) elchange = subtract(elchange, sources.longitudinal_rate);
  m("RL",
    add(longitudinal(mid.current, c), scaled(elchange, c.p.epsilon0)),
    c.op.M1, "A/m^2");
  auto bchange = subtract(right.B, left.B);
  if (prescribed_source) bchange = subtract(bchange, scaled(prescribed_source->B, dt));
  m("CT_update",
    add(scaled(bchange, 1 / dt), mode_apply(c.op.C, mid.cache.ET, c.compute)),
    c.op.MB, "T/s");
  m("divB_update", mode_apply(c.op.DB, bchange, c.compute), c.geometry.wv,
    "T/m");
  m("long_term_divB_change",
    subtract(mode_apply(c.op.DB, right.B, c.compute), cumulative.baseline.divB),
    c.geometry.wv, "T/m");
  scalar("topology_increment", TR.topology - TL.topology - dt*sources.totals.topology, "Wb");
  double charge = 0;
  for (int a = 0; a < 2; ++a)
    charge += c.p.species[a].charge *
              (TR.number[a] - TL.number[a] + dt * (mid.boundary.number[a] - sources.totals.number[a]));
  scalar("total_charge_balance", charge, "C");
  scalar("ion_internal",
         TR.ion_internal - TL.ion_internal +
             dt * mid.boundary.internal_perp[0] - dt * (mid.ledger[0].perp_rate + sources.totals.ion_internal),
         "J");
  scalar("electron_perp",
         TR.electron_perp - TL.electron_perp +
             dt * mid.boundary.internal_perp[1] - dt * (mid.ledger[1].perp_rate + sources.totals.electron_perp),
         "J");
  scalar("electron_parallel",
         TR.electron_parallel - TL.electron_parallel +
             dt * mid.boundary.internal_parallel[1] -
             dt * (mid.ledger[1].parallel_rate + sources.totals.electron_parallel),
         "J");
  double PB = 0, PL = 0;
  for (std::size_t j = 0; j < midU.B.size(); ++j)
    PB += tau * c.op.MB[j % c.op.MB.size()] *
          std::real(std::conj(midU.B[j]) * mid.rhs.B[j]) / c.p.mu0;
  for (std::size_t j = 0; j < mid.cache.EL.size(); ++j)
    PL += tau * c.p.epsilon0 * c.op.M1[j % c.op.M1.size()] *
          std::real(std::conj(mid.cache.EL[j]) * mid.ELdot[j]);
  scalar("field_energy_B", TR.magnetic - TL.magnetic - dt * (PB + sources.totals.magnetic), "J");
  scalar("field_energy_L", TR.longitudinal - TL.longitudinal - dt * (PL + sources.totals.longitudinal), "J");
  scalar("field_exchange",
         PB + PL + mid.ledger[0].electric_edge + mid.ledger[1].electric_edge +
             mid.boundary.field_energy,
         "W");
  scalar("total_energy", TR.energy - TL.energy + dt * (mid.boundary.total_energy - sources.totals.energy),
         "J");
  for (int d = 0; d < 3; ++d)
    scalar("momentum_" + std::string(1, "xyz"[d]),
           end.momentum.value[d] - base.momentum.value[d] +
               dt * (mid.momentum.outward[d] - sources.momentum.value[d]),
           "kg*m/s");
  scalar("angular_momentum",
         end.momentum.angular - base.momentum.angular +
             dt * (mid.momentum.outward_torque - sources.momentum.angular),
         "kg*m^2/s");
  result.totals = end.totals;
  result.momentum = end.momentum;
  gate(out, {"species_continuity",
             "RC_flux",
             "RC_mom",
             "deltaC",
             "RL",
             "CT_update",
             "divB_update",
             "long_term_divB_change",
             "topology_increment",
             "total_charge_balance",
             "total_number_balance",
             "kinetic_energy",
             "ion_internal",
             "electron_perp",
             "electron_parallel",
             "field_energy_B",
             "field_energy_L",
             "field_exchange",
             "total_energy",
             "momentum_x",
             "momentum_y",
             "momentum_z",
             "angular_momentum"});
  return result;
}
StepResult check_step(const State &left, const StageResult &base,
                      const State &midpoint, const StageResult &mid,
                      const State &right, const StageResult &end,
                      const Cumulative &cumulative, const Context &c) {
  return check_step(left, base, midpoint, mid, right, end, cumulative, c, nullptr);
}
} // namespace fomhd
