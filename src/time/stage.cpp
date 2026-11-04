#include "fomhd/backend.hpp"
#include "fomhd/diagnostics.hpp"
#include "fomhd/physics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace fomhd {
namespace {
std::uint64_t fresh(std::uint64_t &value) {
  if (value == std::numeric_limits<std::uint64_t>::max())
    throw Failure("CounterOverflow", "stage identity exhausted");
  return value++;
}
bool same_check(const Check &a, const Check &b) {
  return a.name == b.name && a.population == b.population &&
         a.component == b.component && a.producer == b.producer &&
         a.region == b.region && a.norm == b.norm && a.tag == b.tag;
}
void append_missing(std::vector<Check> &target,
                    const std::vector<Check> &source) {
  for (const auto &check : source)
    if (std::none_of(target.begin(), target.end(),
                     [&](const Check &x) { return same_check(x, check); }))
      target.push_back(check);
}
} // namespace
void retain_failure_evidence(Failure &failure, const StageTag &tag,
                             const Context &c, const State *u,
                             const StageResult *s, const std::string &prefix) {
  if (!failure.stage)
    failure.stage = tag;
  if (s) {
    append_missing(failure.checks, s->checks);
    for (const auto &[key, values] : s->residuals)
      if (!failure.partial_arrays.contains(prefix + "." + key))
        failure.partial_arrays[prefix + "." + key] = values;
    if (!s->cache.E.empty())
      failure.partial_arrays.try_emplace(prefix + "_E", s->cache.E);
  }
  if (u) {
    failure.partial_arrays.try_emplace(prefix + "_B", u->B);
    for (int a = 0; a < 2; ++a) {
      const std::string id = a ? "e" : "i";
      failure.partial_arrays.try_emplace(prefix + "_n_" + id, u->fluid[a].n);
      failure.partial_arrays.try_emplace(prefix + "_momentum_" + id,
                                         u->fluid[a].mom);
      failure.partial_arrays.try_emplace(prefix + "_p_" + id,
                                         u->fluid[a].perp);
      if (a)
        failure.partial_arrays.try_emplace(prefix + "_p_parallel_e",
                                           u->fluid[a].parallel);
    }
  }
  auto required = stage_requirements(c, tag);
  for (auto &check : required)
    if (std::none_of(failure.checks.begin(), failure.checks.end(),
                     [&](const Check &x) { return same_check(x, check); })) {
      check.available = false;
      check.passed = false;
      check.reason = "not run: " + failure.code;
      failure.checks.push_back(std::move(check));
    }
}
StageResult evaluate_stage(const State &u, const StageTag &tag, Context &c,
                           bool alias) {
  StageResult s;
  s.tag=tag;
  try {
  if(tag.step!=u.step||tag.generation!=u.generation||tag.time!=u.time)
    throw Failure("StaleStage","stage identity does not match state");
  check_retained(u,c);
  s.source_id = fresh(c.next_source);
  s.rhs = allocate_state(c);
  s.rhs.time = u.time;
  s.rhs.step = u.step;
  s.rhs.generation = u.generation;
  const auto nc = c.op.Nc, ne = c.op.M1.size(),
             M = static_cast<std::size_t>(2 * c.p.K + 1);
  s.BN = inverse(sample_apply(c.op.QB, u.B, static_cast<int>(M), c.compute),
                 3 * nc, c);
  s.bhat.resize(s.BN.size());
  for (int l = 0; l < c.N; ++l)
    for (std::size_t j = 0; j < nc; ++j) {
      const double magnitude = std::hypot(s.BN[l * 3 * nc + j].real(),
                                          s.BN[l * 3 * nc + nc + j].real(),
                                          s.BN[l * 3 * nc + 2 * nc + j].real());
      if (!std::isfinite(magnitude) || magnitude <= c.p.B_min)
        throw Failure("Magnetization", "nodal B below declared floor");
      for (int d = 0; d < 3; ++d)
        s.bhat[l * 3 * nc + d * nc + j] =
            s.BN[l * 3 * nc + d * nc + j] / magnitude;
    }
  s.rho.resize(M * nc);
  s.current.resize(M * ne);
  s.fJ0.resize(M * ne);
  for (int a = 0; a < 2; ++a) {
    s.fluid[a] = build_fluid(u.fluid[a], a, s.BN, s.bhat, tag, c);
    build_non_electric(s.fluid[a], a, s.BN, s.bhat, c);
    const auto projected =
        project_wall(s.fluid[a].wall, s.fluid[a].non_electric, c);
    const auto current_rate = sample_apply(
        c.op.I,
        scaled(projected.value, c.p.species[a].charge / c.p.species[a].mass),
        c.N, c.compute);
    s.fJ0 = add(s.fJ0, forward(current_rate, ne, c));
    s.rho = add(s.rho, scaled(u.fluid[a].n, c.p.species[a].charge));
    s.current =
        add(s.current, scaled(s.fluid[a].flux_hat, c.p.species[a].charge));
  }
  s.cache = electrostatic(s.rho, tag, c);
  transverse(s, c);
  for (int a = 0; a < 2; ++a) {
    finish_fluid(s, a, c);
    s.ledger[a] = power_ledger(s, a, c);
  }
  s.rhs.B = scaled(mode_apply(c.op.C, s.cache.ET, c.compute), -1);
  const auto JL = longitudinal(s.current, c);
  s.ELdot = scaled(JL, -1 / c.p.epsilon0);
  s.residuals["longitudinal_projection"] =
      mode_apply(c.op.DJ, subtract(s.current, JL), c.compute);
  s.phidot.resize(M * nc);
  const auto dj = mode_apply(c.op.DJ, s.current, c.compute);
  for (std::size_t k = 0; k < M; ++k) {
    auto result = c.compute.linear(
        c.op.Lphi[k], scaled(slice(dj, k * nc, nc), -1 / c.p.epsilon0),
        RealVec(nc, c.p.ref.current / (c.p.ref.length * c.p.epsilon0)),
        RealVec(nc, c.p.ref.electric * c.p.ref.length / c.p.ref.time),
        c.p.rank_rtol, 64 * nc * std::numeric_limits<double>::epsilon());
    put(s.phidot, k * nc, result.x);
  }
  s.ampere = subtract(mode_apply(c.op.Cdag, u.B, c.compute),
                      scaled(subtract(s.current, JL), c.p.mu0));
  s.boundary = boundary_ledger(u, s, c);
  s.totals = state_totals(u, s, c);
  s.momentum = momentum(u, s, c);
  check_stage(u, s, c);
  if (alias) {
    Context alt = c;
    alt.N = 2 * c.N + 1;
    alt.spectral_id = c.spectral_id + 1;
    auto at = tag;
    at.stage = "alias";
    at.spectral_context = alt.spectral_id;
    auto other = evaluate_stage(u, at, alt, false);
    c.next_source = alt.next_source;
    s.checks.insert(s.checks.end(), other.checks.begin(), other.checks.end());
    for (int a = 0; a < 2; ++a) {
      const auto id = c.p.species[a].id;
      measure(s.checks, "alias_n",
              subtract(other.rhs.fluid[a].n, s.rhs.fluid[a].n), c.op.M0,
              "1/(m^3*s)", tag, c, id);
      for (int d = 0; d < 3; ++d) {
        Vec residual(M * nc);
        for (std::size_t k = 0; k < M; ++k)
          put(residual, k * nc,
              subtract(slice(other.rhs.fluid[a].mom, (k * 3 + d) * nc, nc),
                       slice(s.rhs.fluid[a].mom, (k * 3 + d) * nc, nc)));
        measure(s.checks, "alias_mom", residual, c.op.M0, "kg/(m^2*s^2)", tag,
                c, id, false,
                d == 0   ? "R"
                : d == 1 ? "phi"
                         : "Z");
      }
      measure(s.checks, "alias_perp",
              subtract(other.rhs.fluid[a].perp, s.rhs.fluid[a].perp), c.op.M0,
              "Pa/s", tag, c, id);
      if (a)
        measure(s.checks, "alias_parallel",
                subtract(other.rhs.fluid[a].parallel, s.rhs.fluid[a].parallel),
                c.op.M0, "Pa/s", tag, c, id);
    }
    const std::array<std::size_t, 4> offsets{
        0, c.geometry.z_faces.keys.size(),
        c.geometry.z_faces.keys.size() + c.geometry.vertices.keys.size(),
        c.op.Nb};
    for (int d = 0; d < 3; ++d) {
      const auto n = offsets[d + 1] - offsets[d];
      Vec residual(M * n);
      for (std::size_t k = 0; k < M; ++k)
        put(residual, k * n,
            subtract(slice(other.rhs.B, k * c.op.Nb + offsets[d], n),
                     slice(s.rhs.B, k * c.op.Nb + offsets[d], n)));
      RealVec weights(c.op.MB.begin() + offsets[d],
                      c.op.MB.begin() + offsets[d + 1]);
      measure(s.checks, "alias_B", residual, weights, "T/s", tag, c, "", false,
              d == 0   ? "R"
              : d == 1 ? "phi"
                       : "Z");
    }
    gate(s.checks,
         {"alias_n", "alias_mom", "alias_perp", "alias_parallel", "alias_B"});
  }
  return s;
  } catch(Failure& failure) {
    retain_failure_evidence(failure, tag, c, &u, &s, "trial");
    throw;
  } catch(const NumericalError& error) {
    Failure failure("NumericalLibrary", error.what());
    retain_failure_evidence(failure, tag, c, &u, &s, "trial");
    throw failure;
  }
}
Committed initialize(State u, Context &c, Cumulative &cumulative,
                     bool equilibrium, StageResult *captured) {
  if (u.generation == std::numeric_limits<std::uint64_t>::max())
    throw Failure("CounterOverflow", "state generation exhausted");
  c.next_generation = std::max(c.next_generation, u.generation + 1);
  const StageTag tag{
      u.step,        u.generation, fresh(c.next_trial),
      c.spectral_id, u.time,       u.step ? "restart" : "initial"};
  auto s = evaluate_stage(u, tag, c, true);
  try {
    for (const auto &evidence : c.op.setup_checks) {
      Check check;
      check.name = evidence.name + ".mode_" + std::to_string(evidence.mode);
      check.tag = tag;
      check.norm = Norm::value;
      check.unit = check.value_unit = evidence.unit;
      check.raw = evidence.residual;
      check.reference = evidence.scale;
      check.threshold = evidence.roundoff_bound;
      check.available = true;
      check.passed = check.raw <= check.threshold;
      s.checks.push_back(std::move(check));
    }
    measure(s.checks, "initial_divB", mode_apply(c.op.DB, u.B, c.compute),
            c.geometry.wv, "T/m", tag, c);
    measure(s.checks, "initial_ampere", s.ampere, c.op.M1, "T/m", tag, c);
    if (equilibrium) {
      for (int a = 0; a < 2; ++a) {
        const auto id = c.p.species[a].id;
        measure(s.checks, "equilibrium_n", s.rhs.fluid[a].n, c.op.M0,
                "1/(m^3*s)", tag, c, id);
        RealVec weights;
        for (int d = 0; d < 3; ++d)
          weights.insert(weights.end(), c.op.M0.begin(), c.op.M0.end());
        measure(s.checks, "equilibrium_mom", s.rhs.fluid[a].mom, weights,
                "kg/(m^2*s^2)", tag, c, id);
        measure(s.checks, "equilibrium_perp", s.rhs.fluid[a].perp, c.op.M0,
                "Pa/s", tag, c, id);
        if (a)
          measure(s.checks, "equilibrium_parallel", s.rhs.fluid[a].parallel,
                  c.op.M0, "Pa/s", tag, c, id);
      }
      measure(s.checks, "equilibrium_B", s.rhs.B, c.op.MB, "T/s", tag, c);
    }
    gate(s.checks, {"initial_divB", "initial_ampere"});
    cumulative.baseline = {mode_apply(c.op.DB, u.B, c.compute), s.totals,
                           s.momentum, u.time, u.step};
    cumulative.last_step = u.step;
    cumulative.last_time = u.time;
    Committed result{std::move(u), s.cache, true};
    if (captured)
      *captured = std::move(s);
    return result;
  } catch (Failure &failure) {
    retain_failure_evidence(failure, tag, c, &u, &s, "trial");
    throw;
  } catch (const NumericalError &error) {
    Failure failure("NumericalLibrary", error.what());
    retain_failure_evidence(failure, tag, c, &u, &s, "trial");
    throw failure;
  }
}
State closed_reference(const Context &c, double n, double pi, double pe,
                       double B, double velocity) {
  if (n <= 0 || pi <= 0 || pe <= 0 || B <= 0)
    throw Failure("Fixture", "invalid reference parameters");
  auto u = allocate_state(c);
  const auto nc = c.op.Nc, nb = c.op.MB.size();
  for (int a = 0; a < 2; ++a) {
    const double na =
        a ? -c.p.species[0].charge * n / c.p.species[1].charge : n;
    for (std::size_t j = 0; j < nc; ++j) {
      u.fluid[a].n[c.p.K * nc + j] = na;
      u.fluid[a].perp[c.p.K * nc + j] = a ? pe : pi;
      if (a)
        u.fluid[a].parallel[c.p.K * nc + j] = pe;
      if (velocity && c.p.K) {
        for (int k : {-1, 1})
          u.fluid[a].mom[(c.p.K + k) * 3 * nc + nc + j] =
              c.p.species[a].mass * na * velocity / 2;
      }
    }
  }
  Vec harmonic(nb);
  for (std::size_t j = 0; j < nb; ++j)
    harmonic[j] = std::conj(c.op.topology_flux_row(0, j)) / c.op.MB[j];
  harmonic = scaled(harmonic, B / max_abs(c.compute.mv(c.op.QB, harmonic)));
  put(u.B, c.p.K * nb, harmonic);
  u.time = c.p.start_time;
  u.generation = 1;
  return u;
}
} // namespace fomhd
