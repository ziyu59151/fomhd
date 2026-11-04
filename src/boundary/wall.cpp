#include "fomhd/backend.hpp"
#include "fomhd/physics.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace fomhd {
WallContext build_wall(const Vec &n, const Species &s, const StageTag &tag,
                       const Context &c) {
  WallContext w;
  w.tag = tag;
  const auto nc = c.op.Nc;
  if (n.size() != static_cast<std::size_t>(c.N) * nc)
    throw Failure("Shape", "wall density shape");
  for (int l = 0; l < c.N; ++l) {
    WallSample a;
    a.winv.resize(3 * nc);
    for (std::size_t j = 0; j < 3 * nc; ++j) {
      const double nj = n[l * nc + j % nc].real();
      if (!std::isfinite(nj) || nj <= s.n_min)
        throw Failure("Domain", "wall density not positive");
      a.winv[j] = s.mass * nj / c.op.M0[j % nc];
    }
    a.S = c.compute.mm(c.op.T, c.compute.mm(diagonal(a.winv), adjoint(c.op.T)));
    const double nr = s.id == c.p.species[0].id ? c.p.ref.ni : c.p.ref.ne;
    a.row_scale = s.mass * nr * c.p.ref.velocity / c.p.ref.time;
    a.col_scale = a.row_scale / (s.mass * nr / std::pow(c.p.ref.length, 3));
    if (a.S.rows) {
      Matrix scaledS = a.S;
      for (auto &z : scaledS.data)
        z *= a.col_scale / a.row_scale;
      const auto sv = singular_values(scaledS);
      if (sv.empty() || sv.back() <= c.p.rank_rtol * sv.front())
        throw Failure("WallRank", "scaled wall matrix deficient");
      a.lu = factor(scaledS);
    }
    w.samples.push_back(std::move(a));
  }
  return w;
}
Projection project_wall(const WallContext &w, const Vec &r, const Context &c) {
  const auto d = 3 * c.op.Nc, t = c.op.T.rows;
  if (r.size() != c.N * d || w.samples.size() != static_cast<std::size_t>(c.N))
    throw Failure("Shape", "wall response shape");
  Projection out;
  out.value = r;
  out.reaction.resize(r.size());
  out.multiplier.resize(c.N * t);
  out.multiplier_residual.resize(c.N * t);
  for (int l = 0; l < c.N; ++l) {
    if (!t)
      continue;
    const auto &a = w.samples[l];
    const auto rhs = c.compute.mv(c.op.T, slice(r, l * d, d));
    Vec lam;
    if (c.compute.name() == "cpu")
      lam = scaled(solve(a.lu, scaled(rhs, 1 / a.row_scale)), a.col_scale);
    else
      lam = c.compute
                .linear(a.S, rhs, RealVec(t, a.row_scale),
                        RealVec(t, a.col_scale), c.p.rank_rtol, 1e-10)
                .x;
    auto reaction = c.compute.mv(adjoint(c.op.T), lam);
    for (std::size_t j = 0; j < d; ++j) {
      reaction[j] *= -a.winv[j];
      out.value[l * d + j] += reaction[j];
    }
    put(out.reaction, l * d, reaction);
    put(out.multiplier, l * t, lam);
    put(out.multiplier_residual, l * t, subtract(c.compute.mv(a.S, lam), rhs));
  }
  return out;
}

BoundaryLedger boundary_ledger(const State &u, const StageResult &s,
                               const Context &c) {
  BoundaryLedger out;
  const auto nc = c.op.Nc, nr = c.geometry.r_faces.keys.size(),
             nz = c.geometry.z_faces.keys.size();
  const auto nv = c.geometry.vertices.keys.size(), ne = c.op.M1.size(),
             nb = c.op.MB.size();
  const auto Bs = inverse(u.B, nb, c),
             EL = inverse(s.cache.EL, ne, c, true, c.p.ref.electric),
             ET = inverse(s.cache.ET, ne, c, true, c.p.ref.electric);
  const auto nw = c.geometry.walls.size();
  out.traces.resize(c.N * nw);
  for (int l = 0; l < c.N; ++l)
    for (std::size_t w = 0; w < nw; ++w) {
      const auto &wall = c.geometry.walls[w];
      auto &tr = out.traces[l * nw + w];
      const bool radial = wall.component == WallComponent::r_face;
      const auto edge = radial ? wall.face : nr + nc + wall.face;
      const int dir = radial ? 0 : 2;
      const double sigma = wall.sign;
      tr.B[1] = 0.5 * (Bs[l * nb + nz + wall.vertices[0]].real() +
                       Bs[l * nb + nz + wall.vertices[1]].real());
      if (radial)
        tr.B[2] = Bs[l * nb + nz + nv + wall.face].real();
      else
        tr.B[0] = Bs[l * nb + wall.face].real();
      tr.EL[dir] = EL[l * ne + edge].real();
      tr.ET[dir] = ET[l * ne + edge].real();
      const double bmag = std::hypot(tr.B[0], tr.B[1], tr.B[2]);
      if (!(bmag > c.p.B_min))
        throw Failure("WallMagnetization",
                      "wall magnetic magnitude below declared floor");
      const double area = wall.area_per_radian * 2 * std::numbers::pi / c.N;
      for (int a = 0; a < 2; ++a) {
        const auto &f = s.fluid[a];
        const auto &sp = c.p.species[a];
        double density = 0, perp = 0, parallel = 0;
        std::array<double, 3> mom{};
        for (std::size_t j = 0; j < nc; ++j) {
          const double weight = c.op.I(edge, dir * nc + j).real();
          density += weight * f.n[l * nc + j].real();
          perp += weight * f.perp[l * nc + j].real();
          if (a)
            parallel += weight * f.parallel[l * nc + j].real();
          for (int d = 0; d < 3; ++d)
            mom[d] += weight * f.mom[l * 3 * nc + d * nc + j].real();
        }
        if (!(density > sp.face_min))
          throw Failure("WallDensity", "wall trace below density floor");
        tr.density[a] = density;
        tr.perp[a] = perp;
        tr.parallel[a] = a ? parallel : perp;
        double speed2 = 0;
        for (int d = 0; d < 3; ++d) {
          tr.velocity[a][d] = mom[d] / (sp.mass * density);
          speed2 += tr.velocity[a][d] * tr.velocity[a][d];
        }
        const double un = sigma * tr.velocity[a][dir],
                     Fn = sigma * f.flux[l * ne + edge].real();
        tr.normal_flux[a] = Fn;
        const double kinetic = sp.mass * density * speed2 / 2,
                     internal = a ? perp + parallel / 2 : perp / (sp.gamma - 1);
        double pressure_work = perp * un;
        if (a) {
          double uB = 0;
          for (int d = 0; d < 3; ++d)
            uB += tr.velocity[a][d] * tr.B[d] / bmag;
          pressure_work += (parallel - perp) * uB * sigma * tr.B[dir] / bmag;
        }
        out.number[a] += area * Fn;
        out.kinetic[a] += area * kinetic * un;
        const double fluxp = sigma * f.pressure_flux[l * ne + edge].real();
        out.internal_perp[a] += area * fluxp / (a ? 1 : sp.gamma - 1);
        if (a)
          out.internal_parallel[a] +=
              area * sigma * f.parallel_flux[l * ne + edge].real() / 2;
        out.total_energy += area * ((kinetic + internal) * un + pressure_work);
      }
      const std::array<double, 3> E{tr.EL[0] + tr.ET[0], tr.EL[1] + tr.ET[1],
                                    tr.EL[2] + tr.ET[2]};
      const double cross_normal = dir == 0 ? E[1] * tr.B[2] - E[2] * tr.B[1]
                                           : E[0] * tr.B[1] - E[1] * tr.B[0];
      const double field = area * sigma * cross_normal / c.p.mu0;
      out.field_energy += field;
      out.total_energy += field;
    }
  return out;
}
} // namespace fomhd
