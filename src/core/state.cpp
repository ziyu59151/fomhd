#include "fomhd/backend.hpp"
#include "fomhd/physics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace fomhd {
Vec slice(const Vec &v, std::size_t off, std::size_t n) {
  if (off > v.size() || n > v.size() - off)
    throw Failure("Shape", "slice outside array");
  return {v.begin() + static_cast<std::ptrdiff_t>(off),
          v.begin() + static_cast<std::ptrdiff_t>(off + n)};
}
void put(Vec &v, std::size_t off, const Vec &x) {
  if (off > v.size() || x.size() > v.size() - off)
    throw Failure("Shape", "write outside array");
  std::copy(x.begin(), x.end(), v.begin() + static_cast<std::ptrdiff_t>(off));
}
Vec repeat_weights(const RealVec &w, std::size_t count) {
  Vec out(checked_product(w.size(), count, "weights"));
  for (std::size_t l = 0; l < count; ++l)
    std::copy(w.begin(), w.end(),
              out.begin() + static_cast<std::ptrdiff_t>(l * w.size()));
  return out;
}
std::size_t estimated_peak_bytes(const Context &c) {
  const auto product = [](std::size_t a, std::size_t b) {
    return checked_product(a, b, "host memory plan");
  };
  const auto sum = [](std::size_t a, std::size_t b) {
    return checked_sum(a, b, "host memory plan");
  };
  const auto m = static_cast<std::size_t>(2 * c.p.K + 1);
  const auto edges = product(m, c.op.Ne), system = sum(edges, c.op.Bc.rows);
  // Conservative simultaneously live dense systems, LU/SVD copies and query
  // workspaces.
  auto entries = product(32, product(system, system));
  entries = sum(entries, product(4, c.op.Bc_full.data.size()));
  entries =
      sum(entries,
          product(256, product(c.N, sum(sum(3 * c.op.Nc, c.op.Ne), c.op.Nb))));
  entries = sum(entries, product(32, product(m, sum(11 * c.op.Nc, c.op.Nb))));
  std::size_t resident = 0;
  for (auto a : {&c.op.G, &c.op.C, &c.op.DB, &c.op.DJ, &c.op.Cdag, &c.op.Lphi})
    for (const auto &matrix : *a)
      resident = sum(resident, matrix.data.size());
  for (auto a : {&c.op.I, &c.op.H, &c.op.QB, &c.op.T, &c.op.T_full, &c.op.PR_r,
                 &c.op.PZ_r, &c.op.PR_z, &c.op.PZ_z, &c.op.PR_v, &c.op.PZ_v})
    resident = sum(resident, a->data.size());
  entries = sum(entries, product(8, resident));
  return product(entries, sizeof(Complex));
}
State allocate_state(const Context &c) {
  if (c.p.memory_limit == 0)
    throw Failure("Memory", "a positive host memory cap is required");
  require_memory(estimated_peak_bytes(c),
                 {"complete A0 peak storage plan", c.p.memory_limit});
  const auto nc = c.op.Nc, M = static_cast<std::size_t>(2 * c.p.K + 1);
  const auto entries = checked_product(
      M, checked_sum(11 * nc, c.op.MB.size(), "state"), "state");
  require_memory(checked_product(entries, 16 * 20, "state/workspace"),
                 {"state/workspace", c.p.memory_limit});
  State u;
  for (int a = 0; a < 2; ++a) {
    u.fluid[a].n.resize(M * nc);
    u.fluid[a].mom.resize(M * 3 * nc);
    u.fluid[a].perp.resize(M * nc);
    if (a == 1)
      u.fluid[a].parallel.resize(M * nc);
  }
  u.B.resize(M * c.op.MB.size());
  return u;
}
double time_at_step(const Parameters &p, std::uint64_t n) {
  if (n > (1ULL << 52) || !std::isfinite(p.dt) || p.dt <= 0)
    throw Failure("Time", "invalid step/dt");
  const double t = std::fma(static_cast<double>(n), p.dt, p.start_time);
  if (!std::isfinite(t))
    throw Failure("Time", "nonfinite scheduled time");
  return t;
}
double time_at_midpoint(const Parameters &p, std::uint64_t n) {
  const double t = std::fma(static_cast<double>(n) + 0.5, p.dt, p.start_time);
  if (!(time_at_step(p, n) < t && t < time_at_step(p, n + 1)))
    throw Failure("TimeResolution", "midpoint not representable");
  return t;
}
Vec mode_apply(const std::vector<Matrix> &matrices, const Vec &x,
               Compute &backend) {
  if (matrices.empty())
    throw Failure("Shape", "empty mode operator");
  const auto ni = matrices[0].cols, no = matrices[0].rows;
  if (x.size() != matrices.size() * ni)
    throw Failure("Shape", "mode operator shape");
  Vec out(matrices.size() * no);
  for (std::size_t k = 0; k < matrices.size(); ++k)
    put(out, k * no, backend.mv(matrices[k], slice(x, k * ni, ni)));
  return out;
}
Vec sample_apply(const Matrix &A, const Vec &x, int N, Compute &backend) {
  if (N <= 0 || x.size() != static_cast<std::size_t>(N) * A.cols)
    throw Failure("Shape", "sample operator shape");
  Vec out(static_cast<std::size_t>(N) * A.rows);
  for (int l = 0; l < N; ++l)
    put(out, l * A.rows, backend.mv(A, slice(x, l * A.cols, A.cols)));
  return out;
}
Vec inverse(const Vec &x, std::size_t E, const Context &c, bool physical,
            double reference) {
  if (x.size() != static_cast<std::size_t>(2 * c.p.K + 1) * E)
    throw Failure("Shape", "inverse FFT shape");
  Vec y = c.compute.inverse_fft(x, E, c.p.K, c.N);
  if (physical) {
    if (reference < 0 || !std::isfinite(reference))
      throw Failure("Reference", "invalid imaginary-residue reference");
    const double limit = 512 * std::numeric_limits<double>::epsilon() *
                         std::max({max_abs(x), reference, 1e-300});
    for (std::size_t j = 0; j < y.size(); ++j) {
      if (!std::isfinite(y[j].real()) || !std::isfinite(y[j].imag()) ||
          std::abs(y[j].imag()) > limit) {
        std::ostringstream detail;
        detail.precision(17);
        detail << "non-real physical sample at " << j << "; entities=" << E
               << " N=" << c.N << " value=" << y[j] << " limit=" << limit
               << " spectrum_norm=" << max_abs(x);
        throw Failure("Imaginary", detail.str());
      }
      y[j] = {y[j].real(), 0};
    }
  }
  return y;
}
Vec forward(const Vec &x, std::size_t E, const Context &c) {
  return c.compute.forward_fft(x, E, c.p.K, c.N);
}
void check_retained(const State &u, const Context &c) {
  const auto check = [&](const Vec &a, std::size_t E, const char *name) {
    if (a.size() != E * static_cast<std::size_t>(2 * c.p.K + 1) ||
        !is_finite(a))
      throw Failure("State", name);
    const double lim = 512 * std::numeric_limits<double>::epsilon() *
                       std::max(max_abs(a), 1e-300);
    for (std::size_t e = 0; e < E; ++e) {
      if (std::abs(a[c.p.K * E + e].imag()) > lim)
        throw Failure("Hermitian", name);
      for (int k = 1; k <= c.p.K; ++k)
        if (std::abs(a[(c.p.K - k) * E + e] -
                     std::conj(a[(c.p.K + k) * E + e])) > lim)
          throw Failure("Hermitian", name);
    }
  };
  for (int s = 0; s < 2; ++s) {
    const auto &a = u.fluid[s];
    check(a.n, c.op.Nc, "density");
    check(a.mom, 3 * c.op.Nc, "momentum");
    check(a.perp, c.op.Nc, "pressure");
    if (s)
      check(a.parallel, c.op.Nc, "parallel pressure");
    for (int k = 0; k < 2 * c.p.K + 1; ++k) {
      const auto x = slice(a.mom, k * 3 * c.op.Nc, 3 * c.op.Nc);
      const double lim =
          2048 * std::numeric_limits<double>::epsilon() *
          std::max({max_abs(x),
                    c.p.species[s].mass * (s ? c.p.ref.ne : c.p.ref.ni) *
                        c.p.ref.velocity,
                    1e-300});
      const double wall_residual = max_abs(c.compute.mv(c.op.T_full, x));
      if (wall_residual > lim) {
        std::ostringstream detail;
        detail.precision(17);
        detail << "input momentum violates full wall trace: species="
               << c.p.species[s].id << " mode=" << k - c.p.K
               << " residual=" << wall_residual << " limit=" << lim;
        throw Failure("WallState", detail.str());
      }
    }
  }
  check(u.B, c.op.MB.size(), "B");
}
State combine(const State &base, const State &rhs, double weight,
              const StageTag &tag, const Context &c) {
  if (!std::isfinite(weight))
    throw Failure("Time", "nonfinite stage weight");
  State out;
  for (int a = 0; a < 2; ++a) {
    out.fluid[a].n = c.compute.axpy(base.fluid[a].n, rhs.fluid[a].n, weight);
    out.fluid[a].mom =
        c.compute.axpy(base.fluid[a].mom, rhs.fluid[a].mom, weight);
    out.fluid[a].perp =
        c.compute.axpy(base.fluid[a].perp, rhs.fluid[a].perp, weight);
    if (a)
      out.fluid[a].parallel =
          c.compute.axpy(base.fluid[a].parallel, rhs.fluid[a].parallel, weight);
  }
  out.B = c.compute.axpy(base.B, rhs.B, weight);
  out.time = tag.time;
  out.step = tag.step;
  out.generation = tag.generation;
  check_retained(out, c);
  return out;
}
} // namespace fomhd
