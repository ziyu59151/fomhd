#include "support.hpp"
#include "linear_refinement_fixture.hpp"
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
using namespace fomhd;
namespace {
void require(bool v, const char *message) {
  if (!v)
    throw std::runtime_error(message);
}
void near(const Vec &a, const Vec &b, double reference, const char *message) {
  require(a.size() == b.size(), "comparison shape mismatch");
  const double error =
      max_abs(subtract(a, b)) / std::max(reference, max_abs(a));
  if (error > 2e-10)
    throw std::runtime_error(std::string(message) +
                             " relative error=" + std::to_string(error));
}
template <class F> void rejects(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const NumericalError &) {
    rejected = true;
  }
  require(rejected, "invalid GPU input accepted");
}
} // namespace
int main(int argc, char **argv) {
  try {
    std::string which = argc > 1 ? argv[1] : "numeric";
    auto gpu = make_cuda_compute(0, 512ULL << 20);
    Compute cpu;
    if (which == "numeric") {
      Matrix a(3, 2), b(2, 4);
      for (std::size_t i = 0; i < a.data.size(); ++i)
        a.data[i] = {double(i + 1) / 7, double(i % 3) / 5};
      for (std::size_t i = 0; i < b.data.size(); ++i)
        b.data[i] = {double(i + 2) / 9, -double(i % 4) / 6};
      Vec x{{2, 3}, {-4, 1}};
      near(cpu.mv(a, x), gpu->mv(a, x), 1, "GEMV");
      near(cpu.mm(a, b).data, gpu->mm(a, b).data, 1, "GEMM");
      near(cpu.axpy(x, x, -.3), gpu->axpy(x, x, -.3), 1, "AXPY");
      for (int K : {0, 1, 2})
        for (std::size_t entities : {1, 3, 17}) {
          int N = 4 * K + 3;
          Vec h((2 * K + 1) * entities);
          for (std::size_t i = 0; i < h.size(); ++i)
            h[i] = {std::sin(double(i)), std::cos(double(i))};
          auto sample = cpu.inverse_fft(h, entities, K, N);
          near(sample, gpu->inverse_fft(h, entities, K, N), 1,
               "inverse FFT stride/bin");
          near(h, gpu->forward_fft(sample, entities, K, N), 1,
               "forward FFT scaling/bin");
          near(cpu.derivative(sample, entities, N),
               gpu->derivative(sample, entities, N), 1,
               "full sample derivative");
        }
      Matrix linear(3, 3);
      linear(0, 0) = {4, 1};
      linear(0, 1) = {1, -2};
      linear(1, 1) = 5;
      linear(2, 2) = 3;
      linear(2, 0) = {2, 1};
      Vec expected{{1, 2}, {-3, .5}, {.2, -.7}}, rhs = cpu.mv(linear, expected);
      auto result = gpu->linear(linear, rhs, {1e-2, 1, 1e2}, {2e-2, 3, 1e2},
                                1e-12, 1e-12);
      near(expected, result.x, 1, "complex scaled LU");
      auto reference =
          cpu.linear(linear, rhs, {1e-2, 1, 1e2}, {2e-2, 3, 1e2}, 1e-12, 1e-12);
      for (std::size_t i = 0; i < 3; ++i)
        require(
            std::abs(result.singular_values[i] / reference.singular_values[i] -
                     1) < 1e-10,
            "SVD spectrum mismatch");
      test::linear_refinement::numeric(*gpu);
      rejects([&] {
        gpu->linear(Matrix(2, 2), Vec(2), {1, 1}, {1, 1}, 1e-12, 1e-12);
      });
      rejects([&] { gpu->mv(a, Vec(3)); });
      rejects([&] { gpu->forward_fft(Vec(9), 2, 1, 5); });
    } else if (which == "stage") {
      auto p = test::parameters();
      p.nsteps = 2;
      auto g = build_rectangle(4, 4, 1, 0, .2, .2);
      auto op = build_operators(g, p.K);
      Context cc{p, g, op, cpu, p.Nphi}, cg{p, g, op, *gpu, p.Nphi};
      auto initial = closed_reference(cc, p.ref.ni, p.ref.pressure,
                                      p.ref.pressure, p.ref.magnetic, .01);
      Cumulative lc, lg;
      StageResult fc, fg;
      auto uc = initialize(initial, cc, lc, false, &fc),
           ug = initialize(initial, cg, lg, false, &fg);
      near(fc.cache.E, fg.cache.E, p.ref.electric, "Darwin E");
      near(fc.fJ0, fg.fJ0, p.ref.current / p.ref.time,
           "coupled current response");
      for (int step = 0; step < 2; ++step) {
        (void)advance(uc, lc, cc);
        (void)advance(ug, lg, cg);
      }
      for (int s = 0; s < 2; ++s) {
        near(uc.state.fluid[s].n, ug.state.fluid[s].n, p.ref.ni,
             "evolved number density");
        near(uc.state.fluid[s].mom, ug.state.fluid[s].mom,
             p.species[s].mass * p.ref.ni, "evolved momentum");
        near(uc.state.fluid[s].perp, ug.state.fluid[s].perp, p.ref.pressure,
             "evolved pressure");
      }
      near(uc.state.fluid[1].parallel, ug.state.fluid[1].parallel,
           p.ref.pressure, "evolved parallel pressure");
      near(uc.state.B, ug.state.B, p.ref.magnetic, "evolved B");
      require(lg.accepted_intervals == 2, "GPU intervals were not committed");
    } else if (which == "capacity") {
      auto small = make_cuda_compute(0, 8ULL << 20);
      rejects([&] { small->mv(Matrix(1024, 1024), Vec(1024)); });
    } else
      throw std::runtime_error("unknown CUDA test");
    gpu->synchronize();
    require(gpu->h2d_bytes > 0 || which == "capacity",
            "no explicit uploads recorded");
    std::cout << "PASS " << which << " backend=" << gpu->name()
              << " H2D=" << gpu->h2d_bytes << " D2H=" << gpu->d2h_bytes << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
