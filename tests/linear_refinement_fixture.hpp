#pragma once

#include "fomhd/backend.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace fomhd::test::linear_refinement {

inline void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Action> void rejects(Action action, const char *message) {
  bool rejected = false;
  try {
    action();
  } catch (const NumericalError &) {
    rejected = true;
  }
  require(rejected, message);
}

inline double original_backward_error(const Matrix &a, const Vec &x,
                                      const Vec &b) {
  using Wide = std::complex<long double>;
  long double worst = 0;
  for (std::size_t i = 0; i < a.rows; ++i) {
    Wide residual = -Wide(b[i].real(), b[i].imag());
    long double denominator = std::abs(Wide(b[i].real(), b[i].imag()));
    for (std::size_t j = 0; j < a.cols; ++j) {
      const Wide coefficient(a(i, j).real(), a(i, j).imag());
      const Wide value(x[j].real(), x[j].imag());
      residual += coefficient * value;
      denominator += std::abs(coefficient) * std::abs(value);
    }
    worst = std::max(worst, denominator > 0 ? std::abs(residual) / denominator
                                          : std::abs(residual));
  }
  return static_cast<double>(worst);
}

inline void verify_result(Compute &compute, const Matrix &a, const Vec &b,
                          const LinearResult &result, double tolerance) {
  require(is_finite(result.x) && is_finite(result.residual),
          "checked solve returned non-finite output");
  require(std::isfinite(result.backward_error) &&
              result.backward_error <= tolerance,
          "checked solve relaxed the componentwise tolerance");
  require(result.refinement_steps <= 5,
          "checked solve exceeded its residual correction budget");
  require(original_backward_error(a, result.x, b) <= tolerance,
          "independent original-equation backward error exceeded tolerance");
  const Vec residual = subtract(compute.mv(a, result.x), b);
  require(result.residual == residual,
          "reported residual is not from the original equations");
}

inline void homogeneous_rows(Compute &compute) {
  Matrix a(4, 4);
  const double coupling = -1.7832613921822793e-23;
  a(0, 0) = 1;
  a(1, 0) = coupling;
  a(1, 1) = 19.502390444902698;
  a(1, 2) = -5.634988559132248;
  a(1, 3) = coupling;
  a(2, 1) = -5.201527900737459;
  a(2, 2) = 19.531052633287366;
  a(3, 3) = 1;
  const Vec b{{0, -3.591704970194186e-23}, 0, 0,
              {0, 3.591704970194189e-23}};
  const double tolerance = 64 * 4 * std::numeric_limits<double>::epsilon();
  const auto result = compute.linear(a, b, RealVec(4, 1), RealVec(4, 1),
                                     1e-12, tolerance);
  verify_result(compute, a, b, result, tolerance);
  require(result.x[0] == b[0] && result.x[3] == b[3],
          "refinement changed fixed endpoint unknowns");
  std::cout << "PASS homogeneous_rows backward_error=" << result.backward_error
            << " refinement_steps=" << result.refinement_steps
            << " original_wide_error="
            << original_backward_error(a, result.x, b) << '\n';
}

inline void complex_kkt(Compute &compute) {
  Matrix a(4, 4);
  a(0, 0) = 4;
  a(0, 1) = {1, 1};
  a(0, 3) = 1;
  a(1, 0) = {1, -1};
  a(1, 1) = 5;
  a(1, 2) = {0, 2};
  a(1, 3) = {0, -1};
  a(2, 1) = {0, -2};
  a(2, 2) = 6;
  a(2, 3) = 1;
  a(3, 0) = 1;
  a(3, 1) = {0, 1};
  a(3, 2) = 1;
  // Hand-derived KKT solution with a zero constraint right-hand side.
  const Vec expected{{1, 2}, {-2, .5}, -.5, {.25, -.75}};
  const Vec b{{1.75, 5.75}, {-7.75, 2.25}, {-1.75, 3.25}, 0};
  const RealVec rows{.25, 2, 4, .5}, cols{2, .5, 4, .25};
  const Matrix original = a;
  const auto result = compute.linear(a, b, rows, cols, 1e-12, 1e-12);
  verify_result(compute, a, b, result, 1e-12);
  require(max_abs(subtract(result.x, expected)) < 1e-13,
          "complex KKT solution differs from its manufactured solution");
  require(a.data == original.data && result.row_scale == rows &&
              result.column_scale == cols && result.singular_values.size() == 4,
          "checked solve changed inputs or omitted solve diagnostics");
  const auto zero = compute.linear(a, Vec(4), rows, cols, 1e-12, 1e-12);
  require(zero.x == Vec(4) && zero.residual == Vec(4) &&
              zero.scaled_residual == 0 && zero.backward_error == 0 &&
              zero.refinement_steps == 0,
          "zero right-hand side was not solved exactly");
  std::cout << "PASS complex_kkt_and_zero_rhs\n";
}

inline void rejection_controls(Compute &compute) {
  const Matrix a = identity(3);
  const Vec b{{1, 2}, {-3, .25}, {.5, -1}};
  const RealVec scales(3, 1);
  auto singular = a;
  for (std::size_t j = 0; j < 3; ++j)
    singular(1, j) = singular(0, j);
  rejects([&] { compute.linear(singular, b, scales, scales, 1e-12, 1e-12); },
          "singular matrix accepted after refinement");
  for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::infinity()}) {
    auto invalid_a = a;
    invalid_a(0, 1) = invalid;
    rejects([&] { compute.linear(invalid_a, b, scales, scales, 1e-12, 1e-12); },
            "non-finite matrix accepted");
    auto invalid_b = b;
    invalid_b[1] = invalid;
    rejects([&] { compute.linear(a, invalid_b, scales, scales, 1e-12, 1e-12); },
            "non-finite right-hand side accepted");
    rejects([&] { compute.linear(a, b, scales, scales, 1e-12, invalid); },
            "non-finite residual tolerance accepted");
  }
  auto invalid_scales = scales;
  invalid_scales[0] = 0;
  rejects([&] { compute.linear(a, b, invalid_scales, scales, 1e-12, 1e-12); },
          "invalid row scale accepted");
  rejects([&] { compute.linear(a, b, scales, invalid_scales, 1e-12, 1e-12); },
          "invalid column scale accepted");

  Matrix inexact(5, 5);
  for (std::size_t i = 0; i < 5; ++i)
    for (std::size_t j = 0; j < 5; ++j)
      inexact(i, j) = {double(i + 2 * j + 1) / 7 + (i == j ? 3 : 0),
                        double(2 * i + j + 1) / 11};
  const Vec inexact_b{{1.1, .4}, {.8, -1.3}, {-.7, .2}, {.3, .9}, {-1.2, -.1}};
  rejects([&] {
    compute.linear(inexact, inexact_b, RealVec(5, 1), RealVec(5, 1),
                   1e-12, 1e-30);
  }, "unachievable tolerance accepted after bounded refinement");
  std::cout << "PASS rejection_controls\n";
}

inline void numeric(Compute &compute) {
  homogeneous_rows(compute);
  complex_kkt(compute);
  rejection_controls(compute);
}

} // namespace fomhd::test::linear_refinement
