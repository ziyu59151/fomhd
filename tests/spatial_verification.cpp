#include "fomhd/operators.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using fomhd::Complex;
using fomhd::Geometry;
using fomhd::Matrix;
using fomhd::SpatialOperators;
using fomhd::Vec;

constexpr double kR0 = 1.0;
constexpr double kZ0 = 0.0;
constexpr double kBumpR = 1.5;
constexpr double kBumpZ = 0.5;
constexpr double kBumpWidth = 0.42;
constexpr double kRegionRMin = 1.25;
constexpr double kRegionRMax = 1.75;
constexpr double kRegionZMin = 0.25;
constexpr double kRegionZMax = 0.75;
constexpr int kMode = 2;
constexpr std::size_t kMemoryCapBytes = 3U * 1024U * 1024U * 1024U;

struct Bump {
  double value{};
  double dr{};
  double dz{};
};

struct ErrorStats {
  double interior_error{};
  double wall_error{};
  double global_error{};
  double interior_scale{};
  double global_scale{};

  [[nodiscard]] double interior() const {
    return interior_error / interior_scale;
  }

  [[nodiscard]] double wall() const {
    return wall_error / global_scale;
  }

  [[nodiscard]] double global() const {
    return global_error / global_scale;
  }
};

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

double axis_value(double coordinate, double center) {
  const double s = (coordinate - center) / kBumpWidth;
  if (std::abs(s) >= 1.0) {
    return 0.0;
  }
  return std::exp(-1.0 / (1.0 - s * s));
}

double axis_derivative(double coordinate, double center) {
  const double s = (coordinate - center) / kBumpWidth;
  if (std::abs(s) >= 1.0) {
    return 0.0;
  }
  const double one_minus = 1.0 - s * s;
  return axis_value(coordinate, center) *
         (-2.0 * s / (kBumpWidth * one_minus * one_minus));
}

Bump bump(double R, double Z) {
  const double radial = axis_value(R, kBumpR);
  const double axial = axis_value(Z, kBumpZ);
  return {radial * axial,
          axis_derivative(R, kBumpR) * axial,
          radial * axis_derivative(Z, kBumpZ)};
}

bool in_interior(double R, double Z) {
  return R >= kRegionRMin && R <= kRegionRMax && Z >= kRegionZMin &&
         Z <= kRegionZMax;
}

void account(ErrorStats& stats, double R, double Z, bool boundary,
             Complex actual, Complex exact) {
  if (!std::isfinite(R) || !std::isfinite(Z) ||
      !std::isfinite(actual.real()) || !std::isfinite(actual.imag()) ||
      !std::isfinite(exact.real()) || !std::isfinite(exact.imag())) {
    fail("nonfinite coordinates, actual value, or analytic reference");
  }
  const double error = std::abs(actual - exact);
  const double magnitude = std::abs(exact);
  if (!std::isfinite(error) || !std::isfinite(magnitude)) {
    fail("nonfinite error or analytic magnitude");
  }
  stats.global_error = std::max(stats.global_error, error);
  stats.global_scale = std::max(stats.global_scale, magnitude);
  if (boundary) {
    stats.wall_error = std::max(stats.wall_error, error);
  }
  if (in_interior(R, Z)) {
    stats.interior_error = std::max(stats.interior_error, error);
    stats.interior_scale = std::max(stats.interior_scale, magnitude);
  }
}

void nonfinite_negative_controls() {
  const auto rejects = [](Complex actual, Complex exact) {
    ErrorStats stats;
    try {
      account(stats, 1.5, 0.5, false, actual, exact);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double infinity = std::numeric_limits<double>::infinity();
  if (!rejects(Complex{nan, 0.0}, 1.0) ||
      !rejects(1.0, Complex{infinity, 0.0})) {
    fail("nonfinite accumulation negative control was not rejected");
  }
}

ErrorStats gradient_error(const Geometry& geometry, const Matrix& gradient,
                          int mode) {
  const std::size_t nr = geometry.r_faces.keys.size();
  const std::size_t nc = geometry.cells.keys.size();
  const std::size_t nz = geometry.z_faces.keys.size();
  const Complex coefficient{0.8, -0.3};
  Vec scalar(nc);
  for (std::size_t i = 0; i < nc; ++i) {
    scalar[i] =
        coefficient * bump(geometry.cells.R[i], geometry.cells.Z[i]).value;
  }
  const Vec actual = fomhd::matvec(gradient, scalar);
  ErrorStats stats;
  for (std::size_t i = 0; i < nr; ++i) {
    const double R = geometry.r_faces.R[i];
    const double Z = geometry.r_faces.Z[i];
    account(stats, R, Z, geometry.r_faces.boundary[i] != 0, actual[i],
            coefficient * bump(R, Z).dr);
  }
  for (std::size_t i = 0; i < nc; ++i) {
    const double R = geometry.cells.R[i];
    const double Z = geometry.cells.Z[i];
    const Complex exact =
        Complex{0.0, 1.0} * static_cast<double>(mode) * coefficient *
        bump(R, Z).value / R;
    account(stats, R, Z, geometry.cells.boundary[i] != 0,
            actual[nr + i], exact);
  }
  for (std::size_t i = 0; i < nz; ++i) {
    const double R = geometry.z_faces.R[i];
    const double Z = geometry.z_faces.Z[i];
    account(stats, R, Z, geometry.z_faces.boundary[i] != 0,
            actual[nr + nc + i], coefficient * bump(R, Z).dz);
  }
  return stats;
}

constexpr Complex kER{1.0, 0.0};
constexpr Complex kEPhi{-0.5, 0.25};
constexpr Complex kEZ{0.75, 0.0};

Vec electric_fourier(const Geometry& geometry) {
  const std::size_t nr = geometry.r_faces.keys.size();
  const std::size_t nc = geometry.cells.keys.size();
  const std::size_t nz = geometry.z_faces.keys.size();
  Vec electric(nr + nc + nz);
  for (std::size_t i = 0; i < nr; ++i) {
    electric[i] =
        kER * bump(geometry.r_faces.R[i], geometry.r_faces.Z[i]).value;
  }
  for (std::size_t i = 0; i < nc; ++i) {
    electric[nr + i] =
        kEPhi * bump(geometry.cells.R[i], geometry.cells.Z[i]).value;
  }
  for (std::size_t i = 0; i < nz; ++i) {
    electric[nr + nc + i] =
        kEZ * bump(geometry.z_faces.R[i], geometry.z_faces.Z[i]).value;
  }
  return electric;
}

ErrorStats curl_error(const Geometry& geometry, const Matrix& curl,
                      int mode, bool flip_phi_derivative) {
  const std::size_t nr = geometry.r_faces.keys.size();
  const std::size_t nz = geometry.z_faces.keys.size();
  const std::size_t nv = geometry.vertices.keys.size();
  const Vec actual = fomhd::matvec(curl, electric_fourier(geometry));
  const Complex imaginary{0.0, 1.0};
  ErrorStats stats;
  for (std::size_t i = 0; i < nz; ++i) {
    const double R = geometry.z_faces.R[i];
    const double Z = geometry.z_faces.Z[i];
    const Bump values = bump(R, Z);
    const Complex exact = imaginary * static_cast<double>(mode) * kEZ *
                              values.value / R -
                          kEPhi * values.dz;
    account(stats, R, Z, geometry.z_faces.boundary[i] != 0, actual[i],
            exact);
  }
  for (std::size_t i = 0; i < nv; ++i) {
    const double R = geometry.vertices.R[i];
    const double Z = geometry.vertices.Z[i];
    const Bump values = bump(R, Z);
    const Complex exact = kER * values.dz - kEZ * values.dr;
    account(stats, R, Z, geometry.vertices.boundary[i] != 0,
            actual[nz + i], flip_phi_derivative ? -exact : exact);
  }
  for (std::size_t i = 0; i < nr; ++i) {
    const double R = geometry.r_faces.R[i];
    const double Z = geometry.r_faces.Z[i];
    const Bump values = bump(R, Z);
    const Complex exact =
        kEPhi * (values.value / R + values.dr) -
        imaginary * static_cast<double>(mode) * kER * values.value / R;
    account(stats, R, Z, geometry.r_faces.boundary[i] != 0,
            actual[nz + nv + i], exact);
  }
  return stats;
}

constexpr Complex kBR{0.75, 0.1};
constexpr Complex kBPhi{0.25, -0.5};
constexpr Complex kBZ{-0.5, 0.2};

Vec field_fourier(const Geometry& geometry) {
  const std::size_t nr = geometry.r_faces.keys.size();
  const std::size_t nz = geometry.z_faces.keys.size();
  const std::size_t nv = geometry.vertices.keys.size();
  Vec field(nz + nv + nr);
  for (std::size_t i = 0; i < nz; ++i) {
    field[i] = kBR * bump(geometry.z_faces.R[i], geometry.z_faces.Z[i]).value;
  }
  for (std::size_t i = 0; i < nv; ++i) {
    field[nz + i] =
        kBPhi * bump(geometry.vertices.R[i], geometry.vertices.Z[i]).value;
  }
  for (std::size_t i = 0; i < nr; ++i) {
    field[nz + nv + i] =
        kBZ * bump(geometry.r_faces.R[i], geometry.r_faces.Z[i]).value;
  }
  return field;
}

ErrorStats divergence_error(const Geometry& geometry,
                            const Matrix& divergence, int mode) {
  const std::size_t nv = geometry.vertices.keys.size();
  const Vec actual = fomhd::matvec(divergence, field_fourier(geometry));
  const Complex imaginary{0.0, 1.0};
  ErrorStats stats;
  for (std::size_t i = 0; i < nv; ++i) {
    const double R = geometry.vertices.R[i];
    const double Z = geometry.vertices.Z[i];
    const Bump values = bump(R, Z);
    const Complex exact =
        kBR * (values.value / R + values.dr) + kBZ * values.dz +
        imaginary * static_cast<double>(mode) * kBPhi * values.value / R;
    account(stats, R, Z, geometry.vertices.boundary[i] != 0, actual[i],
            exact);
  }
  return stats;
}

constexpr Complex kJR{0.6, -0.2};
constexpr Complex kJPhi{-0.4, 0.3};
constexpr Complex kJZ{0.8, 0.1};

ErrorStats divergence_adjoint_error(const Geometry& geometry,
                                    const Matrix& divergence_adjoint,
                                    int mode) {
  const std::size_t nr = geometry.r_faces.keys.size();
  const std::size_t nc = geometry.cells.keys.size();
  const std::size_t nz = geometry.z_faces.keys.size();
  Vec flux(nr + nc + nz);
  for (std::size_t i = 0; i < nr; ++i) {
    flux[i] = kJR * bump(geometry.r_faces.R[i], geometry.r_faces.Z[i]).value;
  }
  for (std::size_t i = 0; i < nc; ++i) {
    flux[nr + i] =
        kJPhi * bump(geometry.cells.R[i], geometry.cells.Z[i]).value;
  }
  for (std::size_t i = 0; i < nz; ++i) {
    flux[nr + nc + i] =
        kJZ * bump(geometry.z_faces.R[i], geometry.z_faces.Z[i]).value;
  }
  const Vec actual = fomhd::matvec(divergence_adjoint, flux);
  const Complex imaginary{0.0, 1.0};
  ErrorStats stats;
  for (std::size_t i = 0; i < nc; ++i) {
    const double R = geometry.cells.R[i];
    const double Z = geometry.cells.Z[i];
    const Bump values = bump(R, Z);
    const Complex exact =
        kJR * (values.value / R + values.dr) + kJZ * values.dz +
        imaginary * static_cast<double>(mode) * kJPhi * values.value / R;
    account(stats, R, Z, geometry.cells.boundary[i] != 0, actual[i],
            exact);
  }
  return stats;
}

double order(double coarse, double fine) {
  return std::log2(coarse / fine);
}

double max_difference(const Matrix& left, const Matrix& right) {
  if (left.rows != right.rows || left.cols != right.cols) {
    fail("matrix comparison shape mismatch");
  }
  double difference = 0.0;
  for (std::size_t i = 0; i < left.data.size(); ++i) {
    difference = std::max(difference, std::abs(left.data[i] - right.data[i]));
  }
  return difference;
}

void irregular_mask_verification() {
  constexpr std::size_t kCap = 512U * 1024U * 1024U;
  const fomhd::OperatorOptions options{1e-10, kCap};
  const Geometry geometry =
      fomhd::build_stepped_disk(26, 30, 0.25, -0.375, 0.025, 0.025,
                               0.55, 0.0, 0.22);
  const auto operators = fomhd::build_operators(geometry, 0, options);

  const Matrix prpz = fomhd::matmul(operators.PR_v, operators.PZ_v);
  const Matrix pzpr = fomhd::matmul(operators.PZ_v, operators.PR_v);
  if (!(max_difference(prpz, pzpr) > 1e-8)) {
    fail("stepped fixture must make PR_v and PZ_v noncommute");
  }

  const Matrix identity_block = fomhd::identity(operators.Nb);
  Matrix per_column(operators.Nb, operators.Nb);
  Vec basis(operators.Nb);
  for (std::size_t column = 0; column < operators.Nb; ++column) {
    basis[column] = 1.0;
    const Vec action = fomhd::apply_t2_inverse(operators, basis);
    for (std::size_t row = 0; row < operators.Nb; ++row) {
      per_column(row, column) = action[row];
    }
    basis[column] = 0.0;
  }
  std::cout << "stepped_pr_pz_commutator="
            << max_difference(prpz, pzpr) << '\n';
  for (const int mode : {-2, 0, 2}) {
    const SpatialOperators spatial =
        fomhd::build_spatial_operators(geometry, mode, options);
    const Matrix multi_rhs = fomhd::apply_t2_inverse_block(
        operators, identity_block, options);
    const double inverse_difference =
        max_difference(multi_rhs, per_column);
    const double inverse_bound =
        512.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(operators.Nb) * max_abs(multi_rhs);
    if (!std::isfinite(inverse_difference) ||
        inverse_difference > inverse_bound) {
      fail("multi-RHS T2 inverse differs from per-column apply_t2_inverse");
    }

    const Matrix cg = fomhd::matmul(spatial.C, spatial.G);
    const Matrix dbc = fomhd::matmul(spatial.DB, spatial.C);
    const double cg_bound =
        128.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(spatial.C.cols) * max_abs(spatial.C) *
        max_abs(spatial.G);
    const double dbc_bound =
        128.0 * std::numeric_limits<double>::epsilon() *
        static_cast<double>(spatial.DB.cols) * max_abs(spatial.DB) *
        max_abs(spatial.C);
    if (!(max_abs(cg) <= cg_bound) || !(max_abs(dbc) <= dbc_bound)) {
      fail("stepped-geometry full CG/DBC residual exceeds its roundoff bound");
    }

    std::cout << "stepped_mode=" << mode
              << " t2_inverse_equivalence=" << inverse_difference
              << " CG=" << max_abs(cg) << " bound=" << cg_bound
              << " DBC=" << max_abs(dbc) << " bound=" << dbc_bound
              << '\n';
  }
}

void tall_workspace_control() {
  const auto geometry = fomhd::build_rectangle(4, 4, 1, 0, .2, .2);
  const auto operators = fomhd::build_operators(geometry, 0);
  const Matrix tall(4 * operators.Nb, operators.Nb);
  // Two dense blocks cannot cover the live right-solve and conversion copies.
  const std::size_t cap = 2 * tall.data.size() * sizeof(Complex);
  bool rejected = false;
  try {
    (void)fomhd::apply_t2_inverse_block(operators, tall, {1e-10, cap});
  } catch (const fomhd::NumericalError& error) {
    rejected = std::string{error.what()}.find("memory budget exceeded") !=
               std::string::npos;
  }
  if (!rejected) fail("tall right-solve workspace exceeded its declared budget");
}

struct Level {
  int n{};
  ErrorStats gradient;
  ErrorStats curl;
  ErrorStats divergence;
  ErrorStats divergence_adjoint;
};

void validate(const Level& level, const Level& next,
              const ErrorStats& wrong_sign_curl) {
  constexpr double kMinimumFiniteError = 1e-12;
  constexpr double kMinimumOrder = 3.5;
  const std::array<std::pair<std::string, std::pair<double, double>>, 4>
      observations{{
          {"gradient", {level.gradient.interior(), next.gradient.interior()}},
          {"curl", {level.curl.interior(), next.curl.interior()}},
          {"divergence",
           {level.divergence.interior(), next.divergence.interior()}},
          {"divergence_adjoint",
           {level.divergence_adjoint.interior(),
            next.divergence_adjoint.interior()}},
      }};
  for (const auto& [name, errors] : observations) {
    if (!(errors.first > 0.0) || !(errors.second > 0.0) ||
        !std::isfinite(errors.first) || !std::isfinite(errors.second)) {
      fail(name + " has no finite analytic-normalized residual");
    }
    const double observed = order(errors.first, errors.second);
    if (!std::isfinite(observed) || errors.second <= kMinimumFiniteError) {
      fail(name + " does not have a finite, nonzero convergence residual");
    }
    if (observed < kMinimumOrder) {
      fail(name + " interior order " + std::to_string(observed) +
           " is below " + std::to_string(kMinimumOrder));
    }
  }
  if (!(wrong_sign_curl.interior() > 0.1)) {
    fail("deliberate flipped-derivative negative control did not fail");
  }
}

int run() {
  std::cout << std::setprecision(17);
  std::vector<Level> levels;
  ErrorStats wrong_sign_curl;
  for (const int n : {8, 16, 32}) {
    const double h = 1.0 / static_cast<double>(n);
    const Geometry geometry =
        fomhd::build_rectangle(n + 1, n + 1, kR0, kZ0, h, h);
    const auto operators =
        fomhd::build_spatial_operators(
            geometry, kMode, fomhd::OperatorOptions{1e-10, kMemoryCapBytes});
    Level level;
    level.n = n;
    level.gradient = gradient_error(geometry, operators.G, kMode);
    level.curl = curl_error(geometry, operators.C, kMode, false);
    level.divergence = divergence_error(geometry, operators.DB, kMode);
    level.divergence_adjoint =
        divergence_adjoint_error(geometry, operators.DJ, kMode);
    if (n == 8) {
      wrong_sign_curl = curl_error(geometry, operators.C, kMode, true);
      const auto negative_mode =
          fomhd::build_spatial_operators(
              geometry, -2,
              fomhd::OperatorOptions{1e-10, kMemoryCapBytes});
      const ErrorStats negative_curl =
          curl_error(geometry, negative_mode.C, -2, false);
      if (!std::isfinite(negative_curl.interior()) ||
          negative_curl.interior() >= 0.1) {
        fail("negative toroidal mode did not match its signed analytic curl");
      }
      std::cout << "negative_mode=-2 curl interior="
                << negative_curl.interior() << '\n';
    }
    if (n == 32) {
      bool rejected = false;
      try {
        static_cast<void>(fomhd::build_spatial_operators(
            geometry, kMode,
            fomhd::OperatorOptions{
                1e-10, 2U * 1024U * 1024U * 1024U}));
      } catch (const fomhd::NumericalError&) {
        rejected = true;
      }
      if (!rejected) {
        fail("spatial operator memory rejection did not trigger at 2 GiB");
      }
      std::cout << "memory_rejection=2GiB PASS\n";
    }
    std::cout << "level n=" << n << " h=" << h << '\n'
              << "  gradient interior=" << level.gradient.interior()
              << " wall=" << level.gradient.wall()
              << " global=" << level.gradient.global() << '\n'
              << "  curl interior=" << level.curl.interior()
              << " wall=" << level.curl.wall()
              << " global=" << level.curl.global() << '\n'
              << "  divergence interior=" << level.divergence.interior()
              << " wall=" << level.divergence.wall()
              << " global=" << level.divergence.global() << '\n'
              << "  divergence_adjoint interior="
              << level.divergence_adjoint.interior()
              << " wall=" << level.divergence_adjoint.wall()
              << " global=" << level.divergence_adjoint.global() << '\n';
    levels.push_back(level);
  }
  for (std::size_t i = 0; i + 1 < levels.size(); ++i) {
    std::cout << "orders " << levels[i].n << "->" << levels[i + 1].n
              << " gradient="
              << order(levels[i].gradient.interior(),
                       levels[i + 1].gradient.interior())
              << " curl="
              << order(levels[i].curl.interior(),
                       levels[i + 1].curl.interior())
              << " divergence="
              << order(levels[i].divergence.interior(),
                       levels[i + 1].divergence.interior())
              << " divergence_adjoint="
              << order(levels[i].divergence_adjoint.interior(),
                       levels[i + 1].divergence_adjoint.interior())
              << '\n';
    validate(levels[i], levels[i + 1], wrong_sign_curl);
  }
  irregular_mask_verification();
  std::cout << "negative_control flipped_curl_phi_derivative interior="
            << wrong_sign_curl.interior() << '\n'
            << "SPATIAL_VERIFICATION_PASS\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    nonfinite_negative_controls();
    tall_workspace_control();
    std::cout << "nonfinite_negative_controls=PASS\n";
    if (argc > 1 && std::string{argv[1]} == "irregular") {
      irregular_mask_verification();
      std::cout << "IRREGULAR_MASK_VERIFICATION_PASS\n";
      return 0;
    }
    return run();
  } catch (const std::exception& error) {
    std::cerr << "SPATIAL_VERIFICATION_FAIL: " << error.what() << '\n';
    return 1;
  }
}
