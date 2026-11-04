#include "fomhd/geometry.hpp"
#include "fomhd/numerics.hpp"
#include "fomhd/operators.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using fomhd::Complex;
using fomhd::Geometry;
using fomhd::Matrix;
using fomhd::Operators;
using fomhd::Point2;
using fomhd::RealVec;
using fomhd::Vec;

constexpr Complex kI{0.0, 1.0};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near(double actual, double expected, double tolerance,
          const std::string& message) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                                 " expected=" + std::to_string(expected));
    }
}

void near_complex(Complex actual, Complex expected, double tolerance,
                  const std::string& message) {
    near(std::abs(actual - expected), 0.0, tolerance, message);
}

double abs_error(const Vec& actual, const Vec& expected) {
    require(actual.size() == expected.size(), "vector size mismatch");
    double result = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        result = std::max(result, std::abs(actual[i] - expected[i]));
    }
    return result;
}

Vec deterministic_vector(std::size_t size, std::size_t seed) {
    Vec result(size);
    for (std::size_t i = 0; i < size; ++i) {
        const double x = static_cast<double>((i * 13 + seed * 7) % 19) - 9.0;
        const double y = static_cast<double>((i * 5 + seed * 11) % 17) - 8.0;
        result[i] = Complex(x / 10.0, y / 13.0);
    }
    return result;
}

Complex euclidean_dot(const Vec& left, const Vec& right) {
    require(left.size() == right.size(), "dot size mismatch");
    Complex result{};
    for (std::size_t i = 0; i < left.size(); ++i) {
        result += std::conj(left[i]) * right[i];
    }
    return result;
}

RealVec ones(std::size_t size) {
    return RealVec(size, 1.0);
}

Matrix diagonal_product(const Matrix& left, const Matrix& right) {
    return fomhd::matmul(left, right);
}

Matrix matrix_add(const Matrix& left, const Matrix& right) {
    require(left.rows == right.rows && left.cols == right.cols,
            "matrix sum shape mismatch");
    Matrix result(left.rows, left.cols);
    for (std::size_t i = 0; i < result.data.size(); ++i) {
        result.data[i] = left.data[i] + right.data[i];
    }
    return result;
}

Matrix matrix_subtract(const Matrix& left, const Matrix& right) {
    require(left.rows == right.rows && left.cols == right.cols,
            "matrix difference shape mismatch");
    Matrix result(left.rows, left.cols);
    for (std::size_t i = 0; i < result.data.size(); ++i) {
        result.data[i] = left.data[i] - right.data[i];
    }
    return result;
}

Matrix spin_rotation(std::size_t i, std::size_t j, std::size_t size,
                     double angle) {
    Matrix rotation = fomhd::identity(size);
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    rotation(i, i) = c;
    rotation(j, j) = c;
    rotation(i, j) = -s;
    rotation(j, i) = s;
    return rotation;
}

Matrix phase_matrix(const Vec& phases) {
    Matrix result = fomhd::identity(phases.size());
    for (std::size_t i = 0; i < phases.size(); ++i) {
        result(i, i) = phases[i];
    }
    return result;
}

Geometry make_rectangle_geometry() {
    return fomhd::build_rectangle(4, 3, 0.5, -0.2, 0.1, 0.1);
}

Geometry make_l_geometry() {
    const RealVec R{0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1};
    const RealVec Z{-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4};
    const std::vector<Point2> polygon{
        {0.5, -0.3}, {1.1, -0.3}, {1.1, 0.0},
        {0.8, 0.0},  {0.8, 0.4},  {0.5, 0.4},
    };
    return fomhd::build_geometry(R, Z, polygon);
}

std::vector<std::pair<std::string, std::function<void()>>>& registry() {
    static std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"LA_01_complex_lu", [] {
             Matrix A(3, 3);
             A.data = {Complex{4.0, 1.0}, Complex{2.0, -1.0},
                       Complex{0.0, 2.0}, Complex{-1.0, 0.5},
                       Complex{3.0, 2.0}, Complex{1.0, -1.0},
                       Complex{2.0, -2.0}, Complex{0.0, 1.0},
                       Complex{5.0, 0.0}};
             const Vec expected{Complex{1.0, -2.0}, Complex{-0.5, 0.25},
                                Complex{2.0, 1.0}};
             const Vec rhs = fomhd::matvec(A, expected);
             const auto factorization = fomhd::factor(A);
             const Vec actual = fomhd::solve(factorization, rhs);
             require(abs_error(actual, expected) < 2e-13,
                     "complex LU solution mismatch");

             Matrix rhs_matrix(3, 2);
             const Vec second = deterministic_vector(3, 9);
             const Vec rhs_second = fomhd::matvec(A, second);
             for (std::size_t i = 0; i < 3; ++i) {
                 rhs_matrix(i, 0) = rhs[i];
                 rhs_matrix(i, 1) = rhs_second[i];
             }
             const Matrix solved = fomhd::solve(factorization, rhs_matrix);
             require(std::abs(solved(0, 0) - expected[0]) < 2e-13,
                     "complex LU multiple-RHS mismatch");
             require(std::abs(solved(2, 1) - second[2]) < 2e-13,
                     "complex LU multiple-RHS second column mismatch");
         }},
        {"LA_02_rectangular_svd", [] {
             const Matrix U =
                 fomhd::matmul(phase_matrix({Complex{0.0, 1.0}, -1.0,
                                             Complex{0.6, 0.8}}),
                               fomhd::matmul(spin_rotation(0, 1, 3, 0.4),
                                             spin_rotation(1, 2, 3, -0.7)));
             const Matrix V = fomhd::matmul(
                 phase_matrix({Complex{-1.0, 0.0}, Complex{0.0, 1.0}}),
                 spin_rotation(0, 1, 2, 0.35));
             Matrix sigma(3, 2);
             sigma(0, 0) = 3.0;
             sigma(1, 1) = 2.0;
             const Matrix A =
                 fomhd::matmul(U, fomhd::matmul(sigma, fomhd::adjoint(V)));
             require(fomhd::max_abs(matrix_subtract(
                         fomhd::matmul(fomhd::adjoint(U), U),
                         fomhd::identity(3))) < 1e-13,
                     "SVD fixture U is not unitary");
             const RealVec singular = fomhd::singular_values(A);
             require(singular.size() == 2, "rectangular SVD size mismatch");
             near(singular[0], 3.0, 2e-13, "largest singular value");
             near(singular[1], 2.0, 2e-13, "smallest singular value");
             require(std::abs(A(0, 1)) > 1e-3,
                     "SVD fixture must be non-diagonal");
         }},
        {"LA_03_singular_rejection", [] {
             Matrix singular(2, 2);
             singular.data = {Complex{1.0}, Complex{2.0}, Complex{2.0},
                              Complex{4.0}};
             bool factor_threw = false;
             try {
                 static_cast<void>(fomhd::factor(singular));
             } catch (const fomhd::NumericalError&) {
                 factor_threw = true;
             }
             require(factor_threw, "singular LU was not rejected");

             bool solve_threw = false;
             try {
                 static_cast<void>(fomhd::checked_solve(
                     singular, Vec{Complex{1.0}, Complex{2.0}},
                     ones(2), ones(2), 1e-10));
             } catch (const fomhd::NumericalError&) {
                 solve_threw = true;
             }
             require(solve_threw, "singular checked solve was not rejected");
         }},
        {"FFT_01_direct_dft_and_batch", [] {
             constexpr int K = 2;
             constexpr int N = 9;
             constexpr std::size_t E = 3;
             const fomhd::Fourier fourier(K, N);
             const Vec cycles{Complex{1.3, -0.2}, Complex{-0.7, 0.4},
                              Complex{0.2, 0.9}};
             const std::vector<int> modes{-2, 1, -1};
             Vec samples(N * E);
             for (int sample = 0; sample < N; ++sample) {
                 const double phi = 2.0 * std::numbers::pi * sample / N;
                 for (std::size_t entity = 0; entity < E; ++entity) {
                     samples[static_cast<std::size_t>(sample) * E + entity] =
                         cycles[entity] *
                         std::exp(kI * static_cast<double>(modes[entity]) * phi);
                 }
             }
             const Vec actual = fourier.forward(samples, E);
             require(actual.size() == static_cast<std::size_t>(2 * K + 1) * E,
                     "retained Fourier size mismatch");
             for (int mode = -K; mode <= K; ++mode) {
                 for (std::size_t entity = 0; entity < E; ++entity) {
                     const Complex expected =
                         mode == modes[entity] ? cycles[entity] : Complex{};
                     near_complex(
                         actual[static_cast<std::size_t>(mode + K) * E + entity],
                         expected, 3e-14, "direct DFT coefficient");
                 }
             }
             const Vec roundtrip = fourier.inverse(actual, E);
             require(abs_error(roundtrip, samples) < 3e-13,
                     "retained Fourier roundtrip mismatch");
         }},
        {"FFT_02_negative_mode", [] {
             constexpr int K = 3;
             constexpr int N = 13;
             constexpr std::size_t E = 2;
             const fomhd::Fourier fourier(K, N);
             Vec hat(static_cast<std::size_t>(2 * K + 1) * E);
             hat[static_cast<std::size_t>(K - 2) * E] = Complex{0.7, -1.1};
             hat[static_cast<std::size_t>(K + 3) * E + 1] =
                 Complex{-0.4, 0.25};
             const Vec samples = fourier.inverse(hat, E);
             const Vec recovered = fourier.forward(samples, E);
             require(abs_error(recovered, hat) < 2e-13,
                     "negative-mode Fourier roundtrip mismatch");
         }},
        {"FFT_03_derivative_adjoint", [] {
             constexpr int K = 1;
             constexpr int N = 9;
             constexpr std::size_t E = 2;
             const fomhd::Fourier fourier(K, N);
             Vec samples(N * E);
             for (int sample = 0; sample < N; ++sample) {
                 const double phi = 2.0 * std::numbers::pi * sample / N;
                 for (std::size_t entity = 0; entity < E; ++entity) {
                     const double amplitude =
                         entity == 0 ? 0.6 : Complex{-0.2, 0.3}.real();
                     samples[static_cast<std::size_t>(sample) * E + entity] =
                         amplitude * std::exp(kI * 3.0 * phi);
                 }
             }
             const Vec derivative = fourier.derivative(samples, E);
             for (std::size_t i = 0; i < samples.size(); ++i) {
                 near_complex(derivative[i], kI * 3.0 * samples[i], 8e-13,
                              "full-bin derivative");
             }
             const Vec x = deterministic_vector(samples.size(), 2);
             const Vec y = deterministic_vector(samples.size(), 7);
             const Complex pairing =
                 euclidean_dot(y, fourier.derivative(x, E)) +
                 euclidean_dot(fourier.derivative(y, E), x);
             require(std::abs(pairing) < 2e-11,
                     "sampled derivative is not anti-adjoint");
         }},
        {"GEO_01_rectangle_and_stepped_disk", [] {
             const Geometry rectangle = make_rectangle_geometry();
             require(rectangle.cells.keys.size() == 6,
                     "rectangle cell count");
             require(rectangle.r_faces.keys.size() == 8,
                     "rectangle r-face count");
             require(rectangle.z_faces.keys.size() == 9,
                     "rectangle z-face count");
             require(rectangle.vertices.keys.size() == 12,
                     "rectangle vertex count");
             require(rectangle.walls.size() == 10,
                     "rectangle wall segment count");
             require(rectangle.cell_id(0, 0) == 0,
                     "cell ordering must be (j,i)");
             require(rectangle.cell_id(1, 0) == 1,
                     "cell ordering second entity");

             const Geometry disk =
                 fomhd::build_stepped_disk(26, 30, 0.25, -0.375, 0.025,
                                           0.025, 0.55, 0.0, 0.22);
             require(disk.cells.keys.size() > 100,
                     "stepped disk has too few cells");
             require(disk.walls.size() > 20, "stepped disk wall count");
             bool has_r = false;
             bool has_z = false;
             for (const auto& wall : disk.walls) {
                 has_r = has_r || wall.component == fomhd::WallComponent::r_face;
                 has_z = has_z || wall.component == fomhd::WallComponent::z_face;
             }
             require(has_r && has_z,
                     "stepped disk must have both staircase wall directions");
             require(disk.diagnostics.hausdorff_lower_m <=
                         disk.diagnostics.hausdorff_upper_m,
                     "invalid Hausdorff interval");
         }},
        {"GEO_02_integer_chain_and_positive_weights", [] {
             const auto operators = fomhd::build_operators(
                 make_rectangle_geometry(), 0,
                 fomhd::OperatorOptions{1e-10, 256U * 1024U * 1024U});
             const Matrix chain =
                 matrix_add(fomhd::matmul(operators.b2r, operators.b1r),
                            fomhd::matmul(operators.b2z, operators.b1z));
             require(fomhd::max_abs(chain) == 0.0,
                     "integer incidence chain is not exactly zero");
             for (double value : operators.M0) {
                 require(value > 0.0 && std::isfinite(value),
                         "non-positive M0");
             }
             for (double value : operators.M1) {
                 require(value > 0.0 && std::isfinite(value),
                         "non-positive M1");
             }
             for (double value : operators.MB) {
                 require(value > 0.0 && std::isfinite(value),
                         "non-positive MB");
             }
         }},
        {"OP_01_complex_complex_identities", [] {
             const Geometry geometry = make_rectangle_geometry();
             const auto operators = fomhd::build_operators(geometry, 1);
             const Matrix focused_gradient =
                 fomhd::build_gradient_operator(geometry, 0);
             require(fomhd::max_abs(matrix_subtract(
                         focused_gradient, operators.G[1])) < 1e-14,
                     "focused gradient factory is inconsistent");
             const std::size_t Nc = geometry.cells.keys.size();
             const std::size_t Ne = operators.M1.size();
             for (int mode = -1; mode <= 1; ++mode) {
                 const Vec potential = deterministic_vector(
                     Nc, static_cast<std::size_t>(mode + 4));
                 const Vec cg =
                     fomhd::matvec(operators.C[static_cast<std::size_t>(mode + 1)],
                                   fomhd::matvec(
                                       operators.G[static_cast<std::size_t>(mode + 1)],
                                       potential));
                 require(fomhd::max_abs(cg) < 2e-12,
                         "CG is nonzero: max_abs=" +
                             std::to_string(fomhd::max_abs(cg)));

                 const Vec electric = deterministic_vector(
                     Ne, static_cast<std::size_t>(mode + 9));
                 const Vec dbc =
                     fomhd::matvec(operators.DB[static_cast<std::size_t>(mode + 1)],
                                   fomhd::matvec(
                                       operators.C[static_cast<std::size_t>(mode + 1)],
                                       electric));
                 require(fomhd::max_abs(dbc) < 2e-12,
                         "DBC is nonzero: max_abs=" +
                             std::to_string(fomhd::max_abs(dbc)));
             }
         }},
        {"OP_02_divergence_theorem", [] {
             const Geometry geometry = make_rectangle_geometry();
             const auto operators = fomhd::build_operators(geometry, 0);
             const Vec flux = deterministic_vector(operators.M1.size(), 3);
             const Vec divergence = fomhd::matvec(operators.DJ.front(), flux);
             Complex volume{};
             for (std::size_t cell = 0; cell < divergence.size(); ++cell) {
                 volume += operators.M0[cell] * divergence[cell];
             }
             Complex boundary{};
             for (const auto& wall : geometry.walls) {
                 const std::size_t component =
                     wall.component == fomhd::WallComponent::r_face ? 0U : 2U;
                 const std::size_t component_offset =
                     component == 0U
                         ? 0U
                         : geometry.r_faces.keys.size() +
                               geometry.cells.keys.size();
                 boundary +=
                     static_cast<double>(wall.sign) * wall.area_per_radian *
                     flux[component_offset + wall.face];
             }
             near_complex(volume, boundary, 2e-13,
                          "discrete divergence theorem");
         }},
        {"OP_03_I_H_pairing", [] {
             const Geometry geometry = make_rectangle_geometry();
             const auto operators = fomhd::build_operators(geometry, 0);
             const std::size_t Nc = geometry.cells.keys.size();
             const Vec nodes = deterministic_vector(3 * Nc, 6);
             const Vec edges = deterministic_vector(operators.M1.size(), 12);
             const RealVec node_weights = [&] {
                 RealVec result(3 * Nc);
                 for (std::size_t component = 0; component < 3; ++component) {
                     std::copy(operators.M0.begin(), operators.M0.end(),
                               result.begin() +
                                   static_cast<std::ptrdiff_t>(component * Nc));
                 }
                 return result;
             }();
             const Complex left =
                 fomhd::weighted_dot(nodes, fomhd::matvec(operators.H, edges),
                                     node_weights);
             const Complex right =
                 fomhd::weighted_dot(fomhd::matvec(operators.I, nodes), edges,
                                     operators.M1);
             near_complex(left, right, 2e-12, "I/H weighted pairing");
         }},
        {"OP_04_noncommuting_transform_inverse", [] {
             const Geometry geometry = make_l_geometry();
             const auto operators = fomhd::build_operators(geometry, 0);
             const Matrix ordering_defect = matrix_subtract(
                 diagonal_product(operators.PR_v, operators.PZ_v),
                 diagonal_product(operators.PZ_v, operators.PR_v));
             require(fomhd::max_abs(ordering_defect) > 1e-8,
                     "irregular mask fixture must make PR/PZ noncommute");

             const Vec x = deterministic_vector(operators.MB.size(), 4);
             const Vec recovered =
                 fomhd::apply_t2_inverse(operators, fomhd::apply_t2(operators, x));
             require(abs_error(recovered, x) < 3e-12,
                     "T2 inverse roundtrip");
         }},
        {"OP_05_dirichlet_k0_full_rows", [] {
             const Geometry geometry = make_rectangle_geometry();
             const auto operators = fomhd::build_operators(geometry, 1);
             require(operators.Lphi.size() == 3,
                     "all retained modes must carry a phi operator");
             require(operators.Lphi[1].rows == geometry.cells.keys.size() &&
                         operators.Lphi[1].cols == geometry.cells.keys.size(),
                     "k=0 Dirichlet operator deleted charge rows");
             require(operators.phi_factor[1].valid,
                     "k=0 Dirichlet factorization is invalid");
             const Vec charge = deterministic_vector(geometry.cells.keys.size(), 11);
             const Vec potential =
                 fomhd::solve(operators.phi_factor[1], charge);
             const Vec residual = fomhd::subtract(
                 fomhd::matvec(operators.Lphi[1], potential), charge);
             require(fomhd::norm(residual) <
                         1e-11 * std::max(1.0, fomhd::norm(charge)),
                     "k=0 Dirichlet original residual");
         }},
        {"OP_06_wall_full_rows_and_constraints", [] {
             const Geometry geometry = make_rectangle_geometry();
             const auto operators = fomhd::build_operators(geometry, 1);
             require(operators.T_full.rows == geometry.walls.size(),
                     "full wall matrix dropped wall rows");
             require(operators.T.rows == operators.selected_wall_rows.size(),
                     "selected wall row map mismatch");
             require(operators.T.rows > 0, "no independent wall rows selected");
             const std::size_t modes = static_cast<std::size_t>(2 * operators.K + 1);
             require(operators.Bc_full.rows == modes * geometry.cells.keys.size(),
                     "all-mode B constraints are not block diagonal");
             require(operators.Bc.rows ==
                         operators.selected_constraint_rows.size(),
                     "selected B row map mismatch");
         }},
        {"OP_07_gradient_sine_inner_convergence", [] {
             const RealVec errors = [] {
                 RealVec result;
                 constexpr double r_lo = 0.5;
                 constexpr double r_hi = 1.0;
                 constexpr double z_lo = -0.25;
                 constexpr double z_hi = 0.25;
                 constexpr double inner_distance = 0.10;
                 const double a = 3.0;
                 const double b = 4.0;
                 for (int level = 0; level < 3; ++level) {
                     const std::size_t n_cells =
                         static_cast<std::size_t>(8U << level);
                     const double h_r =
                         (r_hi - r_lo) / static_cast<double>(n_cells);
                     const double h_z =
                         (z_hi - z_lo) / static_cast<double>(n_cells / 2);
                     const Geometry geometry = fomhd::build_rectangle(
                         n_cells + 1, n_cells / 2 + 1, r_lo + 0.5 * h_r,
                         z_lo + 0.5 * h_z, h_r, h_z);
                     const Matrix gradient_operator =
                         fomhd::build_gradient_operator(
                             geometry, 0,
                             fomhd::OperatorOptions{
                                 1e-10, 256U * 1024U * 1024U});
                     const std::size_t Nc = geometry.cells.keys.size();
                     Vec phi(Nc);
                     for (std::size_t cell = 0; cell < Nc; ++cell) {
                         phi[cell] = std::sin(a * geometry.cells.R[cell]) *
                                     std::sin(b * geometry.cells.Z[cell]);
                     }
                     const Vec gradient = fomhd::matvec(gradient_operator, phi);
                     double max_error = 0.0;
                     char worst_kind = ' ';
                     double worst_R = 0.0;
                     double worst_Z = 0.0;
                     Complex worst_actual{};
                     double worst_exact = 0.0;
                     for (std::size_t face = 0; face < geometry.r_faces.keys.size();
                          ++face) {
                         const double R = geometry.r_faces.R[face];
                         const double Z = geometry.r_faces.Z[face];
                         if (R - r_lo < inner_distance || r_hi - R < inner_distance ||
                             Z - z_lo < inner_distance || z_hi - Z < inner_distance) {
                             continue;
                         }
                         const double exact =
                             a * std::cos(a * R) * std::sin(b * Z);
                         const double error =
                             std::abs(gradient[face].real() - exact);
                         if (error > max_error) {
                             max_error = error;
                             worst_kind = 'R';
                             worst_R = R;
                             worst_Z = Z;
                             worst_actual = gradient[face];
                             worst_exact = exact;
                         }
                     }
                     for (std::size_t face = 0; face < geometry.z_faces.keys.size();
                          ++face) {
                         const double R = geometry.z_faces.R[face];
                         const double Z = geometry.z_faces.Z[face];
                         if (R - r_lo < inner_distance || r_hi - R < inner_distance ||
                             Z - z_lo < inner_distance || z_hi - Z < inner_distance) {
                             continue;
                         }
                         const double exact =
                             b * std::sin(a * R) * std::cos(b * Z);
                         const Complex actual =
                             gradient[geometry.r_faces.keys.size() + Nc + face];
                         const double error = std::abs(actual.real() - exact);
                         if (error > max_error) {
                             max_error = error;
                             worst_kind = 'Z';
                             worst_R = R;
                             worst_Z = Z;
                             worst_actual = actual;
                             worst_exact = exact;
                         }
                     }
                     require(max_error > 0.0 &&
                                 std::isfinite(max_error),
                             "nonempty inner convergence observations");
                     std::cerr << "CONVERGENCE level=" << level
                               << " error=" << max_error
                               << " kind=" << worst_kind
                               << " R=" << worst_R << " Z=" << worst_Z
                               << " actual=" << worst_actual.real()
                               << " exact=" << worst_exact << '\n';
                     result.push_back(max_error);
                 }
                 return result;
             }();
             const double order_1 =
                 std::log2(errors[0] / errors[1]);
             const double order_2 =
                 std::log2(errors[1] / errors[2]);
             require(order_1 > 3.5 && order_2 > 3.5,
                     "gradient inner convergence order is below fourth order: "
                     "errors=" +
                         std::to_string(errors[0]) + "," +
                         std::to_string(errors[1]) + "," +
                         std::to_string(errors[2]) + " orders=" +
                         std::to_string(order_1) + "," +
                         std::to_string(order_2));
         }},
        {"CAP_01_memory_rejection", [] {
             bool threw = false;
             try {
                 static_cast<void>(fomhd::materialize(
                     100, 100, [](const Vec&) { return Vec(100); },
                     fomhd::MemoryBudget{"tiny_matrix", 8}));
             } catch (const fomhd::NumericalError& error) {
                 threw = std::string(error.what()).find("tiny_matrix") !=
                         std::string::npos;
             }
             require(threw, "memory cap did not reject with its name");
         }},
    };
    return tests;
}

}  // namespace

int main(int argc, char** argv) {
    const auto& tests = registry();
    if (argc == 1) {
        int failures = 0;
        for (const auto& [name, test] : tests) {
            try {
                test();
                std::cout << "PASS " << name << '\n';
            } catch (const std::exception& error) {
                ++failures;
                std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            }
        }
        std::cout << "SUMMARY " << tests.size() - static_cast<std::size_t>(failures)
                  << "/" << tests.size() << " passed\n";
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    for (const auto& [name, test] : tests) {
        if (name == argv[1]) {
            try {
                test();
                std::cout << "PASS " << name << '\n';
                return EXIT_SUCCESS;
            } catch (const std::exception& error) {
                std::cerr << "FAIL " << name << ": " << error.what() << '\n';
                return EXIT_FAILURE;
            }
        }
    }
    std::cerr << "unknown test: " << argv[1] << '\n';
    return EXIT_FAILURE;
}
