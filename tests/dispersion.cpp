#include "periodic_fixture.hpp"

#include <mkl_lapacke.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fomhd::test {
namespace {

using Json = nlohmann::json;
using Vector3 = std::array<Complex, 3>;
using Matrix3 = std::array<Complex, 9>;

constexpr Complex imaginary_unit{0.0, 1.0};
constexpr double matrix_tolerance = 1.0e-5;
constexpr double sensitivity_tolerance = 1.0e-6;
constexpr double eigen_abs_tolerance = 2.0e-8;
constexpr double eigen_relative_tolerance = 2.0e-4;
constexpr double cgl_stable_growth_limit = 1.0e-2;
constexpr double cgl_unstable_growth_min = 1.0e-1;
constexpr double magnetic_divergence_tolerance = 1.0e-8;
constexpr double transverse_ampere_tolerance = 5.0e-3;

Complex matrix3(const Matrix3& matrix, std::size_t row, std::size_t column) {
  return matrix[3 * row + column];
}

Complex& matrix3(Matrix3& matrix, std::size_t row, std::size_t column) {
  return matrix[3 * row + column];
}

Complex dot(const Vector3& left, const Vector3& right) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

Vector3 add3(const Vector3& left, const Vector3& right) {
  return {left[0] + right[0], left[1] + right[1], left[2] + right[2]};
}

Vector3 scale3(const Vector3& value, Complex factor) {
  return {value[0] * factor, value[1] * factor, value[2] * factor};
}

Vector3 cross(const Vector3& left, const Vector3& right) {
  return {left[1] * right[2] - left[2] * right[1],
          left[2] * right[0] - left[0] * right[2],
          left[0] * right[1] - left[1] * right[0]};
}

Vector3 matrix_vector(const Matrix3& matrix, const Vector3& vector) {
  Vector3 result{};
  for (std::size_t row = 0; row < 3; ++row) {
    for (std::size_t column = 0; column < 3; ++column) {
      result[row] +=
          matrix3(matrix, row, column) * vector[column];
    }
  }
  return result;
}

Vector3 transverse_projection(const Vector3& value, const Vector3& wave) {
  const Complex wave_squared = dot(wave, wave);
  return add3(value, scale3(wave, -dot(wave, value) / wave_squared));
}

struct EigenResult {
  std::array<Complex, dispersion_dimension> eigenvalues{};
  std::array<std::array<Complex, dispersion_dimension>, dispersion_dimension>
      eigenvectors{};
  std::size_t count{dispersion_dimension};
};

struct ReferenceResult {
  DispersionMatrix matrix{};
  std::array<Complex, dispersion_dimension> eigenvalues{};
  std::array<std::array<Complex, dispersion_dimension>, dispersion_dimension>
      eigenvectors{};
  std::array<Complex, dispersion_dimension> unrestricted_eigenvalues{};
  std::size_t eigen_count{dispersion_dimension};
  double eigen_residual{};
};

Matrix dense_matrix(const DispersionMatrix& input) {
  Matrix result(dispersion_dimension, dispersion_dimension);
  for (std::size_t i = 0; i < dispersion_dimension; ++i)
    for (std::size_t j = 0; j < dispersion_dimension; ++j)
      result(i, j) = input[i][j];
  return result;
}

Matrix coordinate_transform(double theta) {
  Matrix transform = identity(dispersion_dimension);
  transform(0, 0) = transform(0, 5) = transform(5, 0) = std::sqrt(0.5);
  transform(5, 5) = -std::sqrt(0.5);
  for (const auto [radial, axial] :
       {std::pair{1u, 3u}, std::pair{6u, 8u}, std::pair{11u, 13u}}) {
    transform(radial, radial) = transform(axial, axial) = std::cos(theta);
    transform(radial, axial) = std::sin(theta);
    transform(axial, radial) = -std::sin(theta);
  }
  return transform;
}

Matrix project_matrix(const Matrix& input, const Matrix& basis) {
  Matrix result(basis.cols, basis.cols);
  for (std::size_t row = 0; row < basis.cols; ++row) {
    for (std::size_t column = 0; column < basis.cols; ++column) {
      std::complex<long double> entry{};
      for (std::size_t i = 0; i < input.rows; ++i) {
        for (std::size_t j = 0; j < input.cols; ++j) {
          entry += std::conj(std::complex<long double>{basis(i, row)}) *
                   std::complex<long double>{input(i, j)} *
                   std::complex<long double>{basis(j, column)};
        }
      }
      result(row, column) = Complex{static_cast<double>(entry.real()),
                                    static_cast<double>(entry.imag())};
    }
  }
  return result;
}

EigenResult diagonalize(const Matrix& input, const Matrix& lift) {
  const auto size = static_cast<lapack_int>(input.rows);
  std::vector<MKL_Complex16> matrix(input.rows * input.rows);
  std::vector<MKL_Complex16> values(input.rows);
  std::vector<MKL_Complex16> vectors(input.rows * input.rows);
  for (std::size_t column = 0; column < input.cols; ++column)
    for (std::size_t row = 0; row < input.rows; ++row)
      matrix[row + column * input.rows] =
          MKL_Complex16{input(row, column).real(), input(row, column).imag()};
  const lapack_int status = LAPACKE_zgeev(
      LAPACK_COL_MAJOR, 'N', 'V', size, matrix.data(), size, values.data(),
      nullptr, 1, vectors.data(), size);
  if (status != 0) {
    throw std::runtime_error("LAPACKE_zgeev failed with status " +
                             std::to_string(status));
  }
  EigenResult result;
  result.count = input.rows;
  for (std::size_t column = 0; column < input.cols; ++column) {
    result.eigenvalues[column] = Complex{values[column].real, values[column].imag};
    for (std::size_t row = 0; row < dispersion_dimension; ++row) {
      for (std::size_t inner = 0; inner < input.rows; ++inner) {
        const auto& value = vectors[inner + column * input.rows];
        result.eigenvectors[column][row] +=
            lift(row, inner) * Complex{value.real, value.imag};
      }
    }
    double vector_norm = 0.0;
    for (const Complex value : result.eigenvectors[column])
      vector_norm = std::hypot(vector_norm, std::abs(value));
    if (!(vector_norm > 0.0) || !std::isfinite(vector_norm)) {
      throw std::runtime_error("LAPACKE_zgeev returned a degenerate vector");
    }
    for (Complex& value : result.eigenvectors[column]) value /= vector_norm;
  }
  return result;
}

EigenResult eigen_decomposition(const DispersionMatrix& input, double theta) {
  const Matrix transform = coordinate_transform(theta);
  return diagonalize(project_matrix(dense_matrix(input), transform), transform);
}

struct AdmissibleSubspace {
  Matrix constraints;
  Matrix transform;
  Matrix basis;
  Matrix lift;
  double nullspace_residual{};
  double orthogonality_residual{};
};

AdmissibleSubspace admissible_subspace(const PeriodicFixture& fixture) {
  const auto& c = fixture.configuration();
  const auto& p = fixture.parameters();
  const auto& scales = fixture.variable_scales();
  AdmissibleSubspace result;
  result.constraints = Matrix(3, dispersion_dimension);
  const double sine = std::sin(c.theta), cosine = std::cos(c.theta);
  result.constraints(0, 11) = sine;
  result.constraints(0, 13) = cosine;
  result.constraints(1, 12) = -imaginary_unit;
  result.constraints(2, 11) = imaginary_unit * cosine;
  result.constraints(2, 13) = -imaginary_unit * sine;
  for (std::size_t species = 0; species < 2; ++species) {
    const std::size_t offset = species == 0 ? 1 : 6;
    const double coefficient = -p.mu0 * p.species[species].charge /
        (p.species[species].mass * c.k0 * c.b0);
    result.constraints(1, offset) = coefficient * cosine * scales[offset];
    result.constraints(1, offset + 2) = -coefficient * sine * scales[offset + 2];
    result.constraints(2, offset + 1) = coefficient * scales[offset + 1];
  }
  for (std::size_t row = 0; row < 3; ++row) {
    double row_norm = 0.0;
    for (std::size_t col = 0; col < dispersion_dimension; ++col)
      row_norm = std::hypot(row_norm, std::abs(result.constraints(row, col)));
    for (std::size_t col = 0; col < dispersion_dimension; ++col)
      result.constraints(row, col) /= row_norm;
  }
  result.transform = coordinate_transform(c.theta);
  const Matrix transformed_constraints = matmul(result.constraints, result.transform);
  std::array<MKL_Complex16, 3 * dispersion_dimension> data{};
  std::array<MKL_Complex16, dispersion_dimension * dispersion_dimension> vh{};
  std::array<double, 3> singular{};
  std::array<double, 2> superb{};
  for (std::size_t row = 0; row < 3; ++row)
    for (std::size_t col = 0; col < dispersion_dimension; ++col)
      data[row + 3 * col] = MKL_Complex16{transformed_constraints(row, col).real(),
                                        transformed_constraints(row, col).imag()};
  const lapack_int info = LAPACKE_zgesvd(LAPACK_COL_MAJOR, 'N', 'A', 3,
      static_cast<lapack_int>(dispersion_dimension), data.data(), 3,
      singular.data(), nullptr, 1, vh.data(),
      static_cast<lapack_int>(dispersion_dimension), superb.data());
  if (info != 0 || singular[2] < 1.0e-12 * singular[0]) {
    throw std::runtime_error("field constraint nullspace SVD failed");
  }
  result.basis = Matrix(dispersion_dimension, dispersion_dimension - 3);
  for (std::size_t row = 0; row < dispersion_dimension; ++row)
    for (std::size_t col = 0; col < result.basis.cols; ++col) {
      const auto& entry = vh[col + 3 + row * dispersion_dimension];
      result.basis(row, col) = Complex{entry.real, -entry.imag};
    }
  result.lift = matmul(result.transform, result.basis);
  result.nullspace_residual = max_abs(matmul(result.constraints, result.lift));
  const Matrix gram = matmul(adjoint(result.lift), result.lift);
  result.orthogonality_residual =
      max_abs(subtract(gram.data, identity(result.basis.cols).data));
  if (result.nullspace_residual > 4096 * std::numeric_limits<double>::epsilon() ||
      result.orthogonality_residual > 4096 * std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error("field constraint nullspace algebra failed");
  }
  return result;
}

EigenResult admissible_eigen_decomposition(const DispersionMatrix& input,
                                          const AdmissibleSubspace& subspace) {
  const Matrix conditioned = project_matrix(dense_matrix(input), subspace.transform);
  return diagonalize(project_matrix(conditioned, subspace.basis), subspace.lift);
}

double invariant_residual(const DispersionMatrix& input, const Matrix& constraints) {
  double maximum = 0.0;
  for (std::size_t row = 0; row < constraints.rows; ++row) {
    for (std::size_t col = 0; col < dispersion_dimension; ++col) {
      std::complex<long double> residual{};
      long double scale = 0.0;
      for (std::size_t inner = 0; inner < dispersion_dimension; ++inner) {
        const auto term = std::complex<long double>{constraints(row, inner)} *
                          std::complex<long double>{input[inner][col]};
        residual += term;
        scale += std::abs(term);
      }
      maximum = std::max(maximum,
          static_cast<double>(std::abs(residual) / std::max(1.0L, scale)));
    }
  }
  return maximum;
}

double eigen_residual(const DispersionMatrix& matrix,
                      Complex value,
                      const std::array<Complex, dispersion_dimension>& vector) {
  double residual = 0.0;
  double scale = std::max(1.0, std::abs(value));
  for (std::size_t row = 0; row < dispersion_dimension; ++row) {
    Complex product{};
    for (std::size_t column = 0; column < dispersion_dimension; ++column) {
      product += matrix[row][column] * vector[column];
    }
    residual = std::hypot(residual, std::abs(product - value * vector[row]));
    for (std::size_t column = 0; column < dispersion_dimension; ++column) {
      scale = std::max(scale, std::abs(matrix[row][column]));
    }
  }
  return residual / scale;
}

ReferenceResult analytic_reference(const PeriodicFixture& fixture,
                                   bool reverse_lorentz = false,
                                   bool reverse_anisotropic_stress = false) {
  const auto& configuration = fixture.configuration();
  const auto& parameters = fixture.parameters();
  const auto& scales = fixture.variable_scales();
  const auto& rates = fixture.rate_scales();
  const double ion_mass = parameters.species[0].mass;
  const double ion_charge = parameters.species[0].charge;
  const double electron_mass = parameters.species[1].mass;
  const double electron_charge = parameters.species[1].charge;
  const double ion_gamma = parameters.species[0].gamma;

  const Vector3 wave{
      configuration.k0 * std::sin(configuration.theta), Complex{},
      configuration.k0 * std::cos(configuration.theta)};
  const Vector3 background_field{Complex{}, Complex{},
                                 Complex{configuration.b0, 0.0}};
  const Vector3 direction{Complex{}, Complex{}, Complex{1.0, 0.0}};
  const double wave_squared = std::norm(wave[0]) + std::norm(wave[2]);
  if (!(wave_squared > 0.0)) {
    throw std::invalid_argument("analytic reference has zero wave vector");
  }

  DispersionMatrix physical{};
  for (std::size_t column = 0; column < dispersion_dimension; ++column) {
    std::array<Complex, dispersion_dimension> perturbation{};
    perturbation[column] = 1.0;

    const Vector3 ion_momentum{
        perturbation[1], perturbation[2], perturbation[3]};
    const Vector3 electron_momentum{
        perturbation[6], perturbation[7], perturbation[8]};
    const Vector3 ion_velocity =
        scale3(ion_momentum, 1.0 / (ion_mass * configuration.n0));
    const Vector3 electron_velocity =
        scale3(electron_momentum, 1.0 / (electron_mass * configuration.n0));

    const Matrix3 ion_stress = [&] {
      Matrix3 stress{};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        matrix3(stress, axis, axis) = perturbation[4];
      }
      return stress;
    }();

    const Vector3 magnetic_delta{
        perturbation[11], perturbation[12], perturbation[13]};
    const Vector3 direction_delta = scale3(
        add3(magnetic_delta,
             scale3(direction,
                    -dot(direction, magnetic_delta))),
        1.0 / configuration.b0);
    Matrix3 electron_stress{};
    for (std::size_t row = 0; row < 3; ++row) {
      for (std::size_t column_axis = 0; column_axis < 3; ++column_axis) {
        const Complex parallel_delta =
            perturbation[10] - perturbation[9];
        const double anisotropy =
            configuration.electron_parallel -
            configuration.electron_perp;
        const Complex stress_anisotropy =
            (reverse_anisotropic_stress ? -1.0 : 1.0) * anisotropy;
        matrix3(electron_stress, row, column_axis) =
            (row == column_axis ? perturbation[9] : Complex{}) +
            parallel_delta * direction[row] * direction[column_axis] +
            stress_anisotropy *
                (direction_delta[row] * direction[column_axis] +
                 direction[row] * direction_delta[column_axis]);
      }
    }

    const Vector3 ion_pressure_force = scale3(
        matrix_vector(ion_stress, wave), -imaginary_unit);
    const Vector3 electron_pressure_force = scale3(
        matrix_vector(electron_stress, wave), -imaginary_unit);
    const double lorentz_sign = reverse_lorentz ? -1.0 : 1.0;
    const Vector3 ion_force =
        add3(ion_pressure_force,
             scale3(cross(ion_velocity, background_field),
                    lorentz_sign * ion_charge * configuration.n0));
    const Vector3 electron_force =
        add3(electron_pressure_force,
             scale3(cross(electron_velocity, background_field),
                    lorentz_sign * electron_charge * configuration.n0));

    const Complex charge_density =
        ion_charge * perturbation[0] +
        electron_charge * perturbation[5];
    const Vector3 longitudinal_field = scale3(
        wave, -imaginary_unit * charge_density /
                  (parameters.epsilon0 * wave_squared));
    const Vector3 no_electric_current = add3(
        scale3(ion_force, ion_charge / ion_mass),
        scale3(electron_force, electron_charge / electron_mass));
    const double alpha =
        ion_charge * ion_charge * configuration.n0 / ion_mass +
        electron_charge * electron_charge * configuration.n0 /
            electron_mass;
    const Vector3 transverse_field = scale3(
        transverse_projection(no_electric_current, wave),
        -1.0 / (wave_squared / parameters.mu0 + alpha));
    const Vector3 electric_field =
        add3(longitudinal_field, transverse_field);

    const Vector3 ion_momentum_rate = add3(
        ion_force,
        scale3(electric_field, ion_charge * configuration.n0));
    const Vector3 electron_momentum_rate = add3(
        electron_force,
        scale3(electric_field, electron_charge * configuration.n0));
    const Complex ion_divergence = imaginary_unit * dot(wave, ion_velocity);
    const Complex electron_divergence =
        imaginary_unit * dot(wave, electron_velocity);
    const Complex electron_parallel_rate =
        imaginary_unit * wave[2] * electron_velocity[2];
    const Vector3 magnetic_rate = scale3(
        cross(wave, transverse_field), -imaginary_unit);

    physical[0][column] =
        -imaginary_unit * dot(wave, ion_momentum) / ion_mass;
    for (std::size_t component = 0; component < 3; ++component) {
      physical[1 + component][column] = ion_momentum_rate[component];
      physical[6 + component][column] = electron_momentum_rate[component];
      physical[11 + component][column] = magnetic_rate[component];
    }
    physical[4][column] =
        -ion_gamma * configuration.ion_pressure * ion_divergence;
    physical[5][column] =
        -imaginary_unit * dot(wave, electron_momentum) / electron_mass;
    physical[9][column] =
        configuration.electron_perp *
        (electron_parallel_rate - 2.0 * electron_divergence);
    physical[10][column] =
        -configuration.electron_parallel *
        (electron_divergence + 2.0 * electron_parallel_rate);
  }

  ReferenceResult result;
  for (std::size_t row = 0; row < dispersion_dimension; ++row) {
    for (std::size_t column = 0; column < dispersion_dimension; ++column) {
      result.matrix[row][column] =
          physical[row][column] * scales[column] / rates[row];
    }
  }
  result.unrestricted_eigenvalues =
      eigen_decomposition(result.matrix, configuration.theta).eigenvalues;
  const EigenResult eigen =
      admissible_eigen_decomposition(result.matrix, admissible_subspace(fixture));
  result.eigen_count = eigen.count;
  result.eigenvalues = eigen.eigenvalues;
  result.eigenvectors = eigen.eigenvectors;
  for (std::size_t mode = 0; mode < eigen.count; ++mode) {
    result.eigen_residual =
        std::max(result.eigen_residual,
                 eigen_residual(result.matrix, eigen.eigenvalues[mode],
                                eigen.eigenvectors[mode]));
  }
  return result;
}

struct MatrixDifference {
  double maximum{};
  double rms{};
  double normalized_maximum{};
  std::size_t row{};
  std::size_t column{};
  Complex actual{};
  Complex expected{};
};

MatrixDifference matrix_difference(const DispersionMatrix& actual,
                                   const DispersionMatrix& expected) {
  MatrixDifference result;
  double sum{};
  for (std::size_t row = 0; row < dispersion_dimension; ++row) {
    for (std::size_t column = 0; column < dispersion_dimension; ++column) {
      const double difference =
          std::abs(actual[row][column] - expected[row][column]);
      const double normalized =
          difference / std::max(1.0, std::abs(expected[row][column]));
      result.maximum = std::max(result.maximum, difference);
      sum += difference * difference;
      if (normalized > result.normalized_maximum) {
        result.normalized_maximum = normalized;
        result.row = row;
        result.column = column;
        result.actual = actual[row][column];
        result.expected = expected[row][column];
      }
    }
  }
  result.rms =
      std::sqrt(sum / static_cast<double>(dispersion_dimension *
                                         dispersion_dimension));
  return result;
}

DispersionMatrix average_matrix(const DispersionMatrix& first,
                                const DispersionMatrix& second) {
  DispersionMatrix result{};
  for (std::size_t row = 0; row < dispersion_dimension; ++row) {
    for (std::size_t column = 0; column < dispersion_dimension; ++column) {
      result[row][column] =
          0.5 * (first[row][column] + second[row][column]);
    }
  }
  return result;
}

struct EigenMatch {
  std::array<Complex, dispersion_dimension> production{};
  std::array<std::size_t, dispersion_dimension> production_indices{};
  std::array<double, dispersion_dimension> allowed_error{};
  double maximum_difference{};
  double maximum_normalized_error{};
  bool pass{};
};

EigenMatch match_eigenvalues(
    const std::array<Complex, dispersion_dimension>& analytic,
    const std::array<Complex, dispersion_dimension>& production,
    std::size_t count = dispersion_dimension) {
  EigenMatch result;
  constexpr std::size_t size = dispersion_dimension;
  const std::size_t n = count;
  std::array<std::array<double, size + 1>, size + 1> cost{};
  for (std::size_t row = 1; row <= n; ++row) {
    for (std::size_t column = 1; column <= n; ++column) {
      const double allowed = eigen_abs_tolerance +
                             eigen_relative_tolerance * std::abs(analytic[row - 1]);
      cost[row][column] =
          std::abs(production[column - 1] - analytic[row - 1]) / allowed;
    }
  }
  std::array<double, size + 1> potential_row{};
  std::array<double, size + 1> potential_column{};
  std::array<std::size_t, size + 1> matched_row{};
  std::array<std::size_t, size + 1> predecessor{};
  for (std::size_t row = 1; row <= n; ++row) {
    matched_row[0] = row;
    std::size_t column0 = 0;
    std::array<double, size + 1> minimum;
    minimum.fill(std::numeric_limits<double>::infinity());
    std::array<bool, size + 1> used{};
    do {
      used[column0] = true;
      const std::size_t row0 = matched_row[column0];
      double delta = std::numeric_limits<double>::infinity();
      std::size_t column1 = 0;
      for (std::size_t column = 1; column <= n; ++column) {
        if (used[column]) {
          continue;
        }
        const double reduced =
            cost[row0][column] - potential_row[row0] - potential_column[column];
        if (reduced < minimum[column]) {
          minimum[column] = reduced;
          predecessor[column] = column0;
        }
        if (minimum[column] < delta) {
          delta = minimum[column];
          column1 = column;
        }
      }
      for (std::size_t column = 0; column <= n; ++column) {
        if (used[column]) {
          potential_row[matched_row[column]] += delta;
          potential_column[column] -= delta;
        } else {
          minimum[column] -= delta;
        }
      }
      column0 = column1;
    } while (matched_row[column0] != 0);
    do {
      const std::size_t previous = predecessor[column0];
      matched_row[column0] = matched_row[previous];
      column0 = previous;
    } while (column0 != 0);
  }
  std::array<std::size_t, size> analytic_to_production{};
  for (std::size_t column = 1; column <= n; ++column) {
    analytic_to_production[matched_row[column] - 1] = column - 1;
  }
  for (std::size_t mode = 0; mode < n; ++mode) {
    const std::size_t matched = analytic_to_production[mode];
    const double difference =
        std::abs(production[matched] - analytic[mode]);
    result.production_indices[mode] = matched;
    result.production[mode] = production[matched];
    result.allowed_error[mode] =
        eigen_abs_tolerance +
        eigen_relative_tolerance * std::abs(analytic[mode]);
    result.maximum_difference =
        std::max(result.maximum_difference, difference);
    result.maximum_normalized_error =
        std::max(result.maximum_normalized_error,
                 difference / result.allowed_error[mode]);
  }
  result.pass = result.maximum_normalized_error <= 1.0;
  return result;
}

Json complex_json(Complex value) {
  return Json{{"real", value.real()}, {"imag", value.imag()}};
}

Json matrix_json(const DispersionMatrix& matrix) {
  Json result = Json::array();
  for (const auto& row : matrix) {
    Json values = Json::array();
    for (const Complex value : row) {
      values.push_back(complex_json(value));
    }
    result.push_back(std::move(values));
  }
  return result;
}

Json adapter_checks_json(const AdapterChecks& checks) {
  return {
      {"plane_wave_derivative", checks.plane_wave_derivative},
      {"derivative_imaginary", checks.derivative_imaginary},
      {"derivative_skew", checks.derivative_skew},
      {"derivative_constant_mode", checks.derivative_constant_mode},
      {"direct_curl_action", checks.direct_curl_action},
      {"curl_gradient_identity", checks.cg_identity},
      {"divergence_curl_identity", checks.divergence_curl_identity},
      {"curl_adjoint_identity", checks.curl_adjoint_identity},
      {"current_divergence_minus_gradient_adjoint",
       checks.current_divergence_from_gradient},
      {"tolerance", checks.tolerance},
      {"passed", checks.passed},
  };
}

Json vector_json(const std::array<Complex, dispersion_dimension>& values) {
  Json result = Json::array();
  for (std::size_t component = 0; component < dispersion_dimension;
       ++component) {
    result.push_back(
        {{"component", dispersion_names[component]},
         {"real", values[component].real()},
         {"imag", values[component].imag()}});
  }
  return result;
}

Json spectrum_json(const std::array<Complex, dispersion_dimension>& values,
                   double reference_frequency) {
  Json result = Json::array();
  for (std::size_t mode = 0; mode < values.size(); ++mode) {
    result.push_back({{"mode", mode}, {"normalized_eigenvalue", complex_json(values[mode])},
        {"growth_s^-1", values[mode].real() * reference_frequency},
        {"angular_frequency_rad_s^-1", values[mode].imag() * reference_frequency}});
  }
  return result;
}

Json modes_json(const ReferenceResult& reference, const EigenResult& production,
                const EigenMatch& match, const PeriodicFixture& fixture) {
  const double reference_frequency = fixture.reference_frequency();
  std::vector<std::size_t> order(reference.eigen_count);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](std::size_t left,
                                            std::size_t right) {
    const Complex a = reference.eigenvalues[left];
    const Complex b = reference.eigenvalues[right];
    if (a.imag() != b.imag()) {
      return std::abs(a.imag()) > std::abs(b.imag());
    }
    return a.real() > b.real();
  });
  Json modes = Json::array();
  for (const std::size_t mode : order) {
    auto analytic_physical = reference.eigenvectors[mode];
    auto production_physical = production.eigenvectors[match.production_indices[mode]];
    for (std::size_t component = 0; component < dispersion_dimension; ++component) {
      analytic_physical[component] *= fixture.variable_scales()[component];
      production_physical[component] *= fixture.variable_scales()[component];
    }
    modes.push_back(
        {{"analytic_normalized_eigenvalue",
          complex_json(reference.eigenvalues[mode])},
         {"production_normalized_eigenvalue",
          complex_json(match.production[mode])},
         {"analytic_growth_s^-1",
          reference.eigenvalues[mode].real() * reference_frequency},
         {"analytic_angular_frequency_rad_s^-1",
          reference.eigenvalues[mode].imag() * reference_frequency},
         {"analytic_ordinary_frequency_hz",
          reference.eigenvalues[mode].imag() * reference_frequency /
              (2.0 * std::numbers::pi)},
         {"production_growth_s^-1",
          match.production[mode].real() * reference_frequency},
         {"production_angular_frequency_rad_s^-1",
          match.production[mode].imag() * reference_frequency},
         {"production_ordinary_frequency_hz",
          match.production[mode].imag() * reference_frequency /
              (2.0 * std::numbers::pi)},
         {"allowed_eigen_error_s^-1",
          match.allowed_error[mode] * reference_frequency},
         {"analytic_normalized_polarization", vector_json(reference.eigenvectors[mode])},
         {"production_normalized_polarization",
          vector_json(production.eigenvectors[match.production_indices[mode]])},
         {"analytic_SI_polarization", vector_json(analytic_physical)},
         {"production_SI_polarization", vector_json(production_physical)}});
  }
  return modes;
}

struct ConstraintMetrics {
  std::size_t mode{};
  Complex normalized_eigenvalue{};
  double physical_growth{};
  double physical_angular_frequency{};
  double magnetic_divergence_normalized{};
  double transverse_ampere_normalized{};
  bool passed{};
};

ConstraintMetrics constraint_metrics(
    const PeriodicFixture& fixture, Complex eigenvalue,
    const std::array<Complex, dispersion_dimension>& normalized_polarization) {
  const auto& parameters = fixture.parameters();
  const auto& configuration = fixture.configuration();
  const double ion_mass = parameters.species[0].mass;
  const double ion_charge = parameters.species[0].charge;
  const double electron_mass = parameters.species[1].mass;
  const double electron_charge = parameters.species[1].charge;
  const Vector3 wave{
      configuration.k0 * std::sin(configuration.theta), Complex{},
      configuration.k0 * std::cos(configuration.theta)};

  ConstraintMetrics result;
  result.normalized_eigenvalue = eigenvalue;
  auto polarization = normalized_polarization;
  for (std::size_t component = 0; component < dispersion_dimension; ++component) {
    polarization[component] *= fixture.variable_scales()[component];
  }
  const Vector3 ion_momentum{
      polarization[1], polarization[2], polarization[3]};
  const Vector3 electron_momentum{
      polarization[6], polarization[7], polarization[8]};
  const Vector3 ion_velocity =
      scale3(ion_momentum, 1.0 / (ion_mass * configuration.n0));
  const Vector3 electron_velocity =
      scale3(electron_momentum, 1.0 / (electron_mass * configuration.n0));
  const Vector3 current = add3(
      scale3(ion_velocity, ion_charge * configuration.n0),
      scale3(electron_velocity, electron_charge * configuration.n0));
  const Vector3 magnetic_field{
      polarization[11], polarization[12], polarization[13]};
  const Vector3 transverse_current =
      transverse_projection(current, wave);
  const Complex magnetic_divergence =
      imaginary_unit * dot(wave, magnetic_field);
  const Vector3 ampere_residual = add3(
      scale3(cross(wave, magnetic_field), imaginary_unit),
      scale3(transverse_current, -parameters.mu0));
  const double magnetic_scale =
      configuration.k0 * std::sqrt(std::norm(magnetic_field[0]) +
                                    std::norm(magnetic_field[1]) +
                                    std::norm(magnetic_field[2]));
  const double current_scale = std::max(
      {std::numeric_limits<double>::min(), magnetic_scale,
       parameters.mu0 * std::sqrt(std::norm(transverse_current[0]) +
                                  std::norm(transverse_current[1]) +
                                  std::norm(transverse_current[2]))});
  result.magnetic_divergence_normalized =
      std::abs(magnetic_divergence) /
      std::max(std::numeric_limits<double>::min(), magnetic_scale);
  result.transverse_ampere_normalized =
      std::sqrt(std::norm(ampere_residual[0]) +
                std::norm(ampere_residual[1]) +
                std::norm(ampere_residual[2])) /
      current_scale;
  result.physical_growth =
      result.normalized_eigenvalue.real() * fixture.reference_frequency();
  result.physical_angular_frequency =
      result.normalized_eigenvalue.imag() * fixture.reference_frequency();
  result.passed =
      std::abs(eigenvalue) > 10.0 * eigen_abs_tolerance &&
      result.magnetic_divergence_normalized <=
          magnetic_divergence_tolerance &&
      result.transverse_ampere_normalized <=
          transverse_ampere_tolerance;
  return result;
}

std::size_t select_magnetic_branch(const PeriodicFixture& fixture,
                                   const ReferenceResult& reference) {
  const bool unstable = fixture.configuration().expected_cgl_stability == "unstable";
  double best_score = -std::numeric_limits<double>::infinity();
  std::size_t selected = dispersion_dimension;
  for (std::size_t mode = 0; mode < reference.eigen_count; ++mode) {
    if (std::abs(reference.eigenvalues[mode]) <= 10.0 * eigen_abs_tolerance) {
      continue;
    }
    const auto& vector = reference.eigenvectors[mode];
    const double score = unstable ? reference.eigenvalues[mode].real()
        : std::norm(vector[11]) + std::norm(vector[12]) + std::norm(vector[13]);
    if (score > best_score) {
      best_score = score;
      selected = mode;
    }
  }
  if (selected == dispersion_dimension) {
    throw std::runtime_error("no nonzero magnetic branch is available");
  }
  return selected;
}

Json constraint_json(const ConstraintMetrics& metrics) {
  return {{"normalized_eigenvalue", complex_json(metrics.normalized_eigenvalue)},
          {"growth_s^-1", metrics.physical_growth},
          {"angular_frequency_rad_s^-1", metrics.physical_angular_frequency},
          {"magnetic_divergence_normalized", metrics.magnetic_divergence_normalized},
          {"transverse_ampere_normalized", metrics.transverse_ampere_normalized},
          {"passed", metrics.passed}};
}

Json compare_case(const DispersionCase& configuration) {
  PeriodicFixture fixture(configuration);
  const ReferenceResult reference = analytic_reference(fixture);
  const DispersionMatrix production_large =
      fixture.production_matrix(2.0e-7);
  const DispersionMatrix production_small =
      fixture.production_matrix(1.0e-7);
  const MatrixDifference sensitivity =
      matrix_difference(production_large, production_small);
  const DispersionMatrix production =
      average_matrix(production_large, production_small);
  const MatrixDifference difference =
      matrix_difference(production, reference.matrix);
  const AdmissibleSubspace subspace = admissible_subspace(fixture);
  const EigenResult unrestricted_production =
      eigen_decomposition(production, configuration.theta);
  const EigenResult production_eigen =
      admissible_eigen_decomposition(production, subspace);
  const EigenMatch eigen =
      match_eigenvalues(reference.eigenvalues, production_eigen.eigenvalues,
                        reference.eigen_count);
  const double analytic_invariance = invariant_residual(reference.matrix, subspace.constraints);
  const double production_invariance = invariant_residual(production, subspace.constraints);
  const bool invariant_pass = analytic_invariance <= magnetic_divergence_tolerance &&
                             production_invariance <= magnetic_divergence_tolerance;
  double production_eigen_residual{};
  for (std::size_t mode = 0; mode < production_eigen.count; ++mode) {
    production_eigen_residual =
        std::max(production_eigen_residual,
                 eigen_residual(production, production_eigen.eigenvalues[mode],
                                production_eigen.eigenvectors[mode]));
  }

  const ReferenceResult reversed_lorentz =
      analytic_reference(fixture, true, false);
  const ReferenceResult reversed_stress =
      analytic_reference(fixture, false, true);
  const double lorentz_control_distance =
      matrix_difference(production, reversed_lorentz.matrix)
          .normalized_maximum;
  const double stress_control_distance =
      matrix_difference(production, reversed_stress.matrix)
          .normalized_maximum;
  PeriodicFixture reversed_curl_fixture(configuration, true);
  const DispersionMatrix reversed_curl_production =
      reversed_curl_fixture.production_matrix(1.0e-7);
  const double reversed_curl_production_distance =
      matrix_difference(reversed_curl_production, reference.matrix)
          .normalized_maximum;

  const bool matrix_pass = difference.normalized_maximum <= matrix_tolerance;
  const bool sensitivity_pass =
      sensitivity.normalized_maximum <= sensitivity_tolerance;
  const bool anisotropic =
      std::abs(configuration.electron_perp -
               configuration.electron_parallel) > 1.0e-12;
  const bool controls_pass =
      lorentz_control_distance > matrix_tolerance &&
      (!anisotropic || stress_control_distance > matrix_tolerance) &&
      reversed_curl_production_distance > matrix_tolerance;
  double maximum_growth = -std::numeric_limits<double>::infinity();
  double maximum_production_growth =
      -std::numeric_limits<double>::infinity();
  double maximum_absolute_eigenvalue = 0.0;
  for (const Complex eigenvalue : reference.eigenvalues) {
    maximum_growth = std::max(maximum_growth, eigenvalue.real());
    maximum_absolute_eigenvalue =
        std::max(maximum_absolute_eigenvalue, std::abs(eigenvalue));
  }
  for (const Complex eigenvalue : eigen.production) {
    maximum_production_growth =
        std::max(maximum_production_growth, eigenvalue.real());
  }
  maximum_growth *= fixture.reference_frequency();
  maximum_production_growth *= fixture.reference_frequency();
  bool stability_pass = true;
  if (configuration.expected_cgl_stability == "stable") {
    stability_pass =
        maximum_growth <= cgl_stable_growth_limit &&
        maximum_production_growth <= cgl_stable_growth_limit;
  } else if (configuration.expected_cgl_stability == "unstable") {
    stability_pass =
        maximum_growth >= cgl_unstable_growth_min &&
        maximum_production_growth >= cgl_unstable_growth_min;
  }
  std::array<Complex, dispersion_dimension> zero_slow_production =
      production_eigen.eigenvalues;
  std::size_t slow_modes_zeroed = 0;
  for (std::size_t mode = 0; mode < reference.eigen_count; ++mode) {
    if (std::abs(reference.eigenvalues[mode]) > 10.0 * eigen_abs_tolerance &&
        std::abs(reference.eigenvalues[mode]) <
        0.01 * maximum_absolute_eigenvalue) {
      zero_slow_production[eigen.production_indices[mode]] = Complex{};
      ++slow_modes_zeroed;
    }
  }
  const EigenMatch zero_slow_match = match_eigenvalues(
      reference.eigenvalues, zero_slow_production, reference.eigen_count);
  const bool zero_slow_control_pass =
      slow_modes_zeroed > 0 && !zero_slow_match.pass;
  ConstraintMetrics constraints;
  ConstraintMetrics production_constraints;
  const bool constraints_required =
      configuration.expected_cgl_stability != "unspecified";
  if (constraints_required) {
    const std::size_t mode = select_magnetic_branch(fixture, reference);
    constraints = constraint_metrics(fixture, reference.eigenvalues[mode],
                                     reference.eigenvectors[mode]);
    constraints.mode = mode;
    const std::size_t production_mode = eigen.production_indices[mode];
    production_constraints = constraint_metrics(
        fixture, production_eigen.eigenvalues[production_mode],
        production_eigen.eigenvectors[production_mode]);
  }
  const bool constraints_pass =
      !constraints_required || (constraints.passed && production_constraints.passed);
  const bool pass =
      matrix_pass && sensitivity_pass && eigen.pass && controls_pass &&
      stability_pass && zero_slow_control_pass && constraints_pass && invariant_pass &&
      reference.eigen_residual <= magnetic_divergence_tolerance &&
      production_eigen_residual <= magnetic_divergence_tolerance;

  Json result = {
      {"case", configuration.name},
      {"scope", configuration.scope},
      {"points", configuration.points},
      {"k0_m^-1", configuration.k0},
      {"theta_rad", configuration.theta},
      {"k_R_m^-1", configuration.k0 * std::sin(configuration.theta)},
      {"k_Z_m^-1", configuration.k0 * std::cos(configuration.theta)},
      {"n0_m^-3", configuration.n0},
      {"ion_pressure_Pa", configuration.ion_pressure},
      {"electron_perp_Pa", configuration.electron_perp},
      {"electron_parallel_Pa", configuration.electron_parallel},
      {"B0_T", configuration.b0},
      {"expected_cgl_stability", configuration.expected_cgl_stability},
      {"state_scales_SI", fixture.variable_scales()},
      {"rate_scales_SI_per_s", fixture.rate_scales()},
      {"analytic_maximum_growth_s^-1", maximum_growth},
      {"production_maximum_growth_s^-1", maximum_production_growth},
      {"stable_growth_limit_s^-1", cgl_stable_growth_limit},
      {"unstable_growth_min_s^-1", cgl_unstable_growth_min},
      {"stability_classification_pass", stability_pass},
      {"reference_frequency_s^-1", fixture.reference_frequency()},
      {"relative_amplitudes", {1.0e-7, 2.0e-7}},
      {"linearity_sensitivity_max", sensitivity.normalized_maximum},
      {"linearity_sensitivity_tolerance", sensitivity_tolerance},
      {"matrix_discrepancy_abs_max", difference.maximum},
      {"matrix_discrepancy_rms_abs", difference.rms},
      {"matrix_discrepancy_normalized_max",
       difference.normalized_maximum},
      {"matrix_tolerance_normalized", matrix_tolerance},
      {"worst_matrix_entry",
       {{"row", dispersion_names[difference.row]},
        {"column", dispersion_names[difference.column]},
        {"production", complex_json(difference.actual)},
        {"analytic", complex_json(difference.expected)}}},
      {"eigen_discrepancy_abs_max_normalized",
       eigen.maximum_difference},
      {"eigen_discrepancy_normalized_max",
       eigen.maximum_normalized_error},
      {"eigen_abs_tolerance_normalized", eigen_abs_tolerance},
      {"eigen_relative_tolerance_normalized",
       eigen_relative_tolerance},
      {"analytic_eigen_residual", reference.eigen_residual},
      {"production_eigen_residual", production_eigen_residual},
      {"admissible_eigen_dimension", reference.eigen_count},
      {"field_invariant_count", subspace.constraints.rows},
      {"admissible_nullspace_residual", subspace.nullspace_residual},
      {"admissible_orthogonality_residual", subspace.orthogonality_residual},
      {"analytic_invariant_residual", analytic_invariance},
      {"production_invariant_residual", production_invariance},
      {"invariant_preservation_pass", invariant_pass},
      {"unrestricted_analytic_eigenvalues_diagnostic_only",
       spectrum_json(reference.unrestricted_eigenvalues, fixture.reference_frequency())},
      {"unrestricted_production_eigenvalues_diagnostic_only",
       spectrum_json(unrestricted_production.eigenvalues, fixture.reference_frequency())},
      {"reversed_lorentz_control_distance", lorentz_control_distance},
      {"reversed_anisotropic_stress_control_distance",
       stress_control_distance},
      {"reversed_curl_production_control_distance",
       reversed_curl_production_distance},
      {"controls_pass", controls_pass},
      {"zero_slow_branches_count", slow_modes_zeroed},
      {"zero_slow_branches_rejected", zero_slow_control_pass},
      {"field_constraints_required", constraints_required},
      {"field_constraints_pass", constraints_pass},
      {"selected_branch_mode", constraints.mode},
      {"selected_branch_physical_growth_s^-1",
       constraints.physical_growth},
      {"selected_branch_physical_angular_frequency_rad_s^-1",
       constraints.physical_angular_frequency},
      {"selected_branch_magnetic_divergence_normalized",
       constraints.magnetic_divergence_normalized},
      {"selected_branch_transverse_ampere_normalized",
       constraints.transverse_ampere_normalized},
      {"selected_analytic_branch_constraints", constraint_json(constraints)},
      {"selected_production_branch_constraints", constraint_json(production_constraints)},
      {"production_stage_evaluations", fixture.evaluations()},
      {"hard_stage_checks_checked", fixture.hard_checks_checked()},
      {"empty_wall_checks_checked", fixture.empty_wall_checks()},
      {"reversed_curl_stage_evaluations", reversed_curl_fixture.evaluations()},
      {"reversed_curl_hard_checks_checked", reversed_curl_fixture.hard_checks_checked()},
      {"reversed_curl_empty_wall_checks_checked", reversed_curl_fixture.empty_wall_checks()},
      {"modes", modes_json(reference, production_eigen, eigen, fixture)},
      {"analytic_matrix", matrix_json(reference.matrix)},
      {"production_matrix", matrix_json(production)},
      {"passed", pass},
  };
  return result;
}

Json acceptance_gate_regressions() {
  DispersionMatrix expected{};
  DispersionMatrix actual{};
  expected[0][0] = 1.0e6;
  actual[0][0] = 1.0e6 + 1.0;
  actual[1][1] = 0.1;
  const MatrixDifference discrepancy = matrix_difference(actual, expected);
  std::array<Complex, dispersion_dimension> eigenvalues{};
  eigenvalues[0] = Complex{0.0, 1.0e4};
  eigenvalues[1] = Complex{0.0, -1.0e4};
  eigenvalues[2] = Complex{0.0, 1.0e-5};
  eigenvalues[3] = Complex{0.0, -1.0e-5};
  eigenvalues[4] = eigenvalues[2];
  auto permuted = eigenvalues;
  std::rotate(permuted.begin(), permuted.begin() + 5, permuted.end());
  auto zeroed = eigenvalues;
  zeroed[2] = zeroed[3] = zeroed[4] = Complex{};
  const bool absolute_maximum = discrepancy.maximum == 1.0;
  const bool normalized_maximum = discrepancy.normalized_maximum == 0.1;
  const bool permutation = match_eigenvalues(eigenvalues, permuted).pass;
  const bool slow_rejected = !match_eigenvalues(eigenvalues, zeroed).pass;
  DispersionCase configuration;
  PeriodicFixture fixture(configuration);
  std::array<Complex, dispersion_dimension> polarization{};
  const auto& parameters = fixture.parameters();
  const auto& scales = fixture.variable_scales();
  polarization[11] = 1.0;
  polarization[13] = 1.0e-12;
  polarization[2] = Complex{0.0, configuration.k0 * configuration.b0 / parameters.mu0}
      * parameters.species[0].mass / parameters.species[0].charge / scales[2];
  const auto admissible = constraint_metrics(fixture, 1.0, polarization);
  polarization[13] = 1.0;
  const auto divergent = constraint_metrics(fixture, 1.0, polarization);
  polarization[13] = 0.0;
  polarization[2] = 0.0;
  const auto wrong_current = constraint_metrics(fixture, 1.0, polarization);
  const bool constraints = admissible.passed &&
      std::abs(admissible.magnetic_divergence_normalized - 1.0e-12) < 1.0e-24 &&
      !divergent.passed && !wrong_current.passed;
  const AdmissibleSubspace subspace = admissible_subspace(fixture);
  DispersionMatrix violated_invariant{};
  violated_invariant[13][0] = 1.0;
  const bool invariant_rejected =
      invariant_residual(violated_invariant, subspace.constraints) >
      magnetic_divergence_tolerance;
  configuration.theta = 1.2;
  configuration.k0 = 0.03;
  configuration.b0 = 1.0e-4;
  configuration.electron_perp = 2.0;
  configuration.electron_parallel = 0.2;
  configuration.expected_cgl_stability = "unstable";
  PeriodicFixture oblique(configuration);
  const ReferenceResult oblique_reference = analytic_reference(oblique);
  const std::size_t mode = select_magnetic_branch(oblique, oblique_reference);
  const bool oblique_admissible = constraint_metrics(
      oblique, oblique_reference.eigenvalues[mode],
      oblique_reference.eigenvectors[mode]).passed;
  return {{"true_absolute_matrix_maximum", absolute_maximum},
          {"componentwise_normalized_matrix_maximum", normalized_maximum},
          {"degenerate_eigenvalue_permutation", permutation},
          {"zero_slow_modes_rejected", slow_rejected},
          {"magnetic_constraint_normalization_and_rejections", constraints},
          {"broken_full_matrix_invariant_rejected", invariant_rejected},
          {"oblique_reference_eigenvector_admissible", oblique_admissible},
          {"passed", absolute_maximum && normalized_maximum &&
                         permutation && slow_rejected && constraints &&
                         invariant_rejected && oblique_admissible}};
}

std::string run_local_dispersion(bool preflight) {
  Json adapter_results = Json::array();
  for (const int points : {9, 11}) {
    for (const double theta : {0.0, 0.5 * std::numbers::pi, 0.65}) {
      const Geometry geometry = periodic_geometry(points);
      const Operators operators =
          periodic_operators(geometry, points, 0.03, theta, false);
      const AdapterChecks checks =
          validate_periodic_adapter(operators, points, 0.03, theta);
      Json record = adapter_checks_json(checks);
      record["points"] = points;
      record["theta_rad"] = theta;
      adapter_results.push_back(std::move(record));
      if (!checks.passed) {
        throw std::runtime_error(
            "periodic adapter algebra gate failed before stage evaluation: "
            "plane_wave=" + std::to_string(checks.plane_wave_derivative) +
            " curl_action=" + std::to_string(checks.direct_curl_action) +
            " CG=" + std::to_string(checks.cg_identity) +
            " DBC=" + std::to_string(checks.divergence_curl_identity) +
            " Cdag=" + std::to_string(checks.curl_adjoint_identity) +
            " DJ+Gadj=" +
            std::to_string(checks.current_divergence_from_gradient) +
            " tolerance=" + std::to_string(checks.tolerance));
      }
    }
  }
  const Json acceptance = acceptance_gate_regressions();
  if (preflight) {
    return Json{{"acceptance_gate_regressions", acceptance},
                {"adapter_algebra_checks", adapter_results},
                {"passed", acceptance.at("passed")}}.dump(2);
  }
  const std::vector<DispersionCase> cases{
      {"warm_parallel_isotropic",
       "Uniform warm ion/electron background with k parallel to B0.",
       9, 1.0, 0.0, 1.0e16, 0.40, 0.40, 0.40, 0.10, "unspecified"},
      {"warm_perpendicular_isotropic",
       "Uniform warm ion/electron background with k perpendicular to B0.",
       9, 1.0, 0.5 * std::numbers::pi, 1.0e16, 0.40, 0.40, 0.40, 0.10,
       "unspecified"},
      {"warm_oblique_anisotropic",
       "Oblique warm two-fluid response with independent CGL pressures.",
       9, 1.2, 0.55, 1.0e16, 0.48, 0.26, 0.56, 0.10,
       "unspecified"},
      {"cgl_firehose_stable_control",
       "Low-k CGL stable control: p_perp=p_parallel=0.2 Pa, B0=1e-4 T, k0=0.03 m^-1; no kinetic threshold claim.",
       9, 0.03, 0.0, 1.0e16, 0.40, 0.20, 0.20, 1.0e-4, "stable"},
      {"cgl_firehose_unstable",
       "Low-k CGL parallel-pressure branch: p_parallel=2 Pa exceeds p_perp=0.2 Pa at B0=1e-4 T and k0=0.03 m^-1; this is a fluid CGL result, not a kinetic-threshold claim.",
       9, 0.03, 0.0, 1.0e16, 0.40, 0.20, 2.0, 1.0e-4, "unstable"},
      {"cgl_mirror_like_oblique",
       "Oblique CGL compressive instability: p_perp=2 Pa, p_parallel=0.2 Pa, theta=1.2 rad; the long-wavelength fluid compressive stiffness has a negative eigenvalue. This is not a kinetic mirror threshold.",
       9, 0.03, 1.2, 1.0e16, 0.40, 2.0, 0.20, 1.0e-4, "unstable"},
      {"cgl_mirror_like_stable_control",
       "Oblique isotropic stable counterpart at the same k, B, and angle as the CGL mirror-like case.",
       9, 0.03, 1.2, 1.0e16, 0.40, 0.20, 0.20, 1.0e-4, "stable"},
  };

  Json results = Json::array();
  bool passed = acceptance.at("passed").get<bool>();
  for (const DispersionCase& configuration : cases) {
    Json result = compare_case(configuration);
    passed = passed && result.at("passed").get<bool>();
    results.push_back(std::move(result));
  }
  Json output = {
      {"case", "A0_local_periodic_linear_dispersion"},
      {"method",
       "Independent 14x14 continuum matrix compared with central finite differences of the real production evaluate_stage response."},
      {"matrix_normalization",
       "A_normalized=diag(rate_scale)^-1*A_physical*diag(state_scale). Its eigenvalues are dimensionless; physical growth and angular frequency multiply by reference_frequency."},
      {"eigenvalue_scope",
       "The original 14x14 matrices and unrestricted spectra are retained. Physical eigenvalues compare the 11-dimensional orthonormal kernel of divB and transverse Ampere constraints. All three invariant derivatives are separately gated on the original matrices; no mode is dropped by speed."},
      {"polarization_convention",
       "exp(lambda*t+i*k.x); normalized right eigenvectors have unit Euclidean norm in the listed state scales. SI polarizations are componentwise products with those scales, per unit normalized amplitude. Frequencies are signed Im(lambda)/(2*pi)."},
      {"adapter",
       "Odd real phase chain with a spectral skew derivative; DR=sin(theta)*k0*D, DZ=cos(theta)*k0*D; equal entity counts and unit weights; I=H=QB=identity; no walls."},
      {"local_cartesian_limit",
       "All sample radii are 1e30 m, so production cylindrical 1/R terms are below double-precision significance for this local Cartesian adapter."},
      {"scope",
       "Local periodic model-level theoretical confirmation of fluid/CGL/Darwin physical response. This is not experimental validation, a production Yee accuracy study, or an alternative boundary implementation."},
      {"kinetic_threshold_scope",
       "CGL pressure labels do not assert kinetic firehose or mirror instability. The A0 equations contain no kinetic thresholds or Landau closure."},
      {"pass_criteria",
       {{"componentwise_normalized_matrix_max", matrix_tolerance},
        {"linearization_sensitivity_max", sensitivity_tolerance},
        {"eigenvalue_abs_tolerance_normalized", eigen_abs_tolerance},
        {"eigenvalue_relative_tolerance_normalized",
         eigen_relative_tolerance},
        {"magnetic_divergence_max_normalized",
         magnetic_divergence_tolerance},
        {"transverse_ampere_max_normalized",
         transverse_ampere_tolerance},
        {"original_matrix_invariant_max_normalized", magnetic_divergence_tolerance},
        {"original_matrix_eigen_residual_max_normalized", magnetic_divergence_tolerance},
        {"slow_branch_zeroing_control", "must_fail"},
        {"sign_reversal_control_minimum_normalized", matrix_tolerance}}},
      {"adapter_algebra_checks", std::move(adapter_results)},
      {"acceptance_gate_regressions", acceptance},
      {"cases", std::move(results)},
      {"passed", passed},
  };
  return output.dump(2);
}

} // namespace

std::string run_local_dispersion_benchmark(bool preflight) {
  return run_local_dispersion(preflight);
}

} // namespace fomhd::test

int main(int argc, char** argv) {
  try {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--preflight")) {
      throw std::invalid_argument("usage: fomhd_local_dispersion [--preflight]");
    }
    const std::string result =
        fomhd::test::run_local_dispersion_benchmark(argc == 2);
    std::cout << result << '\n';
    return nlohmann::json::parse(result).at("passed").get<bool>() ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "Local dispersion benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
