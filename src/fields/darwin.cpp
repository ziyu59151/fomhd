#include "fomhd/backend.hpp"
#include "fomhd/physics.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace fomhd {
namespace {

std::size_t modes(const Context &c) {
  return static_cast<std::size_t>(2 * c.p.K + 1);
}

void require_size(const Vec &value, std::size_t expected,
                  const std::string &name) {
  if (value.size() != expected) {
    throw Failure("Shape", name + " has an invalid extent");
  }
}

void require_finite(const Vec &value, const std::string &name) {
  if (!is_finite(value)) {
    throw Failure("Finite", name + " contains non-finite values");
  }
}

RealVec repeated_weights(const RealVec &weights, std::size_t count) {
  RealVec result;
  result.reserve(weights.size() * count);
  for (std::size_t repeat = 0; repeat < count; ++repeat) {
    result.insert(result.end(), weights.begin(), weights.end());
  }
  return result;
}

LinearResult checked_linear(const Matrix &matrix, const Vec &rhs,
                            const RealVec &row_scale,
                            const RealVec &column_scale, const Context &c,
                            const std::string &name) {
  try {
    return c.compute.linear(matrix, rhs, row_scale, column_scale, c.p.rank_rtol,
                            64 * matrix.rows *
                                std::numeric_limits<double>::epsilon());
  } catch (const NumericalError &error) {
    throw Failure("LinearSolve", name + ": " + error.what());
  }
}

double hermitian_defect(const Vec &values, std::size_t entities,
                        const Context &c) {
  return Fourier(c.p.K, c.N).hermitian_defect(values, entities);
}

Matrix matrix_sum(const Matrix &left, const Matrix &right,
                  const std::string &name) {
  if (left.rows != right.rows || left.cols != right.cols) {
    throw Failure("Shape", name + " operands have different shapes");
  }
  Matrix result(left.rows, left.cols);
  for (std::size_t i = 0; i < result.data.size(); ++i) {
    result.data[i] = left.data[i] + right.data[i];
  }
  return result;
}

Matrix matrix_difference(const Matrix &left, const Matrix &right,
                         const std::string &name) {
  if (left.rows != right.rows || left.cols != right.cols) {
    throw Failure("Shape", name + " operands have different shapes");
  }
  Matrix result(left.rows, left.cols);
  for (std::size_t i = 0; i < result.data.size(); ++i) {
    result.data[i] = left.data[i] - right.data[i];
  }
  return result;
}

} // namespace

FieldCache electrostatic(const Vec &charge_hat, const StageTag &tag,
                         const Context &c) {
  require_size(charge_hat, modes(c) * c.op.Nc, "charge density");
  require_finite(charge_hat, "charge density");
  const double charge_scale = std::max(1.0, max_abs(charge_hat));
  if (hermitian_defect(charge_hat, c.op.Nc, c) > 2.0e-11 * charge_scale) {
    throw Failure("Hermitian", "charge density is not physical");
  }

  FieldCache result;
  result.tag = tag;
  result.phi.assign(modes(c) * c.op.Nc, Complex{});
  result.EL.assign(modes(c) * c.op.Ne, Complex{});
  result.poisson_residual.resize(modes(c) * c.op.Nc);
  const RealVec rows(c.op.Nc, c.p.ref.charge / c.p.epsilon0);
  const RealVec columns(c.op.Nc, c.p.ref.electric * c.p.ref.length);
  for (int mode = -c.p.K; mode <= c.p.K; ++mode) {
    const std::size_t k = static_cast<std::size_t>(mode + c.p.K);
    const Vec rhs =
        scaled(slice(charge_hat, k * c.op.Nc, c.op.Nc), 1.0 / c.p.epsilon0);
    const LinearResult linear =
        checked_linear(c.op.Lphi[k], rhs, rows, columns, c, "Poisson solve");
    const Vec poisson_residual =
        subtract(c.compute.mv(c.op.Lphi[k], linear.x), rhs);
    put(result.poisson_residual, k * c.op.Nc, poisson_residual);
    result.poisson_solves.push_back(linear);
    put(result.phi, k * c.op.Nc, linear.x);
    Vec electric = scaled(c.compute.mv(c.op.G[k], linear.x), -1.0);
    // The stage checks all Gauss and original Poisson rows in physical units.
    put(result.EL, k * c.op.Ne, std::move(electric));
  }
  const double potential_scale = std::max(1.0, max_abs(result.phi));
  if (hermitian_defect(result.phi, c.op.Nc, c) > 2.0e-11 * potential_scale ||
      hermitian_defect(result.EL, c.op.Ne, c) >
          2.0e-11 * std::max(1.0, max_abs(result.EL))) {
    throw Failure("Hermitian", "Poisson physical field is not Hermitian");
  }
  return result;
}

Vec longitudinal(const Vec &current_hat, const Context &c) {
  require_size(current_hat, modes(c) * c.op.Ne, "current");
  require_finite(current_hat, "current");
  Vec result(modes(c) * c.op.Ne);
  const RealVec rows(c.op.Nc, c.p.ref.current / c.p.ref.length);
  const RealVec columns(c.op.Nc, c.p.ref.current * c.p.ref.length);
  for (int mode = -c.p.K; mode <= c.p.K; ++mode) {
    const std::size_t k = static_cast<std::size_t>(mode + c.p.K);
    const Vec source = slice(current_hat, k * c.op.Ne, c.op.Ne);
    const Vec rhs = c.compute.mv(c.op.DJ[k], source);
    const LinearResult linear = checked_linear(
        c.op.Lphi[k], rhs, rows, columns, c, "longitudinal projection solve");
    const Vec projected = scaled(c.compute.mv(c.op.G[k], linear.x), -1.0);

    put(result, k * c.op.Ne, projected);
  }
  return result;
}

void transverse(StageResult &stage, const Context &c) {
  const std::size_t edge_size = modes(c) * c.op.Ne;
  require_size(stage.fJ0, edge_size, "no-electric current");
  require_size(stage.cache.EL, edge_size, "electrostatic field");
  require_finite(stage.fJ0, "no-electric current");
  require_finite(stage.cache.EL, "electrostatic field");
  for (int species = 0; species < 2; ++species) {
    const FluidContext &fluid = stage.fluid[static_cast<std::size_t>(species)];
    if (fluid.n.empty() || fluid.wall.samples.empty()) {
      throw Failure("Stage", "fluid context is not prepared");
    }
  }

  const auto action = [&](const Vec &electric_hat) {
    require_size(electric_hat, edge_size, "transverse trial field");
    Vec result(edge_size, Complex{});
    for (int species = 0; species < 2; ++species) {
      const std::size_t s = static_cast<std::size_t>(species);
      const Species &physical = c.p.species[s];
      const Vec response =
          electric_response(stage.fluid[s], species, electric_hat, c);
      const Vec edges = sample_apply(
          c.op.I, scaled(response, physical.charge / physical.mass), c.N,
          c.compute);
      result = add(result, forward(edges, c.op.Ne, c));
    }
    return result;
  };

  const std::size_t ae_bytes = [&] {
    try {
      return checked_product(checked_product(edge_size, edge_size, "AE"),
                             sizeof(Complex), "AE bytes");
    } catch (const NumericalError &error) {
      throw Failure("Memory", error.what());
    }
  }();
  try {
    require_memory(ae_bytes, MemoryBudget{"AE", c.p.memory_limit});
  } catch (const NumericalError &error) {
    throw Failure("Memory", error.what());
  }
  stage.AE = materialize(edge_size, edge_size, action,
                         MemoryBudget{"AE", c.p.memory_limit});

  std::vector<Matrix> curl_blocks;
  curl_blocks.reserve(modes(c));
  const Matrix m1 = diagonal(c.op.M1);
  for (std::size_t k = 0; k < modes(c); ++k) {
    Matrix block = c.compute.mm(m1, c.compute.mm(c.op.Cdag[k], c.op.C[k]));
    for (Complex &value : block.data) {
      value /= c.p.mu0;
    }
    curl_blocks.push_back(std::move(block));
  }
  const Matrix curl = block_diagonal(curl_blocks);
  const RealVec repeated_m1 = repeated_weights(c.op.M1, modes(c));
  const Matrix weighted_response =
      c.compute.mm(diagonal(repeated_m1), stage.AE);
  const double weighted_scale = std::max(1.0, max_abs(weighted_response));
  if (max_abs(matrix_difference(weighted_response, adjoint(weighted_response),
                                "weighted response adjoint")) >
      64 * edge_size * std::numeric_limits<double>::epsilon() *
          weighted_scale) {
    throw Failure("Adjoint", "M1*AE is not weighted self-adjoint");
  }
  const Matrix dynamic =
      matrix_sum(curl, weighted_response, "Darwin dynamic matrix");

  const std::size_t constraints = c.op.Bc.rows;
  const std::size_t system_size = edge_size + constraints;
  const std::size_t matrix_bytes = [&] {
    try {
      const std::size_t square =
          checked_product(system_size, system_size, "KKT");
      const std::size_t total = checked_sum(
          checked_sum(checked_product(edge_size, edge_size, "AE storage"),
                      checked_product(edge_size, edge_size, "dynamic storage"),
                      "Darwin storage"),
          checked_sum(checked_product(edge_size, edge_size,
                                      "weighted response storage"),
                      checked_product(square, 2, "KKT workspace"),
                      "Darwin storage"),
          "Darwin storage");
      return checked_product(total, sizeof(Complex), "Darwin solve bytes");
    } catch (const NumericalError &error) {
      throw Failure("Memory", error.what());
    }
  }();
  try {
    require_memory(matrix_bytes, MemoryBudget{"Darwin KKT", c.p.memory_limit});
  } catch (const NumericalError &error) {
    throw Failure("Memory", error.what());
  }

  const Vec trial_rhs = add(stage.fJ0, action(stage.cache.EL));
  const Vec top_rhs =
      scaled(c.compute.mv(diagonal(repeated_m1), trial_rhs), -1.0);
  Vec rhs(system_size, Complex{});
  for (std::size_t i = 0; i < edge_size; ++i) {
    rhs[i] = top_rhs[i];
  }
  Matrix kkt(system_size, system_size);
  for (std::size_t column = 0; column < edge_size; ++column) {
    for (std::size_t row = 0; row < edge_size; ++row) {
      kkt(row, column) = dynamic(row, column);
    }
  }
  const Matrix bc_adjoint = adjoint(c.op.Bc);
  for (std::size_t column = 0; column < constraints; ++column) {
    for (std::size_t row = 0; row < edge_size; ++row) {
      kkt(row, edge_size + column) = bc_adjoint(row, column);
    }
  }
  for (std::size_t row = 0; row < constraints; ++row) {
    for (std::size_t column = 0; column < edge_size; ++column) {
      kkt(edge_size + row, column) = c.op.Bc(row, column);
    }
  }

  RealVec row_scale(system_size);
  RealVec column_scale(system_size);
  const double current_rate = c.p.ref.current / c.p.ref.time;
  const double lambda_scale = current_rate * std::pow(c.p.ref.length, 4);
  for (std::size_t i = 0; i < edge_size; ++i) {
    row_scale[i] = repeated_m1[i] * current_rate;
    column_scale[i] = c.p.ref.electric;
  }
  for (std::size_t i = 0; i < constraints; ++i) {
    row_scale[edge_size + i] = c.p.ref.electric / c.p.ref.length;
    column_scale[edge_size + i] = lambda_scale;
  }
  LinearResult linear;
  try {
    linear = checked_linear(kkt, rhs, row_scale, column_scale, c,
                            "Darwin KKT solve");
  } catch (const Failure &error) {
    Matrix scaled(system_size, system_size);
    for (std::size_t column = 0; column < system_size; ++column) {
      for (std::size_t row = 0; row < system_size; ++row) {
        scaled(row, column) =
            kkt(row, column) * column_scale[column] / row_scale[row];
      }
    }
    const RealVec singular = singular_values(scaled);
    throw Failure(
        "DarwinRank",
        std::string(error.what()) +
            "; AE=" + std::to_string(max_abs(stage.AE)) +
            " dynamic=" + std::to_string(max_abs(dynamic)) +
            " constraints=" + std::to_string(constraints) + " smax=" +
            std::to_string(singular.empty() ? 0.0 : singular.front()) +
            " smin=" +
            std::to_string(singular.empty() ? 0.0 : singular.back()));
  }
  const Vec transverse_field = slice(linear.x, 0, edge_size);
  const Vec multiplier = slice(linear.x, edge_size, constraints);

  const Vec dynamic_residual =
      subtract(add(c.compute.mv(dynamic, transverse_field),
                   c.compute.mv(bc_adjoint, multiplier)),
               top_rhs);
  const Vec full_divergence = c.compute.mv(c.op.Bc_full, transverse_field);
  stage.residuals["transverse_kkt"] = linear.residual;
  stage.residuals["transverse_dynamic"] = dynamic_residual;
  stage.residuals["transverse_divergence_full"] = full_divergence;
  Vec strong_residual = dynamic_residual;
  for (std::size_t i = 0; i < edge_size; ++i)
    strong_residual[i] /= repeated_m1[i];
  stage.residuals["darwin_dynamic"] = std::move(strong_residual);
  stage.cache.transverse_solve = linear;
  stage.cache.multiplier = multiplier;

  stage.cache.ET = transverse_field;
  stage.cache.E = add(stage.cache.EL, transverse_field);
  stage.cache.tag = stage.tag;
  const double field_scale = std::max(1.0, max_abs(stage.cache.E));
  if (hermitian_defect(stage.cache.ET, c.op.Ne, c) >
          512 * std::numeric_limits<double>::epsilon() *
              std::max(c.p.ref.electric, max_abs(stage.cache.ET)) ||
      hermitian_defect(stage.cache.E, c.op.Ne, c) >
          512 * std::numeric_limits<double>::epsilon() * field_scale) {
    throw Failure("Hermitian", "Darwin electric field is not physical");
  }

  stage.rhs.B = scaled(mode_apply(c.op.C, transverse_field, c.compute), -1.0);
  stage.residuals["transverse_ct"] =
      mode_apply(c.op.DB, stage.rhs.B, c.compute);
}

} // namespace fomhd
