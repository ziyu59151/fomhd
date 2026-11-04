#include "fomhd/backend.hpp"
#include "fomhd/physics.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <numbers>
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

void require_species(int species) {
  if (species < 0 || species > 1) {
    throw Failure("Species", "species index must be 0 or 1");
  }
}

Vec apply_mode_matrix(const Matrix &matrix, const Vec &input,
                      std::size_t entities, const Context &c) {
  require_size(input, modes(c) * entities, "mode input");
  Vec result(modes(c) * matrix.rows);
  for (std::size_t mode = 0; mode < modes(c); ++mode) {
    put(result, mode * matrix.rows,
        c.compute.mv(matrix, slice(input, mode * entities, entities)));
  }
  return result;
}

void validate_magnetic_samples(const Vec &BN, const Vec &bhat,
                               const Context &c) {
  const std::size_t vector_samples =
      static_cast<std::size_t>(c.N) * 3 * c.op.Nc;
  require_size(BN, vector_samples, "magnetic vector");
  require_size(bhat, vector_samples, "magnetic direction");
  require_finite(BN, "magnetic magnitude");
  require_finite(bhat, "magnetic direction");
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      const std::size_t vector = s * 3 * c.op.Nc;
      const double magnitude = std::hypot(
          std::abs(BN[vector + cell]), std::abs(BN[vector + c.op.Nc + cell]),
          std::abs(BN[vector + 2 * c.op.Nc + cell]));
      if (!(magnitude > c.p.B_min)) {
        throw Failure("MagneticDomain", "magnetic magnitude is below B_min");
      }
      double direction_norm = 0.0;
      for (std::size_t component = 0; component < 3; ++component) {
        direction_norm += std::norm(bhat[vector + component * c.op.Nc + cell]);
      }
      if (std::abs(direction_norm - 1.0) > 2.0e-10) {
        throw Failure("MagneticDirection",
                      "magnetic direction is not normalized");
      }
    }
  }
}

Vec component_flux(const Vec &scalar_samples, const Context &c) {
  require_size(scalar_samples, static_cast<std::size_t>(c.N) * c.op.Nc,
               "scalar interpolation");
  Vec nodes(scalar_samples.size() * 3);
  for (int l = 0; l < c.N; ++l)
    for (std::size_t d = 0; d < 3; ++d)
      put(nodes, (l * 3 + d) * c.op.Nc,
          slice(scalar_samples, l * c.op.Nc, c.op.Nc));
  Vec result = sample_apply(c.op.I, nodes, c.N, c.compute);
  require_finite(result, "component interpolation");
  return result;
}

Vec sampled_divergence(const Vec &edge_values, const Context &c) {
  const std::size_t radial_faces = c.op.DR.cols;
  const std::size_t axial_faces = c.op.DZ.cols;
  if (radial_faces + c.op.Nc + axial_faces != c.op.Ne) {
    throw Failure("Layout", "edge component offsets are inconsistent");
  }
  Vec radial_values(static_cast<std::size_t>(c.N) * radial_faces);
  Vec azimuthal_values(static_cast<std::size_t>(c.N) * c.op.Nc);
  Vec axial_values(static_cast<std::size_t>(c.N) * axial_faces);
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    put(radial_values, s * radial_faces,
        slice(edge_values, s * c.op.Ne, radial_faces));
    put(azimuthal_values, s * c.op.Nc,
        slice(edge_values, s * c.op.Ne + radial_faces, c.op.Nc));
    put(axial_values, s * axial_faces,
        slice(edge_values, s * c.op.Ne + radial_faces + c.op.Nc, axial_faces));
  }
  const Vec radial_part = sample_apply(c.op.DR, radial_values, c.N, c.compute);
  const Vec axial_part = sample_apply(c.op.DZ, axial_values, c.N, c.compute);
  const Vec azimuthal_part =
      c.compute.derivative(azimuthal_values, c.op.Nc, c.N);
  Vec result(static_cast<std::size_t>(c.N) * c.op.Nc);
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      const std::size_t index = s * c.op.Nc + cell;
      result[index] = radial_part[index] + axial_part[index] +
                      azimuthal_part[index] / c.geometry.cells.R[cell];
    }
  }
  return result;
}

Vec pressure_tensor(const FluidContext &fluid, int species, const Vec &bhat,
                    const Context &c) {
  Vec result(static_cast<std::size_t>(c.N) * 9 * c.op.Nc);
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      const std::size_t scalar = s * c.op.Nc + cell;
      const std::size_t vector = s * 3 * c.op.Nc;
      const std::size_t tensor = s * 9 * c.op.Nc;
      for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
          Complex value{};
          if (species == 0) {
            value = a == b ? fluid.perp[scalar] : Complex{};
          } else {
            value = (a == b ? fluid.perp[scalar] : Complex{}) +
                    (fluid.parallel[scalar] - fluid.perp[scalar]) *
                        bhat[vector + a * c.op.Nc + cell] *
                        bhat[vector + b * c.op.Nc + cell];
          }
          result[tensor + (a * 3 + b) * c.op.Nc + cell] = value;
        }
      }
    }
  }
  return result;
}

Vec transport_flux(const Vec &pressure, const FluidContext &fluid,
                   const Context &c) {
  const Vec pressure_edges = component_flux(pressure, c);
  Vec result(fluid.flux.size());
  for (std::size_t i = 0; i < result.size(); ++i) {
    if (std::abs(fluid.nf[i]) <= std::numeric_limits<double>::min()) {
      throw Failure("FluidDomain", "interpolated density is zero");
    }
    result[i] = pressure_edges[i] / fluid.nf[i] * fluid.flux[i];
  }
  return result;
}

} // namespace

Vec divergence(const Vec &flux, const Context &c) {
  require_size(flux, static_cast<std::size_t>(c.N) * c.op.Ne, "sampled flux");
  require_finite(flux, "sampled flux");
  return sampled_divergence(flux, c);
}

FluidContext build_fluid(const FluidState &state, int species, const Vec &BN,
                         const Vec &bhat, const StageTag &tag,
                         const Context &c) {
  require_species(species);
  const Species &physical = c.p.species[static_cast<std::size_t>(species)];
  require_size(state.n, modes(c) * c.op.Nc, "density");
  require_size(state.mom, modes(c) * 3 * c.op.Nc, "momentum");
  require_size(state.perp, modes(c) * c.op.Nc, "perpendicular pressure");
  if (species == 1) {
    require_size(state.parallel, modes(c) * c.op.Nc, "parallel pressure");
  }
  require_finite(state.n, "density");
  require_finite(state.mom, "momentum");
  require_finite(state.perp, "perpendicular pressure");
  if (species == 1) {
    require_finite(state.parallel, "parallel pressure");
  }
  validate_magnetic_samples(BN, bhat, c);

  FluidContext fluid;
  fluid.n = inverse(state.n, c.op.Nc, c, true);
  fluid.mom = inverse(state.mom, 3 * c.op.Nc, c, true);
  fluid.perp = inverse(state.perp, c.op.Nc, c, true);
  fluid.parallel =
      species == 1 ? inverse(state.parallel, c.op.Nc, c, true) : Vec{};

  for (std::size_t j = 0; j < fluid.n.size(); ++j) {
    const double density = fluid.n[j].real();
    if (!std::isfinite(density) || !(density > physical.n_min)) {
      throw Failure("Domain", "density is not positive before reciprocal");
    }
    if (!(fluid.perp[j].real() > physical.p_min) ||
        (species == 1 && !(fluid.parallel[j].real() > physical.parallel_min))) {
      throw Failure("Domain", "pressure is below its declared floor");
    }
  }

  fluid.u.resize(fluid.mom.size());
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t component = 0; component < 3; ++component) {
      for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
        const std::size_t vector = (s * 3 + component) * c.op.Nc + cell;
        const std::size_t scalar = s * c.op.Nc + cell;
        fluid.u[vector] =
            fluid.mom[vector] / (physical.mass * fluid.n[scalar].real());
      }
    }
  }

  const Vec momentum_per_mass = scaled(state.mom, 1.0 / physical.mass);
  fluid.flux_hat = apply_mode_matrix(c.op.I, momentum_per_mass, 3 * c.op.Nc, c);
  fluid.flux = inverse(fluid.flux_hat, c.op.Ne, c, true);
  for (std::size_t j = 0; j < fluid.flux.size(); ++j) {
    if (!std::isfinite(std::abs(fluid.flux[j]))) {
      throw Failure("Finite", "sample flux is non-finite");
    }
  }

  fluid.nf = component_flux(fluid.n, c);
  for (std::size_t j = 0; j < fluid.nf.size(); ++j) {
    if (!std::isfinite(std::abs(fluid.nf[j])) ||
        !(fluid.nf[j].real() > physical.face_min)) {
      throw Failure("Domain", "interpolated density is not invertible");
    }
  }

  fluid.divergence_hat = mode_apply(c.op.DJ, fluid.flux_hat, c.compute);
  fluid.divergence = sampled_divergence(fluid.flux, c);
  const Vec sampled_retained = forward(fluid.divergence, c.op.Nc, c);
  const double division_scale =
      std::max({1.0, max_abs(fluid.divergence_hat), max_abs(sampled_retained)});
  if (max_abs(subtract(fluid.divergence_hat, sampled_retained)) >
      2048.0 * std::numeric_limits<double>::epsilon() * division_scale) {
    throw Failure("FluidDivision",
                  "retained and sampled density divergences disagree");
  }

  fluid.tensor = pressure_tensor(fluid, species, bhat, c);
  const Vec &pressure = fluid.tensor;
  fluid.raw_moment.assign(static_cast<std::size_t>(c.N) * 9 * c.op.Nc,
                          Complex{});
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      const std::size_t scalar = s * c.op.Nc + cell;
      const std::size_t vector = s * 3 * c.op.Nc;
      const std::size_t tensor = s * 9 * c.op.Nc;
      for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
          const Complex inertial = physical.mass * fluid.n[scalar] *
                                   fluid.u[vector + a * c.op.Nc + cell] *
                                   fluid.u[vector + b * c.op.Nc + cell];
          const std::size_t index = tensor + (a * 3 + b) * c.op.Nc + cell;
          fluid.raw_moment[index] = inertial + pressure[index];
        }
      }
    }
  }
  fluid.wall = build_wall(fluid.n, physical, tag, c);
  return fluid;
}

Vec velocity_gradient(const FluidContext &fluid, const Vec &x,
                      const Context &c) {
  require_size(x, static_cast<std::size_t>(c.N) * 3 * c.op.Nc,
               "velocity-gradient input");
  require_finite(x, "velocity-gradient input");
  require_size(fluid.n, static_cast<std::size_t>(c.N) * c.op.Nc,
               "velocity-gradient density");
  require_size(fluid.nf, static_cast<std::size_t>(c.N) * c.op.Ne,
               "velocity-gradient denominator");
  const std::size_t radial_faces = c.op.DR.cols;
  const std::size_t axial_offset = radial_faces + c.op.Nc;
  Vec result(static_cast<std::size_t>(c.N) * 9 * c.op.Nc);
  std::vector<Vec> azimuthal_derivatives;
  azimuthal_derivatives.reserve(3);
  for (std::size_t component = 0; component < 3; ++component) {
    Vec values(static_cast<std::size_t>(c.N) * c.op.Nc);
    for (int sample = 0; sample < c.N; ++sample) {
      const std::size_t s = static_cast<std::size_t>(sample);
      put(values, s * c.op.Nc,
          slice(x, (s * 3 + component) * c.op.Nc, c.op.Nc));
    }
    azimuthal_derivatives.push_back(c.compute.derivative(values, c.op.Nc, c.N));
  }

  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t component = 0; component < 3; ++component) {
      Vec weighted(static_cast<std::size_t>(c.N) * 3 * c.op.Nc);
      for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
        // Apply each scalar directional interpolator to this component.
        for (std::size_t direction = 0; direction < 3; ++direction)
          weighted[(s * 3 + direction) * c.op.Nc + cell] =
              fluid.n[s * c.op.Nc + cell] *
              x[(s * 3 + component) * c.op.Nc + cell];
      }
      Vec interpolated = sample_apply(c.op.I, weighted, c.N, c.compute);
      Vec radial(radial_faces);
      Vec axial(c.op.DZ.cols);
      for (std::size_t face = 0; face < radial_faces; ++face) {
        const double denominator = fluid.nf[s * c.op.Ne + face].real();
        if (denominator == 0.0) {
          throw Failure("FluidDomain",
                        "radial interpolation denominator is zero");
        }
        radial[face] = interpolated[s * c.op.Ne + face] / denominator;
      }
      for (std::size_t face = 0; face < axial.size(); ++face) {
        const double denominator =
            fluid.nf[s * c.op.Ne + axial_offset + face].real();
        if (denominator == 0.0) {
          throw Failure("FluidDomain",
                        "axial interpolation denominator is zero");
        }
        axial[face] =
            interpolated[s * c.op.Ne + axial_offset + face] / denominator;
      }
      const Vec radial_derivative = c.compute.mv(c.op.DR, radial);
      const Vec axial_derivative = c.compute.mv(c.op.DZ, axial);
      const Vec original =
          slice(x, s * 3 * c.op.Nc + component * c.op.Nc, c.op.Nc);
      const Vec azimuthal_derivative =
          slice(azimuthal_derivatives[component], s * c.op.Nc, c.op.Nc);
      for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
        const std::size_t tensor = s * 9 * c.op.Nc;
        const double radius = c.geometry.cells.R[cell];
        result[tensor + (component * 3 + 0) * c.op.Nc + cell] =
            radial_derivative[cell] - original[cell] / radius;
        result[tensor + (component * 3 + 1) * c.op.Nc + cell] =
            azimuthal_derivative[cell] / radius;
        result[tensor + (component * 3 + 2) * c.op.Nc + cell] =
            axial_derivative[cell];
      }
    }

    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      const std::size_t tensor = s * 9 * c.op.Nc;
      const double radius = c.geometry.cells.R[cell];
      const Complex radial = x[(s * 3 + 0) * c.op.Nc + cell];
      const Complex azimuthal = x[(s * 3 + 1) * c.op.Nc + cell];
      result[tensor + (0 * 3 + 1) * c.op.Nc + cell] -= azimuthal / radius;
      result[tensor + (1 * 3 + 1) * c.op.Nc + cell] += radial / radius;
    }
  }
  return result;
}

Vec gradient_adjoint(const FluidContext &fluid, const Vec &tensor,
                     const Context &c) {
  require_size(tensor, static_cast<std::size_t>(c.N) * 9 * c.op.Nc,
               "gradient-adjoint tensor");
  require_finite(tensor, "gradient-adjoint tensor");
  require_size(fluid.n, static_cast<std::size_t>(c.N) * c.op.Nc,
               "gradient-adjoint density");
  require_size(fluid.nf, static_cast<std::size_t>(c.N) * c.op.Ne,
               "gradient-adjoint denominator");
  const Matrix dr_adjoint = adjoint(c.op.DR);
  const Matrix dz_adjoint = adjoint(c.op.DZ);
  const Matrix i_adjoint = adjoint(c.op.I);
  const std::size_t radial_faces = c.op.DR.cols;
  const std::size_t axial_offset = radial_faces + c.op.Nc;
  Vec result(static_cast<std::size_t>(c.N) * 3 * c.op.Nc);

  for (std::size_t row = 0; row < 3; ++row) {
    Vec radial_axial(static_cast<std::size_t>(c.N) * c.op.Nc);
    Vec azimuthal_rhs(static_cast<std::size_t>(c.N) * c.op.Nc);
    for (int sample = 0; sample < c.N; ++sample) {
      const std::size_t s = static_cast<std::size_t>(sample);
      const auto weighted = [&](std::size_t column) {
        Vec values(c.op.Nc);
        for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
          values[cell] =
              c.op.M0[cell] *
              tensor[s * 9 * c.op.Nc + (row * 3 + column) * c.op.Nc + cell];
        }
        return values;
      };
      const Vec radial_weighted = weighted(0);
      const Vec azimuthal_weighted = weighted(1);
      const Vec axial_weighted = weighted(2);

      Vec radial_rhs = c.compute.mv(dr_adjoint, radial_weighted);
      Vec radial_edges(c.op.Ne);
      for (std::size_t face = 0; face < radial_faces; ++face) {
        const double denominator = fluid.nf[s * c.op.Ne + face].real();
        if (denominator == 0.0) {
          throw Failure("FluidDomain", "radial adjoint denominator is zero");
        }
        radial_edges[face] = radial_rhs[face] / denominator;
      }
      const Vec radial_nodes = c.compute.mv(i_adjoint, radial_edges);

      for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
        azimuthal_rhs[s * c.op.Nc + cell] =
            azimuthal_weighted[cell] / c.geometry.cells.R[cell];
      }

      Vec axial_rhs = c.compute.mv(dz_adjoint, axial_weighted);
      Vec axial_edges(c.op.Ne);
      for (std::size_t face = 0; face < c.op.DZ.cols; ++face) {
        const double denominator =
            fluid.nf[s * c.op.Ne + axial_offset + face].real();
        if (denominator == 0.0) {
          throw Failure("FluidDomain", "axial adjoint denominator is zero");
        }
        axial_edges[axial_offset + face] = axial_rhs[face] / denominator;
      }
      const Vec axial_nodes = c.compute.mv(i_adjoint, axial_edges);

      for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
        const double radius = c.geometry.cells.R[cell];
        const Complex radial_value =
            fluid.n[s * c.op.Nc + cell] * radial_nodes[cell] -
            radial_weighted[cell] / radius;
        const Complex axial_value =
            fluid.n[s * c.op.Nc + cell] * axial_nodes[2 * c.op.Nc + cell];
        radial_axial[s * c.op.Nc + cell] = radial_value + axial_value;
      }
      if (row == 0) {
        for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
          radial_axial[s * c.op.Nc + cell] +=
              c.op.M0[cell] *
              tensor[s * 9 * c.op.Nc + (1 * 3 + 1) * c.op.Nc + cell] /
              c.geometry.cells.R[cell];
        }
      } else if (row == 1) {
        for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
          radial_axial[s * c.op.Nc + cell] -=
              c.op.M0[cell] *
              tensor[s * 9 * c.op.Nc + (0 * 3 + 1) * c.op.Nc + cell] /
              c.geometry.cells.R[cell];
        }
      }
    }
    const Vec azimuthal_derivative =
        c.compute.derivative(azimuthal_rhs, c.op.Nc, c.N);
    for (int sample = 0; sample < c.N; ++sample) {
      const std::size_t s = static_cast<std::size_t>(sample);
      for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
        const std::size_t scalar = s * c.op.Nc + cell;
        result[s * 3 * c.op.Nc + row * c.op.Nc + cell] =
            (radial_axial[scalar] - azimuthal_derivative[scalar]) /
            c.op.M0[cell];
      }
    }
  }
  return result;
}

void build_non_electric(FluidContext &fluid, int species, const Vec &BN,
                        const Vec &bhat, const Context &c) {
  require_species(species);
  validate_magnetic_samples(BN, bhat, c);
  const Species &physical = c.p.species[static_cast<std::size_t>(species)];
  fluid.gradient = velocity_gradient(fluid, fluid.u, c);
  fluid.theta.assign(static_cast<std::size_t>(c.N) * c.op.Nc, Complex{});
  fluid.strain.assign(static_cast<std::size_t>(c.N) * c.op.Nc, Complex{});
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      const std::size_t scalar = s * c.op.Nc + cell;
      const std::size_t vector = s * 3 * c.op.Nc;
      const std::size_t tensor = s * 9 * c.op.Nc;
      Complex trace{};
      Complex parallel_strain{};
      for (std::size_t a = 0; a < 3; ++a) {
        trace += fluid.gradient[tensor + (a * 3 + a) * c.op.Nc + cell];
        for (std::size_t b = 0; b < 3; ++b) {
          parallel_strain +=
              bhat[vector + a * c.op.Nc + cell] *
              fluid.gradient[tensor + (a * 3 + b) * c.op.Nc + cell] *
              bhat[vector + b * c.op.Nc + cell];
        }
      }
      fluid.theta[scalar] = trace;
      fluid.strain[scalar] = parallel_strain;
    }
  }

  Vec advective_tensor(static_cast<std::size_t>(c.N) * 9 * c.op.Nc, Complex{});
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
          advective_tensor[s * 9 * c.op.Nc + (a * 3 + b) * c.op.Nc + cell] =
              fluid.u[s * 3 * c.op.Nc + a * c.op.Nc + cell] *
              fluid.mom[s * 3 * c.op.Nc + b * c.op.Nc + cell] / physical.mass;
        }
      }
    }
  }
  const Vec advection_adjoint = gradient_adjoint(fluid, advective_tensor, c);
  fluid.convection.assign(static_cast<std::size_t>(c.N) * 3 * c.op.Nc,
                          Complex{});
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      const std::size_t scalar = s * c.op.Nc + cell;
      for (std::size_t a = 0; a < 3; ++a) {
        Complex gradient_flux{};
        for (std::size_t b = 0; b < 3; ++b) {
          gradient_flux +=
              fluid.gradient[s * 9 * c.op.Nc + (a * 3 + b) * c.op.Nc + cell] *
              fluid.mom[s * 3 * c.op.Nc + b * c.op.Nc + cell] / physical.mass;
        }
        const std::size_t index = s * 3 * c.op.Nc + a * c.op.Nc + cell;
        fluid.convection[index] = 0.5 * physical.mass *
                                  (-advection_adjoint[index] + gradient_flux +
                                   fluid.u[index] * fluid.divergence[scalar]);
      }
    }
  }

  const Vec pressure = pressure_tensor(fluid, species, bhat, c);
  fluid.pressure_force = gradient_adjoint(fluid, pressure, c);
  fluid.magnetic_force.assign(static_cast<std::size_t>(c.N) * 3 * c.op.Nc,
                              Complex{});
  fluid.non_electric.assign(fluid.magnetic_force.size(), Complex{});
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      const std::size_t scalar = s * c.op.Nc + cell;
      const std::size_t vector = s * 3 * c.op.Nc;
      const Complex b_r = BN[vector + cell];
      const Complex b_phi = BN[vector + c.op.Nc + cell];
      const Complex b_z = BN[vector + 2 * c.op.Nc + cell];
      const Complex u_r = fluid.u[vector + cell];
      const Complex u_phi = fluid.u[vector + c.op.Nc + cell];
      const Complex u_z = fluid.u[vector + 2 * c.op.Nc + cell];
      fluid.magnetic_force[vector + cell] =
          physical.charge * fluid.n[scalar] * (u_phi * b_z - u_z * b_phi);
      fluid.magnetic_force[vector + c.op.Nc + cell] =
          physical.charge * fluid.n[scalar] * (u_z * b_r - u_r * b_z);
      fluid.magnetic_force[vector + 2 * c.op.Nc + cell] =
          physical.charge * fluid.n[scalar] * (u_r * b_phi - u_phi * b_r);
    }
    for (std::size_t component = 0; component < 3; ++component) {
      for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
        const std::size_t index = (s * 3 + component) * c.op.Nc + cell;
        fluid.non_electric[index] = -fluid.convection[index] +
                                    fluid.pressure_force[index] +
                                    fluid.magnetic_force[index];
      }
    }
  }

  fluid.pressure_flux = transport_flux(fluid.perp, fluid, c);
  const Vec pressure_transport = sampled_divergence(fluid.pressure_flux, c);
  fluid.pdot_perp.assign(static_cast<std::size_t>(c.N) * c.op.Nc, Complex{});
  if (species == 0) {
    for (std::size_t i = 0; i < fluid.pdot_perp.size(); ++i) {
      fluid.pdot_perp[i] = -pressure_transport[i] - (physical.gamma - 1.0) *
                                                        fluid.perp[i] *
                                                        fluid.theta[i];
    }
  } else {
    fluid.parallel_flux = transport_flux(fluid.parallel, fluid, c);
    const Vec parallel_transport = sampled_divergence(fluid.parallel_flux, c);
    fluid.pdot_parallel.assign(fluid.pdot_perp.size(), Complex{});
    for (std::size_t i = 0; i < fluid.pdot_perp.size(); ++i) {
      fluid.pdot_perp[i] = -pressure_transport[i] +
                           fluid.perp[i] * (fluid.strain[i] - fluid.theta[i]);
      fluid.pdot_parallel[i] =
          -parallel_transport[i] - 2.0 * fluid.parallel[i] * fluid.strain[i];
    }
  }
}

Vec electric_response(const FluidContext &fluid, int species,
                      const Vec &electric_hat, const Context &c) {
  require_species(species);
  require_size(electric_hat, modes(c) * c.op.Ne, "electric response field");
  require_finite(electric_hat, "electric response field");
  const Species &physical = c.p.species[static_cast<std::size_t>(species)];
  const Vec electric = inverse(electric_hat, c.op.Ne, c, false);
  const Vec interpolated = sample_apply(c.op.H, electric, c.N, c.compute);
  Vec force(interpolated.size());
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      for (std::size_t component = 0; component < 3; ++component) {
        const std::size_t index = (s * 3 + component) * c.op.Nc + cell;
        force[index] =
            physical.charge * fluid.n[s * c.op.Nc + cell] * interpolated[index];
      }
    }
  }
  return project_wall(fluid.wall, force, c).value;
}

void finish_fluid(StageResult &stage, int species, const Context &c) {
  require_species(species);
  FluidContext &fluid = stage.fluid[static_cast<std::size_t>(species)];
  const Species &physical = c.p.species[static_cast<std::size_t>(species)];
  require_size(stage.cache.E, modes(c) * c.op.Ne, "field cache E");
  const Vec electric =
      inverse(stage.cache.E, c.op.Ne, c, true, c.p.ref.electric);
  const Vec interpolated = sample_apply(c.op.H, electric, c.N, c.compute);
  Vec electric_force(interpolated.size());
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      for (std::size_t component = 0; component < 3; ++component) {
        const std::size_t index = (s * 3 + component) * c.op.Nc + cell;
        electric_force[index] =
            physical.charge * fluid.n[s * c.op.Nc + cell] * interpolated[index];
      }
    }
  }
  const Vec total_raw = add(fluid.non_electric, electric_force);
  const Projection projected = project_wall(fluid.wall, total_raw, c);

  const auto ensure = [](Vec &value, std::size_t size,
                         const std::string &name) {
    if (!value.empty() && value.size() != size) {
      throw Failure("Shape", name + " has an invalid extent");
    }
    value.assign(size, Complex{});
  };
  ensure(stage.rhs.fluid[static_cast<std::size_t>(species)].n,
         modes(c) * c.op.Nc, "density RHS");
  ensure(stage.rhs.fluid[static_cast<std::size_t>(species)].mom,
         modes(c) * 3 * c.op.Nc, "momentum RHS");
  ensure(stage.rhs.fluid[static_cast<std::size_t>(species)].perp,
         modes(c) * c.op.Nc, "perpendicular pressure RHS");
  if (species == 1) {
    ensure(stage.rhs.fluid[static_cast<std::size_t>(species)].parallel,
           modes(c) * c.op.Nc, "parallel pressure RHS");
  }
  stage.rhs.fluid[static_cast<std::size_t>(species)].n = fluid.divergence_hat;
  for (Complex &value : stage.rhs.fluid[static_cast<std::size_t>(species)].n) {
    value = -value;
  }
  stage.rhs.fluid[static_cast<std::size_t>(species)].perp =
      forward(fluid.pdot_perp, c.op.Nc, c);
  stage.rhs.fluid[static_cast<std::size_t>(species)].mom =
      forward(projected.value, 3 * c.op.Nc, c);
  if (species == 1) {
    stage.rhs.fluid[static_cast<std::size_t>(species)].parallel =
        forward(fluid.pdot_parallel, c.op.Nc, c);
  }

  FluidTerms &terms = stage.terms[static_cast<std::size_t>(species)];
  terms.electric = electric_force;
  terms.force = projected.value;
  terms.reaction = projected.reaction;
  terms.multiplier = projected.multiplier;
  terms.multiplier_residual = projected.multiplier_residual;
  const Vec wall_rows = sample_apply(c.op.T_full, terms.force, c.N, c.compute);
  stage.residuals["wall_fluid_" + std::to_string(species)] = wall_rows;
}

namespace {

Complex user_integral(const Vec &values, std::size_t components,
                      const Context &c) {
  if (values.size() != static_cast<std::size_t>(c.N) * components * c.op.Nc) {
    throw Failure("Shape", "cell integral has an invalid extent");
  }
  Complex result{};
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t component = 0; component < components; ++component) {
      for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
        result += c.op.M0[cell] *
                  values[(s * components + component) * c.op.Nc + cell];
      }
    }
  }
  return result * (2.0 * std::numbers::pi / static_cast<double>(c.N));
}

Complex edge_integral(const Vec &values, const Context &c) {
  if (values.size() != static_cast<std::size_t>(c.N) * c.op.Ne) {
    throw Failure("Shape", "edge integral has an invalid extent");
  }
  Complex result{};
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t s = static_cast<std::size_t>(sample);
    for (std::size_t edge = 0; edge < c.op.Ne; ++edge) {
      result += c.op.M1[edge] * values[s * c.op.Ne + edge];
    }
  }
  return result * (2.0 * std::numbers::pi / static_cast<double>(c.N));
}

double real_part(Complex value, const std::string &name) {
  const double scale = std::max(1.0, std::abs(value));
  if (std::abs(value.imag()) >
      4096.0 * std::numeric_limits<double>::epsilon() * scale) {
    throw Failure("Hermitian", name + " has an imaginary physical value");
  }
  return value.real();
}

} // namespace

Ledger power_ledger(const StageResult &stage, int species, const Context &c) {
  require_species(species);
  const std::size_t s = static_cast<std::size_t>(species);
  const FluidContext &fluid = stage.fluid[s];
  const FluidTerms &terms = stage.terms[s];
  const Species &physical = c.p.species[s];
  const Vec electric =
      inverse(stage.cache.E, c.op.Ne, c, true, c.p.ref.electric);

  Vec electric_node(fluid.u.size());
  Vec pressure_kinetic(fluid.u.size());
  Vec magnetic_work(fluid.u.size());
  Vec wall_work(fluid.u.size());
  Vec convection(fluid.u.size());
  Vec kinetic_retained(fluid.u.size());
  Vec kinetic_sample(fluid.u.size());
  for (std::size_t i = 0; i < fluid.u.size(); ++i) {
    electric_node[i] = std::conj(fluid.u[i]) * terms.electric[i];
    pressure_kinetic[i] = std::conj(fluid.u[i]) * fluid.pressure_force[i];
    magnetic_work[i] = std::conj(fluid.u[i]) * fluid.magnetic_force[i];
    wall_work[i] = std::conj(fluid.u[i]) * terms.reaction[i];
    convection[i] = std::conj(fluid.u[i]) * fluid.convection[i];
  }
  const Vec ndot_back = inverse(stage.rhs.fluid[s].n, c.op.Nc, c, true);
  const Vec pidot_back = inverse(stage.rhs.fluid[s].mom, 3 * c.op.Nc, c, true);
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t sample_index = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      double speed_squared = 0.0;
      for (std::size_t component = 0; component < 3; ++component) {
        const std::size_t index =
            (sample_index * 3 + component) * c.op.Nc + cell;
        speed_squared += std::norm(fluid.u[index]);
        kinetic_retained[index] = std::conj(fluid.u[index]) * pidot_back[index];
        kinetic_sample[index] = std::conj(fluid.u[index]) * terms.force[index];
      }
      const std::size_t scalar = sample_index * c.op.Nc + cell;
      const Complex density_term = 0.5 * physical.mass * speed_squared;
      kinetic_retained[sample_index * 3 * c.op.Nc + cell] -=
          density_term * ndot_back[scalar];
      kinetic_sample[sample_index * 3 * c.op.Nc + cell] -=
          density_term * (-fluid.divergence[scalar]);
      convection[sample_index * 3 * c.op.Nc + cell] -=
          density_term * fluid.divergence[scalar];
    }
  }

  Vec pressure_contraction(static_cast<std::size_t>(c.N) * c.op.Nc);
  const Vec pressure = pressure_tensor(fluid, species, stage.bhat, c);
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t sample_index = static_cast<std::size_t>(sample);
    for (std::size_t cell = 0; cell < c.op.Nc; ++cell) {
      Complex value{};
      for (std::size_t a = 0; a < 3; ++a) {
        for (std::size_t b = 0; b < 3; ++b) {
          value += pressure[sample_index * 9 * c.op.Nc + (a * 3 + b) * c.op.Nc +
                            cell] *
                   fluid.gradient[sample_index * 9 * c.op.Nc +
                                  (a * 3 + b) * c.op.Nc + cell];
        }
      }
      pressure_contraction[sample_index * c.op.Nc + cell] = value;
    }
  }

  Vec electric_edge(static_cast<std::size_t>(c.N) * c.op.Ne);
  for (int sample = 0; sample < c.N; ++sample) {
    const std::size_t sample_index = static_cast<std::size_t>(sample);
    for (std::size_t edge = 0; edge < c.op.Ne; ++edge) {
      electric_edge[sample_index * c.op.Ne + edge] =
          physical.charge *
          std::conj(fluid.flux[sample_index * c.op.Ne + edge]) *
          electric[sample_index * c.op.Ne + edge];
    }
  }

  Ledger result;
  result.electric_node =
      real_part(user_integral(electric_node, 3, c), "electric node work");
  result.electric_edge =
      real_part(edge_integral(electric_edge, c), "electric edge work");
  result.pressure_kinetic =
      real_part(user_integral(pressure_kinetic, 3, c), "pressure kinetic work");
  result.pressure_internal = real_part(
      -user_integral(pressure_contraction, 1, c), "pressure internal work");
  result.magnetic =
      real_part(user_integral(magnetic_work, 3, c), "magnetic work");
  result.wall = real_part(user_integral(wall_work, 3, c), "wall work");
  result.convection =
      real_part(user_integral(convection, 3, c), "convection pairing");
  Vec internal_perp(fluid.theta.size()), internal_parallel(fluid.theta.size());
  for(std::size_t i=0;i<internal_perp.size();++i) {
    internal_perp[i]=species==0?-fluid.perp[i]*fluid.theta[i]:fluid.perp[i]*(fluid.strain[i]-fluid.theta[i]);
    if(species==1)internal_parallel[i]=-fluid.parallel[i]*fluid.strain[i];
  }
  result.perp_rate=real_part(user_integral(internal_perp,1,c),"perpendicular internal energy source");
  if(species==1)result.parallel_rate=real_part(user_integral(internal_parallel,1,c),"parallel internal energy source");
  result.spectral_kinetic =
      real_part(user_integral(subtract(kinetic_retained, kinetic_sample), 3, c),
                "kinetic spectral transfer");

  const double perp_factor=species==0?1/(physical.gamma-1):1;
  const Vec perp_back=inverse(stage.rhs.fluid[s].perp,c.op.Nc,c,true,c.p.ref.pressure/c.p.ref.time);
  result.spectral_perp=real_part(user_integral(scaled(subtract(perp_back,fluid.pdot_perp),perp_factor),1,c),"perpendicular internal spectral transfer");
  if(species==1) {
    const Vec parallel_back=inverse(stage.rhs.fluid[s].parallel,c.op.Nc,c,true,c.p.ref.pressure/c.p.ref.time);
    result.spectral_parallel=real_part(user_integral(scaled(subtract(parallel_back,fluid.pdot_parallel),.5),1,c),"parallel internal spectral transfer");
  }
  return result;
}

} // namespace fomhd
