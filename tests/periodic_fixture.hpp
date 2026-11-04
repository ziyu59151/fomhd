#pragma once

#include "fomhd/backend.hpp"
#include "fomhd/operators.hpp"
#include "fomhd/physics.hpp"
#include "support.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fomhd::test {

constexpr std::size_t dispersion_dimension = 14;
constexpr std::array<const char*, dispersion_dimension> dispersion_names{
    "n_i",       "mom_i_R",     "mom_i_phi", "mom_i_Z", "p_i",
    "n_e",       "mom_e_R",     "mom_e_phi", "mom_e_Z", "p_perp_e",
    "p_parallel_e", "B_R",      "B_phi",     "B_Z"};

using DispersionMatrix =
    std::array<std::array<Complex, dispersion_dimension>, dispersion_dimension>;

struct DispersionCase {
  std::string name;
  std::string scope;
  int points{9};
  double k0{1.0};
  double theta{0.0};
  double n0{1.0e16};
  double ion_pressure{0.4};
  double electron_perp{0.4};
  double electron_parallel{0.4};
  double b0{0.1};
  std::string expected_cgl_stability{"unspecified"};
};

class PeriodicFixture {
public:
  explicit PeriodicFixture(const DispersionCase& configuration,
                           bool reverse_curl = false);

  State uniform_state() const;
  StageResult evaluate(const State& state, bool alias = false);
  DispersionMatrix production_matrix(double relative_amplitude);
  Vec spectral_mode(const Vec& samples, std::size_t entities) const;

  const DispersionCase& configuration() const noexcept { return config_; }
  const Parameters& parameters() const noexcept { return parameters_; }
  const Geometry& geometry() const noexcept { return geometry_; }
  const Operators& operators() const noexcept { return operators_; }
  Context& context() noexcept { return context_; }
  const std::array<double, dispersion_dimension>& variable_scales() const {
    return variable_scales_;
  }
  const std::array<double, dispersion_dimension>& rate_scales() const {
    return rate_scales_;
  }
  double reference_frequency() const noexcept { return reference_frequency_; }
  std::uint64_t evaluations() const noexcept { return evaluations_; }
  std::uint64_t hard_checks_checked() const noexcept {
    return hard_checks_checked_;
  }
  std::uint64_t empty_wall_checks() const noexcept {
    return empty_wall_checks_;
  }

private:
  DispersionCase config_;
  Parameters parameters_;
  Geometry geometry_;
  Operators operators_;
  Compute compute_;
  Context context_;
  std::array<double, dispersion_dimension> variable_scales_{};
  std::array<double, dispersion_dimension> rate_scales_{};
  double reference_frequency_{};
  std::uint64_t next_trial_{1};
  std::uint64_t evaluations_{};
  std::uint64_t hard_checks_checked_{};
  std::uint64_t empty_wall_checks_{};

  void perturb(State& state, std::size_t column, double amplitude) const;
};

inline Parameters periodic_parameters(int points) {
  if (points != 9 && points != 11) {
    throw std::invalid_argument("periodic fixture requires 9 or 11 points");
  }
  Parameters p = parameters(0);
  p.K = 0;
  p.Nphi = 1;
  p.B_min = 1.0e-12;
  p.inner_distance = 0.0;
  p.rank_rtol = 1.0e-12;
  p.memory_limit = 512ULL * 1024ULL * 1024ULL;
  p.device_memory_limit = p.memory_limit;
  return p;
}

inline void align_periodic_budget_references(Parameters& p) {
  double response = 0.0;
  for (const Species& species : p.species) {
    response += species.charge * species.charge * p.ref.ni / species.mass;
  }
  p.ref.current =
      p.ref.time * p.ref.electric *
      (response + 1.0 / (p.mu0 * p.ref.length * p.ref.length));
  const double length = p.ref.length;
  const double time = p.ref.time;
  const double density = p.ref.ni;
  const double electric = p.ref.electric;
  const double magnetic = p.ref.magnetic;
  const double pressure = p.ref.pressure;
  const double force =
      p.species[0].mass * density * p.ref.velocity / time;
  const double power =
      pressure * length * length * length / time;
  const double energy =
      pressure * length * length * length +
      magnetic * magnetic * length * length * length / p.mu0;
  const auto array = [&](const std::string& key, double reference) {
    for (const std::string suffix : {".linf", ".weighted_L2"}) {
      auto iterator = p.budgets.find(key + suffix);
      if (iterator != p.budgets.end()) {
        iterator->second.reference = reference;
      }
    }
  };
  const auto scalar = [&](const std::string& key, double reference) {
    auto iterator = p.budgets.find(key + ".value");
    if (iterator != p.budgets.end()) {
      iterator->second.reference = reference;
    }
  };
  for (const char* name :
       {"gauss", "poisson_linear", "transverse_divergence"}) {
    array(name, electric / length);
  }
  array("longitudinal_projection", p.ref.current / length);
  array("darwin_dynamic", p.ref.current / time);
  for (const char* name : {"current_divB", "current_ampere"}) {
    array(name, magnetic / length);
  }
  for (const char* name :
       {"initial_divB", "initial_ampere", "divB_update",
        "long_term_divB_change"}) {
    array(name, magnetic / length);
  }
  array("CT_increment", magnetic / (length * time));
  for (const char* name : {"CT_update", "equilibrium_B", "alias_B"}) {
    array(name, magnetic / time);
  }
  array("current_response", p.ref.current / time);
  for (const char* name : {"RC_flux", "RC_mom", "deltaC"}) {
    array(name, p.ref.charge / time);
  }
  array("RL", p.ref.current);
  for (const char* name :
       {"ion_internal", "electron_perp", "electron_parallel",
        "field_energy_B", "field_energy_L", "total_energy"}) {
    scalar(name, energy);
  }
  scalar("field_exchange", energy / time);
  scalar("topology_increment", magnetic * length * length);
  scalar("total_charge_balance", p.ref.charge * length * length * length);
  for (const char* name : {"momentum_x", "momentum_y", "momentum_z"}) {
    scalar(name, p.species[0].mass * density * length * length * length *
                     p.ref.velocity);
  }
  scalar("angular_momentum",
         p.species[0].mass * density * std::pow(length, 4) *
             p.ref.velocity);
  for (int species = 0; species < 2; ++species) {
    const std::string id = "." + p.species[species].id;
    for (const char* name :
         {"stage_continuity", "species_continuity", "alias_n",
          "equilibrium_n"}) {
      array(name + id, density / time);
    }
    for (const char* name :
         {"wall_rhs", "wall_multiplier", "alias_mom", "equilibrium_mom"}) {
      array(name + id, force);
    }
    array("wall_state" + id, force * time);
    array("theta_identity" + id, 1.0 / time);
    array("wall_number_flux" + id, density * p.ref.velocity);
    array("wall_perp_flux" + id, pressure * p.ref.velocity);
    if (species) {
      array("wall_parallel_flux" + id, pressure * p.ref.velocity);
    }
    for (const char* name : {"alias_perp", "equilibrium_perp"}) {
      array(name + id, pressure / time);
    }
    if (species) {
      for (const char* name : {"cgl_source", "alias_parallel",
                               "equilibrium_parallel"}) {
        array(name + id, pressure / time);
      }
    }
    for (const char* name :
         {"electric_pairing", "pressure_pairing", "magnetic_work",
          "wall_work", "convection_pairing"}) {
      scalar(name + id, power);
    }
    scalar("spectral_transfer_defect" + id, power);
    scalar("total_number_balance" + id, density * length * length * length);
    scalar("kinetic_energy" + id, energy);
  }
}

inline EntityTable periodic_entity_table(std::string name, int points) {
  EntityTable table;
  table.name = std::move(name);
  table.keys.resize(static_cast<std::size_t>(points));
  table.dense_to_active.resize(static_cast<std::size_t>(points));
  table.active_to_dense.resize(static_cast<std::size_t>(points));
  table.R.assign(static_cast<std::size_t>(points), 1.0e30);
  table.Z.resize(static_cast<std::size_t>(points));
  table.boundary.assign(static_cast<std::size_t>(points), 0);
  for (int j = 0; j < points; ++j) {
    const auto index = static_cast<std::size_t>(j);
    table.keys[index] = EntityKey{index, 0};
    table.dense_to_active[index] = index;
    table.active_to_dense[index] = index;
    table.Z[index] = 2.0 * std::numbers::pi * static_cast<double>(j) /
                     static_cast<double>(points);
    table.lookup.emplace(table.keys[index], index);
  }
  return table;
}

inline Geometry periodic_geometry(int points) {
  if (points != 9 && points != 11) {
    throw std::invalid_argument("periodic geometry requires 9 or 11 points");
  }
  const auto size = static_cast<std::size_t>(points);
  Geometry geometry;
  geometry.R = {1.0e30};
  geometry.Z = {0.0, 1.0};
  geometry.active.assign(1, 0);
  geometry.cells = periodic_entity_table("cells", points);
  geometry.r_faces = periodic_entity_table("r_faces", points);
  geometry.z_faces = periodic_entity_table("z_faces", points);
  geometry.vertices = periodic_entity_table("vertices", points);
  geometry.M0.assign(size, 1.0);
  geometry.M1.assign(3 * size, 1.0);
  geometry.MB.assign(3 * size, 1.0);
  geometry.lr.assign(size, 1.0);
  geometry.lz.assign(size, 1.0);
  geometry.av.assign(size, 1.0);
  geometry.wv.assign(size, 1.0);
  geometry.Rbar_r.assign(size, 1.0e30);
  geometry.Rbar_v.assign(size, 1.0e30);
  geometry.cell_wall_distance.assign(
      size, std::numeric_limits<double>::infinity());
  geometry.diagnostics.active_cells = size;
  geometry.diagnostics.wall_segments = 0;
  geometry.diagnostics.hausdorff_lower_m = 0.0;
  geometry.diagnostics.hausdorff_upper_m = 0.0;
  geometry.diagnostics.signed_polygon_area = 0.0;
  return geometry;
}

inline Matrix periodic_spectral_derivative(int points) {
  const auto n = static_cast<std::size_t>(points);
  Matrix derivative(n, n);
  const std::size_t half = (n - 1) / 2;
  std::vector<long double> coefficients(half + 1);
  for (std::size_t distance = 1; distance <= half; ++distance) {
    long double sum = 0.0L;
    for (std::size_t mode = 1; mode <= half; ++mode) {
      const long double argument =
          2.0L * std::numbers::pi_v<long double> * mode * distance /
          static_cast<long double>(n);
      sum += static_cast<long double>(mode) * std::sin(argument);
    }
    coefficients[distance] =
        4.0L * std::numbers::pi_v<long double> * sum /
        (static_cast<long double>(n) * static_cast<long double>(n));
  }
  for (std::size_t row = 0; row < n; ++row) {
    derivative(row, row) = Complex{};
    for (std::size_t distance = 1; distance <= half; ++distance) {
      const std::size_t forward = (row + distance) % n;
      const std::size_t backward = (row + n - distance) % n;
      const double value = static_cast<double>(coefficients[distance]);
      derivative(row, forward) = value;
      derivative(row, backward) = -value;
    }
  }
  return derivative;
}

inline Operators periodic_operators(const Geometry& geometry, int points,
                                    double k0, double theta,
                                    bool reverse_curl) {
  const auto n = static_cast<std::size_t>(points);
  const auto edges = 3 * n;
  const Matrix spectral = periodic_spectral_derivative(points);
  Matrix radial(n, n);
  Matrix axial(n, n);
  const double physical_derivative_scale =
      k0 * static_cast<double>(points) / (2.0 * std::numbers::pi);
  for (std::size_t i = 0; i < spectral.data.size(); ++i) {
    radial.data[i] =
        spectral.data[i] * (physical_derivative_scale * std::sin(theta));
    axial.data[i] =
        spectral.data[i] * (physical_derivative_scale * std::cos(theta));
  }

  Operators operators;
  operators.K = 0;
  operators.Nc = n;
  operators.Ne = edges;
  operators.Nb = edges;
  operators.M0 = geometry.M0;
  operators.M1 = geometry.M1;
  operators.MB = geometry.MB;
  operators.Rbar_r = geometry.Rbar_r;
  operators.Rbar_v = geometry.Rbar_v;

  Matrix gradient(edges, n);
  Matrix divergence(n, edges);
  Matrix curl(edges, edges);
  for (std::size_t column = 0; column < n; ++column) {
    for (std::size_t row = 0; row < n; ++row) {
      gradient(row, column) = radial(row, column);
      gradient(2 * n + row, column) = axial(row, column);
      curl(row, n + column) = -axial(row, column);
      curl(n + row, column) = axial(row, column);
      curl(n + row, 2 * n + column) = -radial(row, column);
      curl(2 * n + row, n + column) = radial(row, column);
      divergence(row, column) = radial(row, column);
      divergence(row, 2 * n + column) = axial(row, column);
    }
  }
  Matrix current_divergence = adjoint(gradient);
  for (Complex& value : current_divergence.data) {
    value = -value;
  }
  if (reverse_curl) {
    for (Complex& value : curl.data) {
      value = -value;
    }
  }
  Matrix poisson = matmul(adjoint(gradient), gradient);
  const double gauge = k0 * k0 / static_cast<double>(n);
  for (std::size_t row = 0; row < n; ++row) {
    for (std::size_t column = 0; column < n; ++column) {
      poisson(row, column) += gauge;
    }
  }

  operators.G = {gradient};
  operators.C = {curl};
  operators.DB = {divergence};
  operators.DJ = {current_divergence};
  operators.Cdag = {adjoint(curl)};
  operators.Lphi = {poisson};
  operators.phi_factor = {factor(poisson)};
  operators.DR = radial;
  operators.DZ = axial;
  operators.I = identity(edges);
  operators.H = identity(edges);
  operators.QB = identity(edges);
  operators.PR_r = identity(n);
  operators.PZ_r = identity(n);
  operators.PR_z = identity(n);
  operators.PZ_z = identity(n);
  operators.PR_v = identity(n);
  operators.PZ_v = identity(n);
  operators.PR_r_factor = factor(operators.PR_r);
  operators.PZ_r_factor = factor(operators.PZ_r);
  operators.PR_z_factor = factor(operators.PR_z);
  operators.PZ_z_factor = factor(operators.PZ_z);
  operators.PR_v_factor = factor(operators.PR_v);
  operators.PZ_v_factor = factor(operators.PZ_v);
  operators.T_full = Matrix(0, edges);
  operators.T = Matrix(0, edges);
  operators.Bc_full = current_divergence;
  operators.selected_constraint_rows =
      independent_rows(operators.Bc_full, 1.0e-12);
  operators.Bc = Matrix(operators.selected_constraint_rows.size(), edges);
  for (std::size_t row = 0; row < operators.selected_constraint_rows.size();
       ++row) {
    for (std::size_t column = 0; column < edges; ++column) {
      operators.Bc(row, column) =
          operators.Bc_full(operators.selected_constraint_rows[row], column);
    }
  }
  operators.topology_flux_row = Matrix(1, edges);
  for (std::size_t column = 2 * n; column < edges; ++column) {
    operators.topology_flux_row(0, column) = 1.0 / static_cast<double>(n);
  }
  return operators;
}

struct AdapterChecks {
  double plane_wave_derivative{};
  double derivative_imaginary{};
  double derivative_skew{};
  double derivative_constant_mode{};
  double direct_curl_action{};
  double cg_identity{};
  double divergence_curl_identity{};
  double curl_adjoint_identity{};
  double current_divergence_from_gradient{};
  double tolerance{};
  bool passed{};
};

inline AdapterChecks validate_periodic_adapter(const Operators& operators,
                                               int points, double k0,
                                               double theta) {
  const auto n = static_cast<std::size_t>(points);
  AdapterChecks checks;
  Vec mode(n);
  for (std::size_t cell = 0; cell < n; ++cell) {
    mode[cell] = std::exp(
        Complex{0.0, 2.0 * std::numbers::pi * static_cast<double>(cell) /
                          static_cast<double>(n)});
  }
  const Vec radial = matvec(operators.DR, mode);
  const Vec axial = matvec(operators.DZ, mode);
  for (std::size_t cell = 0; cell < n; ++cell) {
    checks.plane_wave_derivative = std::max(
        {checks.plane_wave_derivative,
         std::abs(radial[cell] - Complex{0.0, k0 * std::sin(theta)} * mode[cell]),
         std::abs(axial[cell] - Complex{0.0, k0 * std::cos(theta)} * mode[cell])});
  }
  for (const Complex value : operators.DR.data) {
    checks.derivative_imaginary =
        std::max(checks.derivative_imaginary, std::abs(value.imag()));
  }
  for (const Complex value : operators.DZ.data) {
    checks.derivative_imaginary =
        std::max(checks.derivative_imaginary, std::abs(value.imag()));
  }
  checks.derivative_skew = std::max(
      max_abs(add(operators.DR.data, adjoint(operators.DR).data)),
      max_abs(add(operators.DZ.data, adjoint(operators.DZ).data)));
  Vec constant_mode(n, Complex{1.0, 0.0});
  checks.derivative_constant_mode =
      std::max(max_abs(matvec(operators.DR, constant_mode)),
               max_abs(matvec(operators.DZ, constant_mode)));

  const Matrix& gradient = operators.G.front();
  const Matrix& curl = operators.C.front();
  const Matrix& divergence = operators.DB.front();
  const Matrix& current_divergence = operators.DJ.front();
  const Matrix& curl_adjoint = operators.Cdag.front();
  checks.cg_identity = max_abs(matmul(curl, gradient));
  checks.divergence_curl_identity = max_abs(matmul(divergence, curl));
  checks.curl_adjoint_identity =
      max_abs(subtract(curl_adjoint.data, adjoint(curl).data));
  checks.current_divergence_from_gradient =
      max_abs(subtract(current_divergence.data,
                       scaled(adjoint(gradient).data, -1.0)));

  double scale = 1.0;
  for (std::size_t component = 0; component < 3; ++component) {
    Vec curl_input(3 * n);
    for (std::size_t cell = 0; cell < n; ++cell) {
      curl_input[component * n + cell] = mode[cell];
    }
    const Vec actual = matvec(curl, curl_input);
    const Vec radial_part = matvec(operators.DR, mode);
    const Vec axial_part = matvec(operators.DZ, mode);
    Vec expected(3 * n);
    if (component == 0) {
      for (std::size_t cell = 0; cell < n; ++cell)
        expected[n + cell] = axial_part[cell];
    } else if (component == 1) {
      for (std::size_t cell = 0; cell < n; ++cell) {
        expected[cell] = -axial_part[cell];
        expected[2 * n + cell] = radial_part[cell];
      }
    } else {
      for (std::size_t cell = 0; cell < n; ++cell)
        expected[n + cell] = -radial_part[cell];
    }
    checks.direct_curl_action = std::max(
        checks.direct_curl_action, max_abs(subtract(actual, expected)));
    for (std::size_t row = 0; row < 3 * n; ++row) {
      for (std::size_t column = 0; column < 3 * n; ++column) {
        scale = std::max(scale, std::abs(curl(row, column)));
      }
    }
  }
  scale = std::max({scale, max_abs(gradient), max_abs(divergence),
                    max_abs(current_divergence), max_abs(curl_adjoint)});
  checks.tolerance =
      4096.0 * std::numeric_limits<double>::epsilon() * scale;
  checks.passed =
      checks.plane_wave_derivative <= checks.tolerance &&
      checks.derivative_imaginary == 0.0 &&
      checks.derivative_skew == 0.0 &&
      checks.derivative_constant_mode <= checks.tolerance &&
      checks.direct_curl_action <= checks.tolerance &&
      checks.cg_identity <= checks.tolerance &&
      checks.divergence_curl_identity <= checks.tolerance &&
      checks.curl_adjoint_identity <= checks.tolerance &&
      checks.current_divergence_from_gradient <= checks.tolerance;
  return checks;
}

inline PeriodicFixture::PeriodicFixture(const DispersionCase& configuration,
                                        bool reverse_curl)
    : config_(configuration),
      parameters_(periodic_parameters(configuration.points)),
      geometry_(periodic_geometry(configuration.points)),
      operators_(periodic_operators(geometry_, configuration.points,
                                    configuration.k0, configuration.theta,
                                    reverse_curl)),
      compute_(),
      context_{parameters_, geometry_, operators_, compute_,
               parameters_.Nphi, 7, 1, 1, 1, "", "", ""} {
  const double ion_mass = parameters_.species[0].mass;
  const double electron_mass = parameters_.species[1].mass;
  const double ion_thermal =
      std::sqrt(configuration.ion_pressure /
                (configuration.n0 * ion_mass));
  const double alfven =
      configuration.b0 / std::sqrt(parameters_.mu0 * ion_mass *
                                   configuration.n0);
  const double sound =
      std::sqrt((parameters_.species[0].gamma *
                     configuration.ion_pressure +
                 2.0 * configuration.electron_parallel) /
                (ion_mass * configuration.n0));
  const double hall =
      configuration.b0 * std::abs(parameters_.species[0].charge) /
      std::sqrt(ion_mass * electron_mass);
  reference_frequency_ =
      std::hypot(configuration.k0 * std::hypot(alfven, sound), hall);
  if (!(reference_frequency_ > 0.0) || !std::isfinite(reference_frequency_)) {
    throw std::invalid_argument("periodic fixture frequency scale is invalid");
  }
  parameters_.ref.length = 1.0;
  parameters_.ref.time = 1.0 / reference_frequency_;
  parameters_.ref.ni = configuration.n0;
  parameters_.ref.ne = configuration.n0;
  parameters_.ref.velocity = ion_thermal;
  parameters_.ref.magnetic = configuration.b0;
  parameters_.ref.pressure = configuration.ion_pressure;
  parameters_.ref.charge =
      std::max(std::abs(parameters_.species[0].charge),
               std::abs(parameters_.species[1].charge)) *
      configuration.n0;
  parameters_.ref.electric =
      parameters_.ref.charge /
      (parameters_.epsilon0 * std::max(configuration.k0, 1.0e-6));
  align_periodic_budget_references(parameters_);

  variable_scales_ = {
      configuration.n0,
      ion_mass * configuration.n0 * ion_thermal,
      ion_mass * configuration.n0 * ion_thermal,
      ion_mass * configuration.n0 * ion_thermal,
      configuration.ion_pressure,
      configuration.n0,
      electron_mass * configuration.n0 * ion_thermal,
      electron_mass * configuration.n0 * ion_thermal,
      electron_mass * configuration.n0 * ion_thermal,
      configuration.electron_perp,
      configuration.electron_parallel,
      configuration.b0,
      configuration.b0,
      configuration.b0,
  };
  rate_scales_ = {
      configuration.n0 * reference_frequency_,
      ion_mass * configuration.n0 * ion_thermal * reference_frequency_,
      ion_mass * configuration.n0 * ion_thermal * reference_frequency_,
      ion_mass * configuration.n0 * ion_thermal * reference_frequency_,
      configuration.ion_pressure * reference_frequency_,
      configuration.n0 * reference_frequency_,
      electron_mass * configuration.n0 * ion_thermal *
          reference_frequency_,
      electron_mass * configuration.n0 * ion_thermal *
          reference_frequency_,
      electron_mass * configuration.n0 * ion_thermal *
          reference_frequency_,
      configuration.electron_perp * reference_frequency_,
      configuration.electron_parallel * reference_frequency_,
      configuration.b0 * reference_frequency_,
      configuration.b0 * reference_frequency_,
      configuration.b0 * reference_frequency_,
  };
}

inline State PeriodicFixture::uniform_state() const {
  const auto n = operators_.Nc;
  State state;
  for (int species = 0; species < 2; ++species) {
    state.fluid[static_cast<std::size_t>(species)].n.assign(
        n, species == 0 ? config_.n0 : config_.n0);
    state.fluid[static_cast<std::size_t>(species)].mom.assign(3 * n,
                                                               Complex{});
    state.fluid[static_cast<std::size_t>(species)].perp.assign(
        n, species == 0 ? config_.ion_pressure : config_.electron_perp);
    if (species == 1) {
      state.fluid[1].parallel.assign(n, config_.electron_parallel);
    }
  }
  state.B.assign(3 * n, Complex{});
  for (std::size_t cell = 0; cell < n; ++cell) {
    state.B[2 * n + cell] = config_.b0;
  }
  state.step = 0;
  state.generation = 1;
  state.time = parameters_.start_time;
  return state;
}

inline void PeriodicFixture::perturb(State& state, std::size_t column,
                                     double amplitude) const {
  if (column >= dispersion_dimension) {
    throw std::out_of_range("dispersion perturbation column");
  }
  const auto n = operators_.Nc;
  Vec* destination = nullptr;
  std::size_t offset = 0;
  std::size_t entities = n;
  if (column == 0) {
    destination = &state.fluid[0].n;
  } else if (column >= 1 && column <= 3) {
    destination = &state.fluid[0].mom;
    offset = (column - 1) * n;
    entities = n;
  } else if (column == 4) {
    destination = &state.fluid[0].perp;
  } else if (column == 5) {
    destination = &state.fluid[1].n;
  } else if (column >= 6 && column <= 8) {
    destination = &state.fluid[1].mom;
    offset = (column - 6) * n;
    entities = n;
  } else if (column == 9) {
    destination = &state.fluid[1].perp;
  } else if (column == 10) {
    destination = &state.fluid[1].parallel;
  } else {
    destination = &state.B;
    offset = column == 11 ? 0 : column == 12 ? n : 2 * n;
  }
  const double physical_amplitude =
      amplitude * variable_scales_[column];
  for (std::size_t cell = 0; cell < entities; ++cell) {
    const double coordinate =
        2.0 * std::numbers::pi * static_cast<double>(cell) /
        static_cast<double>(n);
    (*destination)[offset + cell] +=
        physical_amplitude * std::cos(coordinate);
  }
}

inline StageResult PeriodicFixture::evaluate(const State& state, bool alias) {
  ++evaluations_;
  const Vec direct_momentum = compute_.inverse_fft(
      state.fluid[0].mom, 3 * operators_.Nc, parameters_.K, parameters_.Nphi);
  for (const Complex value : direct_momentum) {
    if (value.imag() != 0.0) {
      throw Failure("FixtureInverse",
                    "direct K=0 inverse introduced an imaginary sample");
    }
  }
  const StageTag tag{state.step, state.generation, next_trial_++, 7,
                     state.time, "dispersion"};
  StageResult result = evaluate_stage(state, tag, context_, alias);
  for (const Check& check : result.checks) {
    if (check.name.starts_with("wall_") && !check.available &&
        !check.hard && !check.passed &&
        check.reason == "not applicable: no boundary entities") {
      ++empty_wall_checks_;
    }
    if (check.hard) {
      ++hard_checks_checked_;
      if (!check.available || !check.passed) {
        throw Failure("DispersionHardGate",
                      "hard production stage diagnostic failed: " +
                          check.name);
      }
    }
  }
  return result;
}

inline Vec PeriodicFixture::spectral_mode(const Vec& samples,
                                          std::size_t entities) const {
  const auto n = operators_.Nc;
  if (samples.size() != entities * n) {
    throw std::invalid_argument("periodic spectral-mode shape");
  }
  Vec result(entities, Complex{});
  for (std::size_t component = 0; component < entities; ++component) {
    for (std::size_t cell = 0; cell < n; ++cell) {
      const double coordinate =
          2.0 * std::numbers::pi * static_cast<double>(cell) /
          static_cast<double>(n);
      result[component] +=
          std::exp(Complex{0.0, -coordinate}) *
          samples[component * n + cell];
    }
    result[component] /= static_cast<double>(n);
  }
  return result;
}

inline Vec rhs_columns(const StageResult& stage, std::size_t n) {
  Vec result;
  result.reserve(dispersion_dimension * n);
  const auto append = [&](const Vec& values, std::size_t entities) {
    result.insert(result.end(), values.begin(),
                  values.begin() + static_cast<std::ptrdiff_t>(entities * n));
  };
  append(stage.rhs.fluid[0].n, 1);
  append(stage.rhs.fluid[0].mom, 3);
  append(stage.rhs.fluid[0].perp, 1);
  append(stage.rhs.fluid[1].n, 1);
  append(stage.rhs.fluid[1].mom, 3);
  append(stage.rhs.fluid[1].perp, 1);
  append(stage.rhs.fluid[1].parallel, 1);
  append(stage.rhs.B, 3);
  return result;
}

inline DispersionMatrix PeriodicFixture::production_matrix(
    double relative_amplitude) {
  if (!(relative_amplitude > 0.0) || !std::isfinite(relative_amplitude)) {
    throw std::invalid_argument("dispersion perturbation amplitude is invalid");
  }
  const auto n = operators_.Nc;
  DispersionMatrix matrix{};
  for (std::size_t column = 0; column < dispersion_dimension; ++column) {
    State plus = uniform_state();
    State minus = uniform_state();
    perturb(plus, column, relative_amplitude);
    perturb(minus, column, -relative_amplitude);
    StageResult plus_stage;
    StageResult minus_stage;
    try {
      plus_stage = evaluate(plus);
    } catch (const Failure& error) {
      throw Failure(error.code,
                    std::string(error.what()) + "; positive perturbation in " +
                        dispersion_names[column]);
    }
    try {
      minus_stage = evaluate(minus);
    } catch (const Failure& error) {
      throw Failure(error.code,
                    std::string(error.what()) + "; negative perturbation in " +
                        dispersion_names[column]);
    }
    const Vec plus_rhs = rhs_columns(plus_stage, n);
    const Vec minus_rhs = rhs_columns(minus_stage, n);
    for (std::size_t row = 0; row < dispersion_dimension; ++row) {
      const Vec plus_mode =
          spectral_mode(Vec(plus_rhs.begin() +
                                static_cast<std::ptrdiff_t>(row * n),
                            plus_rhs.begin() +
                                static_cast<std::ptrdiff_t>((row + 1) * n)),
                        1);
      const Vec minus_mode =
          spectral_mode(Vec(minus_rhs.begin() +
                                static_cast<std::ptrdiff_t>(row * n),
                            minus_rhs.begin() +
                                static_cast<std::ptrdiff_t>((row + 1) * n)),
                        1);
      const Complex cosine_amplitude{0.5, 0.0};
      const Complex derivative =
          (plus_mode.front() - minus_mode.front()) /
          (2.0 * relative_amplitude * cosine_amplitude);
      matrix[row][column] = derivative / rate_scales_[row];
    }
  }
  return matrix;
}

} // namespace fomhd::test
