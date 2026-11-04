#pragma once

#include "fomhd/operators.hpp"
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>

namespace fomhd {

struct Species {
  std::string id;
  double mass{}, charge{}, gamma{};
  double n_min{}, face_min{}, p_min{}, parallel_min{};
};
struct References {
  double length{}, time{}, ni{}, ne{}, velocity{}, electric{}, magnetic{};
  double current{}, charge{}, pressure{};
};
enum class Norm { linf, weighted_l2, value };
struct Budget {
  double atol{}, rtol{}, reference{};
  std::string unit;
  Norm norm{Norm::linf};
  bool hard{true};
};
struct Parameters {
  std::array<Species, 2> species;
  double epsilon0{}, mu0{}, B_min{}, dt{}, start_time{}, rank_rtol{};
  std::uint64_t nsteps{};
  int K{}, Nphi{};
  std::size_t memory_limit{}, device_memory_limit{};
  double inner_distance{};
  References ref;
  std::map<std::string, Budget> budgets;
};
struct StageTag {
  std::uint64_t step{}, generation{}, trial{}, spectral_context{};
  double time{};
  std::string stage;
  auto operator<=>(const StageTag &) const = default;
};

// Scalars: [mode][cell]; vectors: [mode][component][cell], component R/phi/Z.
struct FluidState {
  Vec n, mom, perp, parallel;
};
struct State {
  std::array<FluidState, 2> fluid;
  Vec B; // [mode][BR(z), Bphi(v), BZ(r)] in tesla.
  std::uint64_t step{}, generation{};
  double time{};
};
struct Check {
  std::string name, population, region{"all"}, unit, value_unit, reason;
  std::string component, producer{"diagnostics"};
  StageTag tag;
  double raw{}, reference{}, threshold{};
  Norm norm{Norm::linf};
  bool hard{true}, available{true}, passed{};
  std::size_t worst{};
  std::size_t selected{};
};
struct Failure : NumericalError {
  std::string code;
  std::vector<Check> checks;
  std::optional<StageTag> stage;
  std::map<std::string,Vec> partial_arrays;
  explicit Failure(std::string code_, std::string message)
      : NumericalError(std::move(message)), code(std::move(code_)) {}
};
struct FieldCache {
  Vec phi, EL, ET, E, multiplier, poisson_residual;
  StageTag tag;
  std::vector<LinearResult> poisson_solves;
  LinearResult transverse_solve;
};
struct WallSample {
  Matrix S;
  LU lu;
  RealVec winv;
  double row_scale{}, col_scale{};
};
struct WallContext {
  std::vector<WallSample> samples;
  StageTag tag;
};
struct Projection {
  Vec value, reaction, multiplier, multiplier_residual;
};
struct FluidContext {
  Vec n, mom, u, perp, parallel, nf, flux, flux_hat, divergence, divergence_hat;
  Vec tensor, raw_moment, gradient, theta, strain, convection, pressure_force;
  Vec magnetic_force, non_electric, pdot_perp, pdot_parallel;
  Vec pressure_flux, parallel_flux;
  WallContext wall;
};
struct FluidTerms {
  Vec electric, force, reaction, multiplier, multiplier_residual;
};
struct Ledger {
  double electric_node{}, electric_edge{}, pressure_kinetic{},
      pressure_internal{};
  double magnetic{}, wall{}, convection{}, perp_rate{}, parallel_rate{};
  double spectral_kinetic{}, spectral_perp{}, spectral_parallel{};
};
struct WallTrace {
  std::array<double, 3> B{}, EL{}, ET{};
  std::array<std::array<double, 3>, 2> velocity{};
  std::array<double, 2> density{}, normal_flux{}, perp{}, parallel{};
};
struct BoundaryLedger {
  std::vector<WallTrace> traces; // [sample][wall]
  std::array<double, 2> number{}, kinetic{}, internal_perp{},
      internal_parallel{};
  double field_energy{}, total_energy{};
};
struct Momentum {
  std::array<double, 3> value{}, outward{}, wall_reaction{};
  double angular{}, outward_torque{}, wall_torque{};
  int angular_samples{};
};
struct Totals {
  std::array<double, 2> number{}, kinetic{};
  double ion_internal{}, electron_perp{}, electron_parallel{}, magnetic{},
      longitudinal{}, energy{};
  double topology{}, physical_flux{};
};
struct StageResult {
  StageTag tag;
  State rhs;
  FieldCache cache;
  Vec BN, bhat, rho, current, fJ0, ELdot, phidot, ampere;
  Matrix AE;
  std::array<FluidContext, 2> fluid;
  std::array<FluidTerms, 2> terms;
  std::array<Ledger, 2> ledger;
  BoundaryLedger boundary;
  Momentum momentum;
  Totals totals;
  std::vector<Check> checks;
  std::map<std::string, Vec> residuals;
  std::uint64_t source_id{};
};
struct Baseline {
  Vec divB;
  Totals totals;
  Momentum momentum;
  double time{};
  std::uint64_t step{};
};
struct Cumulative {
  Baseline baseline;
  std::array<Ledger, 2> integrals;
  BoundaryLedger boundary_integrals;
  std::uint64_t accepted_intervals{}, last_trial{}, last_step{};
  double last_time{};
};
struct IntervalSources {
  std::array<Vec, 2> number_flux, current;
  std::uint64_t source_id{}, trial{}, left_step{}, right_step{};
  double weight{}, left_time{}, right_time{};
  bool accepted{};
};
struct Committed {
  State state;
  FieldCache cache;
  bool accepted{};
};
struct StepResult {
  std::vector<Check> checks;
  Totals totals;
  Momentum momentum;
  IntervalSources interval;
};

class Compute;
struct Context {
  const Parameters &p;
  const Geometry &geometry;
  const Operators &op;
  Compute &compute;
  int N;
  std::uint64_t spectral_id{}, next_generation{1}, next_trial{1},
      next_source{1};
  std::string geometry_hash, operator_hash, rule_hash;
};

Vec slice(const Vec &, std::size_t offset, std::size_t count);
void put(Vec &, std::size_t offset, const Vec &);
Vec repeat_weights(const RealVec &, std::size_t count);
State allocate_state(const Context &);
State combine(const State &, const State &, double, const StageTag &,
              const Context &);
void check_retained(const State &, const Context &);
double time_at_step(const Parameters &, std::uint64_t);
double time_at_midpoint(const Parameters &, std::uint64_t);

} // namespace fomhd
