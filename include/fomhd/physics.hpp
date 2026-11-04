#pragma once
#include "fomhd/types.hpp"

namespace fomhd {
std::size_t estimated_peak_bytes(const Context &);
Vec mode_apply(const std::vector<Matrix> &, const Vec &, Compute &);
Vec sample_apply(const Matrix &, const Vec &, int samples, Compute &);
Vec inverse(const Vec &, std::size_t entities, const Context &,
            bool real = true, double physical_reference = 0);
Vec forward(const Vec &, std::size_t entities, const Context &);
Vec divergence(const Vec &, const Context &);
WallContext build_wall(const Vec &n, const Species &, const StageTag &,
                       const Context &);
Projection project_wall(const WallContext &, const Vec &, const Context &);
FluidContext build_fluid(const FluidState &, int species, const Vec &BN,
                         const Vec &bhat, const StageTag &, const Context &);
Vec velocity_gradient(const FluidContext &, const Vec &, const Context &);
Vec gradient_adjoint(const FluidContext &, const Vec &, const Context &);
void build_non_electric(FluidContext &, int, const Vec &BN, const Vec &bhat,
                        const Context &);
Vec electric_response(const FluidContext &, int, const Vec &, const Context &);
FieldCache electrostatic(const Vec &, const StageTag &, const Context &);
Vec longitudinal(const Vec &, const Context &);
void transverse(StageResult &, const Context &);
void finish_fluid(StageResult &, int, const Context &);
Ledger power_ledger(const StageResult &, int, const Context &);
BoundaryLedger boundary_ledger(const State &, const StageResult &,
                               const Context &);
Totals state_totals(const State &, const StageResult &, const Context &);
Momentum momentum(const State &, const StageResult &, const Context &);
StageResult evaluate_stage(const State &, const StageTag &, Context &,
                           bool alias = false);
void retain_failure_evidence(Failure &, const StageTag &, const Context &,
                             const State *, const StageResult *,
                             const std::string &prefix);
Committed initialize(State, Context &, Cumulative &, bool equilibrium,
                     StageResult *captured = nullptr);
StepResult advance(Committed &, Cumulative &, Context &);
State closed_reference(const Context &, double density, double ion_pressure,
                       double electron_pressure, double B_reference,
                       double velocity = 0);
} // namespace fomhd
