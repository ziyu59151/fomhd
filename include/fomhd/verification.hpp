#pragma once
#include "fomhd/physics.hpp"
#include <functional>

namespace fomhd::verification {
// Test harness only. Rates have the state layout and SI units divided by seconds.
// The source must preserve the wall trace and magnetic divergence. Production
// case files cannot select or inject this callback.
using Source = std::function<State(const State&, const StageTag&, Context&)>;
StepResult advance(Committed&, Cumulative&, Context&, const Source&);
void check_source(const State&, const Context&);
struct SourceRates {
  Totals totals;
  Momentum momentum;
  Vec longitudinal_rate;
};
SourceRates source_rates(const State& source, const State& midpoint,
                         const StageResult&, const Context&);
} // namespace fomhd::verification
