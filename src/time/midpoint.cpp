#include "fomhd/verification.hpp"
#include "fomhd/backend.hpp"
#include "fomhd/diagnostics.hpp"
#include "fomhd/physics.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace fomhd {
namespace {
std::uint64_t fresh(std::uint64_t &n) {
  if (n == std::numeric_limits<std::uint64_t>::max())
    throw Failure("CounterOverflow", "time identity exhausted");
  return n++;
}
void accumulate(Ledger &a, const Ledger &b, double dt) {
  const std::array<double Ledger::*, 12> fields{
      &Ledger::electric_node,    &Ledger::electric_edge,
      &Ledger::pressure_kinetic, &Ledger::pressure_internal,
      &Ledger::magnetic,         &Ledger::wall,
      &Ledger::convection,       &Ledger::perp_rate,
      &Ledger::parallel_rate,    &Ledger::spectral_kinetic,
      &Ledger::spectral_perp,    &Ledger::spectral_parallel};
  for (auto f : fields) {
    a.*f += dt * (b.*f);
    if (!std::isfinite(a.*f))
      throw Failure("Ledger", "nonfinite cumulative ledger");
  }
}
bool same_check(const Check &a, const Check &b) {
  return a.name == b.name && a.population == b.population &&
         a.component == b.component && a.producer == b.producer &&
         a.region == b.region && a.norm == b.norm && a.tag == b.tag;
}
void append_missing(std::vector<Check> &target,
                    const std::vector<Check> &source) {
  for (const auto &check : source)
    if (std::none_of(target.begin(), target.end(),
                     [&](const Check &x) { return same_check(x, check); }))
      target.push_back(check);
}
} // namespace
StepResult advance_impl(Committed &committed, Cumulative &cumulative, Context &c,
                        const verification::Source *source) {
  if (!committed.accepted || committed.state.step >= c.p.nsteps)
    throw Failure("Time", "invalid advancement request");
  const auto &left = committed.state;
  const auto n = left.step;
  const auto trial = fresh(c.next_trial);
  if (left.time != time_at_step(c.p, n))
    throw Failure("Time", "accepted time does not match scheduler");
  if (cumulative.last_step != n || cumulative.last_time != left.time)
    throw Failure("Ledger", "cumulative endpoint mismatch");
  const StageTag base_tag{n, left.generation, trial, c.spectral_id, left.time,
                          "base"};
  const StageTag midtag{n, fresh(c.next_generation), trial, c.spectral_id,
                        time_at_midpoint(c.p, n), "midpoint"};
  const StageTag endtag{n + 1,         fresh(c.next_generation), trial,
                        c.spectral_id, time_at_step(c.p, n + 1), "endpoint"};
  auto interval_tag = endtag;
  interval_tag.stage = "interval";
  std::optional<StageResult> base_result, mid_result, end_result;
  std::optional<StepResult> step_result;
  std::optional<State> midstate, candidate;
  State base_source, mid_source;
  bool have_base_source{}, have_mid_source{};
  const auto numerical_failure = [](const StageTag &tag,
                                    const NumericalError &error) {
    Failure failure("NumericalLibrary", error.what());
    failure.stage = tag;
    return failure;
  };
  const auto add_source = [&](const State &rhs, const State &state, const StageTag &tag,
                              State &saved_source, bool &available) {
    if (!source) return rhs;
    try {
      saved_source = (*source)(state, tag, c);
      available = true;
      verification::check_source(saved_source, c);
    } catch (Failure &failure) {
      if (!failure.stage)
        failure.stage = tag;
      throw;
    } catch (const NumericalError &error) {
      throw numerical_failure(tag, error);
    }
    State out=rhs;
    for (int a=0;a<2;++a) {
      out.fluid[a].n=add(out.fluid[a].n,saved_source.fluid[a].n);
      out.fluid[a].mom=add(out.fluid[a].mom,saved_source.fluid[a].mom);
      out.fluid[a].perp=add(out.fluid[a].perp,saved_source.fluid[a].perp);
      if(a)out.fluid[a].parallel=add(out.fluid[a].parallel,saved_source.fluid[a].parallel);
    }
    out.B=add(out.B,saved_source.B);
    return out;
  };
  const auto combine_stage = [&](const State &base, const State &rhs,
                                 double weight, const StageTag &tag) {
    try {
      return combine(base, rhs, weight, tag, c);
    } catch (Failure &failure) {
      if (!failure.stage)
        failure.stage = tag;
      throw;
    } catch (const NumericalError &error) {
      throw numerical_failure(tag, error);
    }
  };
  const auto retain_failure = [&](Failure &failure) {
    const auto failed_stage = failure.stage;
    retain_failure_evidence(failure, base_tag, c, &left,
                            base_result ? &*base_result : nullptr, "base");
    if (have_base_source)
      retain_failure_evidence(failure, base_tag, c, &base_source, nullptr,
                              "base_source");
    retain_failure_evidence(failure, midtag, c,
                            midstate ? &*midstate : nullptr,
                            mid_result ? &*mid_result : nullptr, "midpoint");
    if (have_mid_source)
      retain_failure_evidence(failure, midtag, c, &mid_source, nullptr,
                              "midpoint_source");
    retain_failure_evidence(failure, endtag, c,
                            candidate ? &*candidate : nullptr,
                            end_result ? &*end_result : nullptr, "endpoint");
    if (step_result) {
      append_missing(failure.checks, step_result->checks);
      for (int a = 0; a < 2; ++a) {
        const std::string id = a ? "e" : "i";
        failure.partial_arrays.try_emplace(
            "interval_number_flux_" + id, step_result->interval.number_flux[a]);
        failure.partial_arrays.try_emplace(
            "interval_current_" + id, step_result->interval.current[a]);
      }
    }
    failure.stage = failed_stage ? failed_stage : interval_tag;
  };
  try {
    base_result.emplace(evaluate_stage(left, base_tag, c, false));
    auto &base = *base_result;
    midstate.emplace(combine_stage(
        left,
        add_source(base.rhs, left, base.tag, base_source, have_base_source),
        c.p.dt / 2, midtag));
    mid_result.emplace(evaluate_stage(*midstate, midtag, c, true));
    auto &mid = *mid_result;
    candidate.emplace(combine_stage(
        left,
        add_source(mid.rhs, *midstate, mid.tag, mid_source, have_mid_source),
        c.p.dt, endtag));
    end_result.emplace(evaluate_stage(*candidate, endtag, c, false));
    auto &end = *end_result;
    step_result.emplace(check_step(left, base, *midstate, mid, *candidate, end,
                                   cumulative, c,
                                   source ? &mid_source : nullptr));
    auto &result = *step_result;
    result.checks.insert(result.checks.end(), base.checks.begin(),
                         base.checks.end());
    result.checks.insert(result.checks.end(), mid.checks.begin(),
                         mid.checks.end());
    result.checks.insert(result.checks.end(), end.checks.begin(),
                         end.checks.end());
    gate(result.checks);
    if (cumulative.accepted_intervals && cumulative.last_trial == trial)
      throw Failure("DuplicateInterval", "interval already accumulated");
    Cumulative pending = cumulative;
    for (int a = 0; a < 2; ++a) {
      accumulate(pending.integrals[a], mid.ledger[a], c.p.dt);
      pending.boundary_integrals.number[a] += c.p.dt * mid.boundary.number[a];
      pending.boundary_integrals.kinetic[a] += c.p.dt * mid.boundary.kinetic[a];
      pending.boundary_integrals.internal_perp[a] +=
          c.p.dt * mid.boundary.internal_perp[a];
      pending.boundary_integrals.internal_parallel[a] +=
          c.p.dt * mid.boundary.internal_parallel[a];
    }
    pending.boundary_integrals.field_energy +=
        c.p.dt * mid.boundary.field_energy;
    pending.boundary_integrals.total_energy +=
        c.p.dt * mid.boundary.total_energy;
    ++pending.accepted_intervals;
    pending.last_trial = trial;
    pending.last_step = n + 1;
    pending.last_time = endtag.time;
    c.compute.synchronize();
    // All allocations and rejectable work finish before the single commit.
    static_assert(std::is_nothrow_swappable_v<State> &&
                  std::is_nothrow_swappable_v<FieldCache> &&
                  std::is_nothrow_swappable_v<Cumulative> &&
                  std::is_nothrow_move_constructible_v<StepResult>);
    std::swap(committed.state, *candidate);
    std::swap(committed.cache, end.cache);
    std::swap(cumulative, pending);
    result.interval.accepted = true;
    return std::move(result);
  } catch (Failure &failure) {
    retain_failure(failure);
    throw;
  } catch (const NumericalError &error) {
    Failure failure("NumericalLibrary", error.what());
    retain_failure(failure);
    throw failure;
  }
}
StepResult advance(Committed &state, Cumulative &cumulative, Context &c) {
  return advance_impl(state,cumulative,c,nullptr);
}
namespace verification {
StepResult advance(Committed &state, Cumulative &cumulative, Context &c, const Source &source) {
  if (!source) throw Failure("ManufacturedSource", "a source callback is required");
  return advance_impl(state,cumulative,c,&source);
}
} // namespace verification
} // namespace fomhd
