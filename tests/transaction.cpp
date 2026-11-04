#include "fomhd/verification.hpp"
#include "support.hpp"
#include <cstdlib>
#include <iostream>
#include <new>
#include <type_traits>

namespace {
bool fail_next_allocation{};
}

void *operator new(std::size_t size) {
  if (fail_next_allocation) {
    fail_next_allocation = false;
    throw std::bad_alloc();
  }
  if (void *pointer = std::malloc(size ? size : 1))
    return pointer;
  throw std::bad_alloc();
}
void operator delete(void *pointer) noexcept { std::free(pointer); }
void operator delete(void *pointer, std::size_t) noexcept {
  std::free(pointer);
}

using namespace fomhd;

void require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}

struct CommitAllocationProbe final : Compute {
  void synchronize() override {
    if (armed)
      fail_next_allocation = true;
  }
  bool armed{};
};

struct AxpyFailure final : Compute {
  Vec axpy(const Vec &a, const Vec &b, double alpha) override {
    if (fail_on && ++calls == fail_on)
      throw NumericalError("injected combine AXPY failure");
    return Compute::axpy(a, b, alpha);
  }
  int fail_on{}, calls{};
};

int main() {
  try {
    static_assert(std::is_nothrow_move_constructible_v<StepResult>);
    const auto parameters = test::parameters();
    const auto geometry = build_rectangle(4, 4, 1, 0, .2, .2);
    const auto operators = build_operators(geometry, parameters.K);

    {
      CommitAllocationProbe compute;
      Context context{parameters, geometry, operators, compute,
                      parameters.Nphi};
      Cumulative cumulative;
      auto state = initialize(
          closed_reference(context, parameters.ref.ni, parameters.ref.pressure,
                           parameters.ref.pressure, parameters.ref.magnetic),
          context, cumulative, false);
      compute.armed = true;
      bool bad_alloc = false;
      bool accepted = false;
      try {
        auto result = advance(state, cumulative, context);
        fail_next_allocation = false;
        accepted = result.interval.accepted;
      } catch (const std::bad_alloc &) {
        fail_next_allocation = false;
        bad_alloc = true;
      }
      compute.armed = false;
      require(!bad_alloc, "accepted result return allocated");
      require(accepted, "accepted result was not returned");
      require(state.state.step == 1, "accepted result did not commit");
      require(cumulative.accepted_intervals == 1,
              "accepted result did not accumulate interval");
    }

    for (const std::string failed_stage : {"base", "midpoint"}) {
      Compute compute;
      Context context{parameters, geometry, operators, compute,
                      parameters.Nphi};
      Cumulative cumulative;
      auto state = initialize(
          closed_reference(context, parameters.ref.ni, parameters.ref.pressure,
                           parameters.ref.pressure, parameters.ref.magnetic),
          context, cumulative, false);
      const auto before = state.state;
      const auto source = [&](const State &, const StageTag &tag,
                              Context &source_context) {
        if (tag.stage == failed_stage)
          throw NumericalError("injected source callback failure");
        return allocate_state(source_context);
      };
      bool rejected = false;
      try {
        (void)verification::advance(state, cumulative, context, source);
      } catch (const Failure &failure) {
        rejected = true;
        require(failure.stage.has_value(), "source failure lost stage");
        require(failure.stage->stage == failed_stage,
                "source failure used wrong stage");
        require(failure.code == "NumericalLibrary",
                "source failure used wrong code");
        require(!failure.checks.empty(), "source failure lost checks");
        require(!failure.partial_arrays.empty(),
                "source failure lost partial arrays");
      }
      require(rejected, "source callback failure was accepted");
      require(state.state.B == before.B && state.state.step == before.step &&
                  cumulative.accepted_intervals == 0,
              "source callback failure changed committed state");
    }

    for (const int fail_on : {1, 9}) {
      AxpyFailure compute;
      Context context{parameters, geometry, operators, compute,
                      parameters.Nphi};
      Cumulative cumulative;
      auto state = initialize(
          closed_reference(context, parameters.ref.ni, parameters.ref.pressure,
                           parameters.ref.pressure, parameters.ref.magnetic),
          context, cumulative, false);
      compute.fail_on = fail_on;
      bool rejected = false;
      try {
        (void)advance(state, cumulative, context);
      } catch (const Failure &failure) {
        rejected = true;
        require(failure.stage.has_value(), "combine failure lost stage");
        require(failure.stage->stage ==
                    (fail_on == 1 ? "midpoint" : "endpoint"),
                "combine failure used wrong stage");
        require(failure.code == "NumericalLibrary",
                "combine failure used wrong code");
      }
      require(rejected, "combine failure was accepted");
      require(state.state.step == 0 && cumulative.accepted_intervals == 0,
              "combine failure changed committed state");
    }
    std::cout << "PASS transaction\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
