#include "linear_refinement_fixture.hpp"
#include "support.hpp"
#include "fomhd/io.hpp"

#include <iomanip>
#include <iostream>
#include <string>

using namespace fomhd;

namespace {
class TrackedCompute final : public Compute {
public:
  std::size_t refined_solves{}, refinement_steps{};
  double maximum_error{};

  LinearResult linear(const Matrix &a, const Vec &b, const RealVec &rows,
                      const RealVec &cols, double rank,
                      double tolerance) override {
    auto result = Compute::linear(a, b, rows, cols, rank, tolerance);
    test::linear_refinement::require(result.refinement_steps <= 5 &&
                                        std::isfinite(result.backward_error) &&
                                        result.backward_error <= tolerance,
                                    "invalid disk linear solve diagnostics");
    refined_solves += result.refinement_steps > 0 ? 1 : 0;
    refinement_steps += result.refinement_steps;
    maximum_error = std::max(maximum_error, result.backward_error);
    if (result.refinement_steps)
      std::cout << "disk_solve=" << solve_calls << " dimension=" << a.rows
                << " refinement_steps=" << result.refinement_steps
                << " backward_error=" << result.backward_error << '\n';
    return result;
  }
};

void one_interval(State initial, Context &context) {
  Cumulative cumulative;
  StageResult frame;
  auto committed = initialize(initial, context, cumulative, false, &frame);
  std::cout << "PASS disk_initialize backward_error="
            << frame.cache.transverse_solve.backward_error
            << " refinement_steps=" << frame.cache.transverse_solve.refinement_steps
            << '\n';
  const auto advanced = advance(committed, cumulative, context);
  test::linear_refinement::require(
      advanced.interval.accepted && committed.state.step == 1 &&
          cumulative.accepted_intervals == 1,
      "stepped disk did not accept its first interval");
  std::cout << "PASS disk_advance step=" << committed.state.step
            << " backward_error=" << committed.cache.transverse_solve.backward_error
            << " refinement_steps=" << committed.cache.transverse_solve.refinement_steps
            << '\n';
}

void disk() {
  TrackedCompute compute;
  auto parameters = test::parameters();
  parameters.nsteps = 1;
  auto geometry = build_stepped_disk(6, 6, 1, 0, .1, .1, 1.25, .25, .18);
  auto operators = build_operators(geometry, parameters.K);
  Context context{parameters, geometry, operators, compute, parameters.Nphi,
                  0, 1, 1, 1, {}, {}, {}};
  auto initial = closed_reference(context, parameters.ref.ni,
                                   parameters.ref.pressure, parameters.ref.pressure,
                                   parameters.ref.magnetic);
  one_interval(initial, context);
  std::cout << "disk_refined_solves=" << compute.refined_solves
            << " total_refinement_steps=" << compute.refinement_steps
            << " maximum_error=" << compute.maximum_error << '\n';
}

void captured_case(const char *path) {
  auto config = io::load_case(path);
  TrackedCompute compute;
  Context context{config.parameters, config.geometry, config.operators, compute,
                  config.parameters.Nphi, 0, 1, 1, 1, {}, {}, {}};
  one_interval(io::load_initial(config), context);
  std::cout << "case_refined_solves=" << compute.refined_solves
            << " total_refinement_steps=" << compute.refinement_steps
            << " maximum_error=" << compute.maximum_error << '\n';
}
} // namespace

int main(int argc, char **argv) {
  try {
    std::cout << std::setprecision(17);
    const std::string which = argc > 1 ? argv[1] : "all";
    if (which == "numeric" || which == "all") {
      Compute compute;
      test::linear_refinement::numeric(compute);
    }
    if (which == "disk" || which == "all")
      disk();
    if (which == "case") {
      if (argc != 3)
        throw std::runtime_error("case mode requires a case TOML path");
      captured_case(argv[2]);
    } else if (which != "numeric" && which != "disk" && which != "all")
      throw std::runtime_error("unknown linear refinement test");
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
