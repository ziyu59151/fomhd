#include "fomhd/backend.hpp"
#include "fomhd/io.hpp"
#include <chrono>
#include <csignal>
#include <iostream>
#include <set>
namespace {
volatile std::sig_atomic_t stop_requested = 0;
void stop(int) { stop_requested = 1; }
const char *usage = "Usage: fomhd validate CASE [--output DIR]\n       fomhd "
                    "run CASE [--output DIR] [--restart CHECKPOINT]\n       "
                    "fomhd inspect RUN_DIR\n       fomhd --help | --version\n";
} // namespace
int main(int argc, char **argv) {
  std::unique_ptr<fomhd::io::Run> run;
  std::optional<std::uint64_t> accepted;
  int phase = 2;
  auto runtime_phase = fomhd::io::RuntimePhase::pre_commit_input;
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << usage;
      return 0;
    }
    if (argc == 2 && std::string(argv[1]) == "--version") {
      std::cout << "FOMHD 0.1.0 A0 C++20 FP64/complex128\n";
      return 0;
    }
    if (argc < 3)
      throw std::runtime_error(usage);
    const std::string command = argv[1];
    if (command == "inspect") {
      if (argc != 3)
        throw std::runtime_error(usage);
      std::cout << fomhd::io::inspect(argv[2]).dump(2) << '\n';
      return 0;
    }
    if (command != "run" && command != "validate")
      throw std::runtime_error(usage);
    std::signal(SIGINT, stop);
    std::signal(SIGTERM, stop);
    fomhd::io::Overrides overrides;
    std::optional<std::filesystem::path> restart;
    std::set<std::string> seen;
    for (int i = 3; i < argc; ++i) {
      std::string option = argv[i];
      if (!seen.insert(option).second || i + 1 >= argc)
        throw std::runtime_error("duplicate or incomplete option");
      if (option == "--output")
        overrides.output = argv[++i];
      else if (option == "--restart" && command == "run")
        restart = argv[++i];
      else
        throw std::runtime_error("unknown option: " + option);
    }
    auto config = fomhd::io::load_case(argv[2], overrides);
    phase = 3;
    std::unique_ptr<fomhd::Compute> compute =
        config.backend == "cpu"
            ? std::make_unique<fomhd::Compute>()
            : fomhd::make_cuda_compute(config.device,
                                       config.parameters.device_memory_limit);
    fomhd::Context context{config.parameters, config.geometry, config.operators,
                           *compute, config.parameters.Nphi};
    context.geometry_hash = config.geometry_hash;
    context.operator_hash = config.operator_hash;
    phase = 5;
    run = std::make_unique<fomhd::io::Run>(config, FOMHD_BUILD_MANIFEST);
    const auto clock = std::chrono::steady_clock::now();
    auto elapsed = [&] {
      return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                           clock)
          .count();
    };
    fomhd::Cumulative cumulative;
    fomhd::StageResult frame;
    fomhd::Committed state;
    std::optional<fomhd::io::Json> restart_parent;
    phase = 4;
    if (restart) {
      auto saved = fomhd::io::read_restart(*restart, config);
      context.next_trial = saved.trial + 1;
      context.next_source = saved.source + 1;
      fomhd::Cumulative discarded;
      state = fomhd::initialize(saved.state, context, discarded, false, &frame);
      cumulative = std::move(saved.cumulative);
      restart_parent = saved.parent;
    } else
      state = fomhd::initialize(fomhd::io::load_initial(config), context,
                                cumulative,
                                config.initial_kind == "equilibrium", &frame);
    accepted = state.state.step;
    phase = 5;
    runtime_phase = fomhd::io::RuntimePhase::post_commit_output;
    if (restart_parent)
      run->parent(*restart_parent);
    run->diagnostics(state.state, frame.checks, frame.totals, frame.momentum,
                     elapsed());
    run->snapshot(state, &frame);
    run->checkpoint(state, cumulative, context);
    if (command == "validate") {
      run->status("validated_initial_state");
      std::cout << "Validated initial state: " << run->path() << '\n';
      return 0;
    }
    run->status("running");
    while (state.state.step < config.parameters.nsteps) {
      if (stop_requested) {
        run->checkpoint(state, cumulative, context);
        run->status("stopped_at_accepted_boundary");
        return 130;
      }
      phase = 4;
      runtime_phase = fomhd::io::RuntimePhase::numerical_trial;
      auto result = fomhd::advance(state, cumulative, context);
      accepted = state.state.step;
      phase = 5;
      runtime_phase = fomhd::io::RuntimePhase::post_commit_output;
      run->diagnostics(state.state, result.checks, result.totals,
                       result.momentum, elapsed());
      if (state.state.step % config.snapshot_stride == 0 ||
          state.state.step == config.parameters.nsteps) {
        if (config.residual_arrays) {
          phase = 4;
          auto tag = state.cache.tag;
          frame = fomhd::evaluate_stage(state.state, tag, context, false);
          phase = 5;
          run->snapshot(state, &frame);
        } else
          run->snapshot(state);
      }
      if (state.state.step % config.checkpoint_stride == 0 ||
          state.state.step == config.parameters.nsteps)
        run->checkpoint(state, cumulative, context);
    }
    compute->synchronize();
    run->status("completed");
    std::cout << "Completed " << state.state.step << " steps: " << run->path()
              << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FOMHD error: " << e.what() << '\n';
    if (run)
      try {
        run->failure(e, accepted, runtime_phase);
      } catch (const std::exception &io) {
        std::cerr << "Failure recording also failed: " << io.what() << '\n';
      }
    return phase;
  }
}
