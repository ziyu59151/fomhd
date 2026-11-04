#pragma once
#include "fomhd/hash.hpp"
#include "fomhd/npz.hpp"
#include "fomhd/physics.hpp"
#include <filesystem>
#include <functional>
#include <nlohmann/json.hpp>
#include <optional>
namespace fomhd::io {
using Json = nlohmann::json;
using Arrays = std::map<std::string, Array>;
enum class RuntimePhase {
  pre_commit_input,
  numerical_trial,
  post_commit_output,
};
const char *runtime_phase_name(RuntimePhase) noexcept;
const char *failure_class(RuntimePhase) noexcept;
const char *failure_status(RuntimePhase) noexcept;
struct Case {
  Parameters parameters;
  Geometry geometry;
  Operators operators;
  Limits limits;
  std::filesystem::path case_path, geometry_path, initial_path, source_path,
      output;
  std::string name, backend, initial_kind, raw, config_hash, semantic_hash,
      geometry_hash, operator_hash;
  int device{};
  std::uint64_t snapshot_stride{}, checkpoint_stride{};
  bool residual_arrays{};
  Json resolved, source, input_hashes;
};
struct Overrides {
  std::optional<std::filesystem::path> output;
  std::optional<std::uint64_t> nsteps;
};
Json read_json(const std::filesystem::path &);
Json parse_json(const std::string &);
std::string read_text(const std::filesystem::path &,
                      std::size_t cap = 4U << 20);
void write_text(const std::filesystem::path &, const std::string &);
void require_finite_json(const Json &);
Case load_case(const std::filesystem::path &, Overrides = {});
Arrays encode_state(const State &, const Geometry &);
State decode_state(const Arrays &, const Geometry &, int K);
State load_initial(const Case &);
Array encode_vertex(const Vec &, const Geometry &, int K);
Vec decode_vertex(const Array &, const Geometry &, int K);
struct PublicationError : std::runtime_error {
  std::string phase;
  std::filesystem::path temporary;
  bool published;
  PublicationError(std::string p, std::filesystem::path t, bool v,
                   const std::string &message);
};
using Fault = std::function<void(const std::string &)>;
std::string
atomic_publish(const std::filesystem::path &,
               const std::function<void(const std::filesystem::path &)> &,
               bool replace = false, const Fault &fault = {});
struct Restart {
  State state;
  Cumulative cumulative;
  Json parent;
  std::uint64_t trial{}, source{};
};
Restart read_restart(const std::filesystem::path &, const Case &);
class Run {
public:
  Run(const Case &, const std::filesystem::path &build_manifest);
  ~Run();
  Run(const Run &) = delete;
  Run &operator=(const Run &) = delete;
  void parent(const Json &);
  void diagnostics(const State &, const std::vector<Check> &, const Totals &,
                   const Momentum &, double elapsed);
  void snapshot(const Committed &, const StageResult *captured = nullptr);
  void checkpoint(const Committed &, const Cumulative &, const Context &);
  void status(const std::string &);
  void failure(const std::exception &,
               std::optional<std::uint64_t> accepted_step,
               RuntimePhase phase = RuntimePhase::pre_commit_input);
  const std::filesystem::path &path() const { return directory_; }

private:
  const Case &case_;
  std::filesystem::path directory_;
  Json manifest_;
  int lock_{-1};
  std::optional<std::uint64_t> last_history_;
  void publish_manifest();
};
Json inspect(const std::filesystem::path &);
} // namespace fomhd::io
