#include "fomhd/backend.hpp"
#include "fomhd/io.hpp"
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>
namespace fomhd {
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Ledger, electric_node, electric_edge,
                                   pressure_kinetic, pressure_internal,
                                   magnetic, wall, convection, perp_rate,
                                   parallel_rate, spectral_kinetic,
                                   spectral_perp, spectral_parallel)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Totals, number, kinetic, ion_internal,
                                   electron_perp, electron_parallel, magnetic,
                                   longitudinal, energy, topology,
                                   physical_flux)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Momentum, value, outward, wall_reaction,
                                   angular, outward_torque, wall_torque,
                                   angular_samples)
namespace io {
const char *runtime_phase_name(RuntimePhase phase) noexcept {
  switch (phase) {
  case RuntimePhase::pre_commit_input:
    return "pre_commit_input";
  case RuntimePhase::numerical_trial:
    return "numerical_trial";
  case RuntimePhase::post_commit_output:
    return "post_commit_output";
  }
  return "unknown";
}
const char *failure_class(RuntimePhase phase) noexcept {
  switch (phase) {
  case RuntimePhase::pre_commit_input:
    return "pre_commit_or_input";
  case RuntimePhase::numerical_trial:
    return "rejected_numerical_trial";
  case RuntimePhase::post_commit_output:
    return "output_after_commit";
  }
  return "unknown";
}
const char *failure_status(RuntimePhase phase) noexcept {
  switch (phase) {
  case RuntimePhase::pre_commit_input:
    return "failed_before_commit";
  case RuntimePhase::numerical_trial:
    return "failed_numerical_trial";
  case RuntimePhase::post_commit_output:
    return "io_failed_after_commit";
  }
  return "failed";
}
namespace {
struct FD {
  int fd{-1};
  explicit FD(int v) : fd(v) {}
  ~FD() {
    if (fd >= 0)
      ::close(fd);
  }
  FD(const FD &) = delete;
};
std::string error(const char *call) {
  return std::string(call) + ": " + std::strerror(errno);
}
std::string step_name(const char *prefix, std::uint64_t step,
                      const char *suffix) {
  std::ostringstream s;
  s << prefix << std::setw(10) << std::setfill('0') << step << suffix;
  return s.str();
}
void sync_file(const std::filesystem::path &path) {
  FD f(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (f.fd < 0 || ::fsync(f.fd) != 0)
    throw std::runtime_error(error("fsync file"));
}
void append(const std::filesystem::path &p, const std::string &value) {
  std::ofstream f(p, std::ios::binary | std::ios::app);
  f << value;
  f.flush();
  if (!f)
    throw std::runtime_error("append failed: " + p.string());
}
Json value(double x) {
  if (std::isfinite(x))
    return x;
  return Json{{"value", std::isnan(x) ? "NaN"
                        : x > 0       ? "+Inf"
                                      : "-Inf"},
              {"type", "float64"}};
}
Json check_json(const Check &x) {
  return {{"name", x.name},
          {"population", x.population},
          {"component", x.component},
          {"producer", x.producer},
          {"region", x.region},
          {"selected_count", x.selected},
          {"step", x.tag.step},
          {"generation", x.tag.generation},
          {"trial", x.tag.trial},
          {"spectral_context", x.tag.spectral_context},
          {"time", value(x.tag.time)},
          {"stage", x.tag.stage},
          {"raw", value(x.raw)},
          {"reference", value(x.reference)},
          {"threshold", value(x.threshold)},
          {"unit", x.unit},
          {"value_unit", x.value_unit},
          {"norm", x.norm == Norm::linf          ? "linf"
                   : x.norm == Norm::weighted_l2 ? "weighted_L2"
                                                 : "value"},
          {"gate", x.hard ? "hard" : "observe"},
          {"status", !x.available ? "UNAVAILABLE"
                     : x.passed   ? "PASS"
                                  : "FAIL"},
          {"reason", x.reason},
          {"worst_index", x.worst}};
}
Json stage_json(const StageTag &x) {
  return {{"step", x.step},
          {"generation", x.generation},
          {"trial", x.trial},
          {"spectral_context", x.spectral_context},
          {"time", value(x.time)},
          {"stage", x.stage}};
}
Json boundary_json(const BoundaryLedger &x) {
  return {{"number", x.number},
          {"kinetic", x.kinetic},
          {"internal_perp", x.internal_perp},
          {"internal_parallel", x.internal_parallel},
          {"field_energy", x.field_energy},
          {"total_energy", x.total_energy}};
}
BoundaryLedger boundary_from(const Json &j) {
  BoundaryLedger x;
  x.number = j.at("number");
  x.kinetic = j.at("kinetic");
  x.internal_perp = j.at("internal_perp");
  x.internal_parallel = j.at("internal_parallel");
  x.field_energy = j.at("field_energy");
  x.total_energy = j.at("total_energy");
  return x;
}
Json cumulative_json(const Cumulative &x) {
  return {{"baseline",
           {{"totals", x.baseline.totals},
            {"momentum", x.baseline.momentum},
            {"step", x.baseline.step},
            {"time", x.baseline.time}}},
          {"integrals", x.integrals},
          {"boundary_integrals", boundary_json(x.boundary_integrals)},
          {"accepted_intervals", x.accepted_intervals},
          {"last_trial", x.last_trial},
          {"last_step", x.last_step},
          {"last_time", x.last_time}};
}
Cumulative cumulative_from(const Json &j) {
  Cumulative x;
  const auto &b = j.at("baseline");
  x.baseline.totals = b.at("totals").get<Totals>();
  x.baseline.momentum = b.at("momentum").get<Momentum>();
  x.baseline.step = b.at("step").get<std::uint64_t>();
  x.baseline.time = b.at("time").get<double>();
  x.integrals = j.at("integrals").get<std::array<Ledger, 2>>();
  x.boundary_integrals = boundary_from(j.at("boundary_integrals"));
  x.accepted_intervals = j.at("accepted_intervals");
  x.last_trial = j.at("last_trial");
  x.last_step = j.at("last_step");
  x.last_time = j.at("last_time");
  return x;
}
void accepted(const Committed &u) {
  if (!u.accepted || u.cache.tag.step != u.state.step ||
      u.cache.tag.generation != u.state.generation ||
      u.cache.tag.time != u.state.time ||
      (u.cache.tag.stage != "initial" && u.cache.tag.stage != "restart" &&
       u.cache.tag.stage != "endpoint"))
    throw std::runtime_error("output requires a matching accepted cache");
}
void identity(Arrays &a, const Committed &u, const Case &c,
              const std::string &run) {
  accepted(u);
  a["step"] = from_uint64(u.state.step);
  a["generation"] = from_uint64(u.state.generation);
  a["time"] = from_double(u.state.time);
  a["dt"] = from_double(c.parameters.dt);
  a["start_time"] = from_double(c.parameters.start_time);
  a["accepted"] = from_bool(true);
  a["schema_version"] = from_uint64(1);
  a["layout_version"] = from_uint64(1);
  a["model_id"] = from_utf8("A0");
  a["config_hash"] = from_utf8(c.config_hash);
  a["semantic_hash"] = from_utf8(c.semantic_hash);
  a["geometry_hash"] = from_utf8(c.geometry_hash);
  a["operator_rule_hash"] = from_utf8(sha256("A0-compatible-operators-v2"));
  a["matrix_fingerprint"] = from_utf8(c.operator_hash);
  a["original_run_id"] = from_utf8(run);
}
void json_publish(const std::filesystem::path &p, const Json &j,
                  bool replace = false) {
  require_finite_json(j);
  atomic_publish(p, [&](auto t) { write_text(t, j.dump(2) + "\n"); }, replace);
}
void copy_verified(const std::filesystem::path &source,
                   const std::filesystem::path &target,
                   const std::string &hash) {
  if (sha256_file(source) != hash)
    throw std::runtime_error("source changed before provenance copy: " +
                             source.string());
  std::filesystem::copy_file(source, target);
  if (sha256_file(target) != hash)
    throw std::runtime_error("provenance copy hash mismatch");
}
} // namespace
PublicationError::PublicationError(std::string p, std::filesystem::path t,
                                   bool v, const std::string &message)
    : std::runtime_error(p + ": " + message + "; temporary=" + t.string()),
      phase(std::move(p)), temporary(std::move(t)), published(v) {}
std::string
atomic_publish(const std::filesystem::path &target,
               const std::function<void(const std::filesystem::path &)> &writer,
               bool replace, const Fault &fault) {
  static std::atomic<std::uint64_t> sequence{0};
  const auto parent = target.parent_path().empty() ? std::filesystem::path{"."}
                                                   : target.parent_path();
  if (std::filesystem::is_symlink(std::filesystem::symlink_status(parent)))
    throw std::runtime_error("output parent must not be a symbolic link");
  if (!replace &&
      std::filesystem::exists(std::filesystem::symlink_status(target)))
    throw std::runtime_error("target already exists");
  auto temp =
      parent / (target.filename().string() + ".tmp." +
                std::to_string(getpid()) + "." + std::to_string(++sequence));
  std::string phase = "temp_created";
  bool published = false;
  try {
    {
      FD f(::open(temp.c_str(),
                  O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC | O_NOFOLLOW, 0600));
      if (f.fd < 0)
        throw std::runtime_error(error("create temp"));
    }
    if (fault)
      fault(phase);
    writer(temp);
    phase = "written";
    if (fault)
      fault(phase);
    sync_file(temp);
    const auto hash = sha256_file(temp);
    phase = "validated";
    if (fault)
      fault(phase);
    if (replace) {
      if (::rename(temp.c_str(), target.c_str()) != 0)
        throw std::runtime_error(error("rename"));
    } else {
      if (::link(temp.c_str(), target.c_str()) != 0)
        throw std::runtime_error(error("no-clobber publish"));
      published = true;
      if (::unlink(temp.c_str()) != 0)
        throw std::runtime_error(error("unlink temp"));
    }
    published = true;
    phase = "renamed";
    if (fault)
      fault(phase);
    FD directory(::open(parent.c_str(),
                        O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (directory.fd < 0 || ::fsync(directory.fd) != 0)
      throw std::runtime_error(error("fsync directory"));
    phase = "directory_synced";
    if (fault)
      fault(phase);
    return hash;
  } catch (const std::exception &e) {
    throw PublicationError(phase, temp, published, e.what());
  }
}
Run::Run(const Case &c, const std::filesystem::path &build_manifest)
    : case_(c), directory_(c.output) {
  const auto status = std::filesystem::symlink_status(directory_);
  if (std::filesystem::is_symlink(status) ||
      (std::filesystem::exists(status) &&
       (!std::filesystem::is_directory(status) ||
        !std::filesystem::is_empty(directory_))))
    throw std::runtime_error("RunDirectoryNotEmpty: " + directory_.string());
  std::filesystem::create_directories(directory_);
  lock_ = ::open((directory_ / "lock").c_str(),
                 O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (lock_ < 0)
    throw std::runtime_error(error("exclusive run lock"));
  try {
    for (auto name : {"inputs", "provenance", "logs", "snapshots",
                      "checkpoints", "failure"})
      std::filesystem::create_directory(directory_ / name);
    auto build = read_json(build_manifest);
    const auto binary = build.at("binary_path").get<std::string>(),
               binary_hash = build.at("binary_sha256").get<std::string>();
    copy_verified(binary, directory_ / "provenance/fomhd", binary_hash);
    if (sha256_running_executable() != binary_hash)
      throw std::runtime_error(
          "build manifest does not identify this executable");
    const auto snapshot =
        std::filesystem::path(build.at("snapshot_path").get<std::string>());
    const auto files = build.at("source_files");
    if (!files.is_object() || files.empty())
      throw std::runtime_error("build source snapshot is missing");
    for (auto i = files.begin(); i != files.end(); ++i) {
      const auto relative = std::filesystem::path(i.key());
      if (relative.is_absolute() ||
          relative.string().find("..") != std::string::npos)
        throw std::runtime_error("invalid source snapshot path");
      auto dest = directory_ / "provenance/source" / relative;
      std::filesystem::create_directories(dest.parent_path());
      copy_verified(snapshot / relative, dest, i.value());
    }
    copy_verified(build_manifest, directory_ / "provenance/build_manifest.json",
                  sha256_file(build_manifest));
    write_text(directory_ / "inputs/case.toml", c.raw);
    copy_verified(c.geometry_path, directory_ / "inputs/geometry.npz",
                  c.input_hashes.at("geometry"));
    copy_verified(c.initial_path, directory_ / "inputs/initial.npz",
                  c.input_hashes.at("initial"));
    copy_verified(c.source_path, directory_ / "inputs/source_manifest.json",
                  c.input_hashes.at("source"));
    json_publish(directory_ / "inputs/resolved.json", c.resolved);
    auto stamp = std::chrono::system_clock::now().time_since_epoch().count();
    const std::string id =
        c.name + "__" + std::to_string(stamp) + "__" + std::to_string(getpid());
    manifest_ = {
        {"schema_version", 1},
        {"run_id", id},
        {"status", "initializing"},
        {"model", "A0"},
        {"units", "SI"},
        {"layout_version", 1},
        {"config_hash", c.config_hash},
        {"semantic_hash", c.semantic_hash},
        {"geometry_hash", c.geometry_hash},
        {"matrix_fingerprint", c.operator_hash},
        {"input_hashes", c.input_hashes},
        {"build", build},
        {"backend", c.backend},
        {"precision", "fp64/complex128"},
        {"source", c.source},
        {"last_accepted_step", nullptr},
        {"last_accepted_time", nullptr},
        {"durable_checkpoint", nullptr},
        {"geometry_distance_m",
         {{"lower", c.geometry.diagnostics.hausdorff_lower_m},
          {"upper", c.geometry.diagnostics.hausdorff_upper_m}}},
        {"history_columns",
         "step,time_s,number_i,number_e,kinetic_i_J,kinetic_e_J,ion_internal_J,"
         "electron_perp_J,electron_parallel_J,magnetic_J,longitudinal_J,total_"
         "energy_J,topology_Wb,physical_flux_Wb,Lz_kg_m2_per_s,elapsed_s"}};
    if (const char *job = std::getenv("SLURM_JOB_ID"))
      manifest_["slurm_job_id"] = job;
    publish_manifest();
    write_text(directory_ / "history.csv",
               manifest_["history_columns"].get<std::string>() + "\n");
    write_text(directory_ / "checks.jsonl", "");
    write_text(directory_ / "observations.jsonl", "");
    write_text(directory_ / "logs/run.log", "Run initialized.\n");
  } catch (...) {
    ::close(lock_);
    lock_ = -1;
    throw;
  }
}
Run::~Run() {
  if (lock_ >= 0)
    ::close(lock_);
}
void Run::publish_manifest() {
  json_publish(directory_ / "run.json", manifest_, true);
}
void Run::parent(const Json &p) {
  manifest_["parent"] = p;
  publish_manifest();
}
void Run::status(const std::string &s) {
  manifest_["status"] = s;
  publish_manifest();
  append(directory_ / "logs/run.log", s + "\n");
}
void Run::diagnostics(const State &u, const std::vector<Check> &checks,
                      const Totals &t, const Momentum &m, double elapsed) {
  if (last_history_ && u.step != *last_history_ + 1)
    throw std::runtime_error("duplicate or noncontiguous accepted history");
  if (checks.empty())
    throw std::runtime_error("accepted result has no diagnostic records");
  for (const auto &c : checks)
    append(directory_ / "checks.jsonl", check_json(c).dump() + "\n");
  Json finite = {{"totals", t}, {"momentum", m}, {"elapsed", elapsed}};
  require_finite_json(finite);
  std::ostringstream s;
  s << std::setprecision(17) << u.step << ',' << u.time << ',' << t.number[0]
    << ',' << t.number[1] << ',' << t.kinetic[0] << ',' << t.kinetic[1] << ','
    << t.ion_internal << ',' << t.electron_perp << ',' << t.electron_parallel
    << ',' << t.magnetic << ',' << t.longitudinal << ',' << t.energy << ','
    << t.topology << ',' << t.physical_flux << ',' << m.angular << ','
    << elapsed << '\n';
  append(directory_ / "history.csv", s.str());
  const auto &observations =
      case_.resolved.at("diagnostics").at("mode_observations");
  if (!observations.empty()) {
    auto arrays = encode_state(u, case_.geometry);
    for (const auto &ob : observations) {
      auto field = ob.at("field").get<std::string>(),
           component = ob.at("component").get<std::string>();
      if (field == "B")
        field = "B_" + component;
      const auto &a = arrays.at(field);
      const auto values = to_complex(a);
      auto shape = a.shape;
      std::size_t comp = 0;
      if (shape.size() == 4) {
        if (component == "phi")
          comp = 1;
        else if (component == "Z")
          comp = 2;
        else if (component != "R")
          throw std::runtime_error("unknown observation vector component");
      } else if (component != "scalar" && field != "B_" + component)
        throw std::runtime_error(
            "scalar observation requires scalar component");
      auto ix = ob.at("entity_key").at(0).get<std::size_t>(),
           iy = ob.at("entity_key").at(1).get<std::size_t>();
      auto nx = shape.back(), ny = shape[shape.size() - 2];
      if (ix >= nx || iy >= ny)
        throw std::runtime_error("observation entity outside grid");
      auto k = ob.at("mode").get<int>() + case_.parameters.K;
      auto z = values[(comp * (2 * case_.parameters.K + 1) + k) * nx * ny +
                      iy * nx + ix];
      append(directory_ / "observations.jsonl", Json{{"step", u.step},
                                                     {"time", u.time},
                                                     {"definition", ob},
                                                     {"real", z.real()},
                                                     {"imag", z.imag()}}
                                                        .dump() +
                                                    "\n");
    }
  }
  last_history_ = u.step;
  manifest_["last_accepted_step"] = u.step;
  manifest_["last_accepted_time"] = u.time;
  publish_manifest();
}
void Run::snapshot(const Committed &u, const StageResult *frame) {
  auto a = encode_state(u.state, case_.geometry);
  identity(a, u, case_, manifest_["run_id"]);
  if (frame) {
    if (frame->tag != u.cache.tag)
      throw std::runtime_error("snapshot frame is stale");
    a["derived_E"] = from_complex(
        frame->cache.E, {static_cast<std::size_t>(2 * case_.parameters.K + 1),
                         case_.operators.Ne});
    a["derived_phi"] = from_complex(
        frame->cache.phi, {static_cast<std::size_t>(2 * case_.parameters.K + 1),
                           case_.operators.Nc});
    if (case_.residual_arrays)
      for (const auto &[key, v] : frame->residuals)
        a["derived_residual_" + key] = from_complex(v, {v.size()});
    const auto solve = [&](const std::string &prefix, const LinearResult &r) {
      a[prefix + "_singular_values"] =
          from_doubles(r.singular_values, {r.singular_values.size()});
      a[prefix + "_row_scale"] =
          from_doubles(r.row_scale, {r.row_scale.size()});
      a[prefix + "_column_scale"] =
          from_doubles(r.column_scale, {r.column_scale.size()});
      a[prefix + "_physical_residual"] =
          from_complex(r.residual, {r.residual.size()});
      a[prefix + "_backward_error"] = from_double(r.backward_error);
      a[prefix + "_refinement_steps"] = from_uint64(r.refinement_steps);
    };
    for (std::size_t k = 0; k < frame->cache.poisson_solves.size(); ++k)
      solve("derived_poisson_" + std::to_string(k),
            frame->cache.poisson_solves[k]);
    solve("derived_darwin", frame->cache.transverse_solve);
  } else if (case_.residual_arrays)
    throw std::runtime_error(
        "requested residual arrays require an accepted diagnostic frame");
  auto path =
      directory_ / "snapshots" / step_name("state_", u.state.step, ".npz");
  atomic_publish(path, [&](auto t) { write_npz_temporary(t, a); });
}
void Run::checkpoint(const Committed &u, const Cumulative &cumulative,
                     const Context &c) {
  accepted(u);
  if (cumulative.last_step != u.state.step ||
      cumulative.last_time != u.state.time)
    throw std::runtime_error("checkpoint ledger endpoint mismatch");
  const auto &durable = manifest_.at("durable_checkpoint");
  if (!durable.is_null() &&
      durable.at("step").get<std::uint64_t>() == u.state.step) {
    const auto pointer_path = directory_ / "checkpoints/latest.json";
    if (!std::filesystem::is_regular_file(pointer_path))
      throw std::runtime_error(
          "durable checkpoint pointer is missing or not a regular file");
    const auto pointer = read_json(pointer_path);
    if (pointer.at("file") != durable.at("file") ||
        pointer.at("sha256") != durable.at("sha256") ||
        pointer.at("step") != durable.at("step") ||
        pointer.at("time") != durable.at("time") ||
        pointer.at("accepted") != durable.at("accepted"))
      throw std::runtime_error(
          "durable checkpoint pointer does not match manifest");
    const auto path =
        directory_ / "checkpoints" / durable.at("file").get<std::string>();
    if (!std::filesystem::is_regular_file(path) ||
        sha256_file(path) != durable.at("sha256").get<std::string>())
      throw std::runtime_error(
          "durable checkpoint metadata does not match its file");
    return;
  }
  c.compute.synchronize();
  auto a = encode_state(u.state, case_.geometry);
  identity(a, u, case_, manifest_["run_id"]);
  auto sums = cumulative_json(cumulative);
  require_finite_json(sums);
  a["cumulative_json"] = from_utf8(sums.dump());
  a["baseline_divB"] =
      encode_vertex(cumulative.baseline.divB, case_.geometry, c.p.K);
  a["last_source_id"] = from_uint64(c.next_source - 1);
  a["last_trial_id"] = from_uint64(c.next_trial - 1);
  auto path =
      directory_ / "checkpoints" / step_name("restart_", u.state.step, ".npz");
  auto hash = atomic_publish(path, [&](auto t) { write_npz_temporary(t, a); });
  Json latest = {{"file", path.filename().string()},
                 {"sha256", hash},
                 {"step", u.state.step},
                 {"time", u.state.time},
                 {"accepted", true}};
  sync_file(directory_ / "history.csv");
  sync_file(directory_ / "checks.jsonl");
  json_publish(directory_ / "checkpoints/latest.json", latest, true);
  manifest_["durable_checkpoint"] = latest;
  publish_manifest();
}
void Run::failure(const std::exception &e, std::optional<std::uint64_t> step,
                  RuntimePhase phase) {
  static std::atomic<std::uint64_t> sequence{0};
  const auto stamp =
      std::chrono::system_clock::now().time_since_epoch().count();
  const auto id = "failure_" + std::to_string(stamp) + "_" +
                  std::to_string(getpid()) + "_" +
                  std::to_string(++sequence);
  Json record = {{"error", e.what()},
                 {"last_accepted_step", step ? Json(*step) : Json(nullptr)},
                 {"accepted_trial",
                  phase == RuntimePhase::post_commit_output},
                 {"runtime_phase", runtime_phase_name(phase)},
                 {"failure_class", failure_class(phase)},
                 {"last_durable_checkpoint",
                  manifest_.at("durable_checkpoint")}};
  if (auto f = dynamic_cast<const Failure *>(&e)) {
    record["code"] = f->code;
    record["checks"] = Json::array();
    for (auto x : f->checks)
      record["checks"].push_back(check_json(x));
    record["stage"] = f->stage.has_value() ? stage_json(*f->stage) : Json(nullptr);
    record["partial_array_keys"] = Json::array();
    for (const auto &[key, values] : f->partial_arrays) {
      static_cast<void>(values);
      record["partial_array_keys"].push_back(key);
    }
    record["partial_array_count"] = f->partial_arrays.size();
    if (!f->partial_arrays.empty()) {
      Arrays arrays;
      for (const auto &[key, values] : f->partial_arrays)
        arrays.emplace(key, from_complex(values, {values.size()}));
      const auto partial_path = directory_ / "failure" / (id + ".npz");
      try {
        const auto hash = atomic_publish(
            partial_path,
            [&](const auto &temporary) { write_npz_temporary(temporary, arrays); });
        record["partial_file"] = partial_path.filename().string();
        record["partial_sha256"] = hash;
        record["partial_published"] = true;
      } catch (const std::exception &partial_error) {
        record["partial_published"] = false;
        record["partial_write_error"] = partial_error.what();
        if (const auto *publication =
                dynamic_cast<const PublicationError *>(&partial_error)) {
          record["partial_publication_phase"] = publication->phase;
          record["partial_target_may_be_published"] = publication->published;
          record["partial_temporary"] = publication->temporary.string();
        }
      }
    } else {
      record["partial_published"] = false;
    }
  }
  if (auto f = dynamic_cast<const PublicationError *>(&e)) {
    record["publication_phase"] = f->phase;
    record["target_may_be_published"] = f->published;
    record["temporary"] = f->temporary.string();
  }
  json_publish(directory_ / "failure" / (id + ".json"), record);
  manifest_["failure"] = record;
  status(failure_status(phase));
}
Restart read_restart(const std::filesystem::path &path, const Case &c) {
  Keys allowed{"n_i",
               "mom_i",
               "p_i",
               "n_e",
               "mom_e",
               "p_perp_e",
               "p_parallel_e",
               "B_R",
               "B_phi",
               "B_Z",
               "step",
               "generation",
               "time",
               "dt",
               "start_time",
               "accepted",
               "schema_version",
               "layout_version",
               "model_id",
               "config_hash",
               "semantic_hash",
               "geometry_hash",
               "operator_rule_hash",
               "matrix_fingerprint",
               "original_run_id",
               "cumulative_json",
               "baseline_divB",
               "last_source_id",
               "last_trial_id"};
  auto a = read_npz(path, c.limits, allowed);
  if (a.size() != allowed.size())
    throw std::runtime_error("checkpoint schema is incomplete");
  if (!to_bool(a.at("accepted")) || to_uint64(a.at("schema_version")) != 1 ||
      to_uint64(a.at("layout_version")) != 1 ||
      to_utf8(a.at("model_id")) != "A0")
    throw std::runtime_error("checkpoint was not an accepted A0 state");
  if (to_utf8(a.at("semantic_hash")) != c.semantic_hash ||
      to_utf8(a.at("geometry_hash")) != c.geometry_hash ||
      to_utf8(a.at("matrix_fingerprint")) != c.operator_hash ||
      to_utf8(a.at("operator_rule_hash")) !=
          sha256("A0-compatible-operators-v2"))
    throw std::runtime_error(
        "restart semantic or operator fingerprint mismatch");
  if (to_double(a.at("dt")) != c.parameters.dt ||
      to_double(a.at("start_time")) != c.parameters.start_time)
    throw std::runtime_error("restart time configuration mismatch");
  Restart result;
  result.state = decode_state(a, c.geometry, c.parameters.K);
  auto &u = result.state;
  u.step = to_uint64(a.at("step"));
  u.time = to_double(a.at("time"));
  u.generation = to_uint64(a.at("generation"));
  if (u.step > c.parameters.nsteps ||
      u.time != time_at_step(c.parameters, u.step) ||
      u.generation == UINT64_MAX)
    throw std::runtime_error("restart scheduler identity mismatch");
  ++u.generation;
  auto sums = parse_json(to_utf8(a.at("cumulative_json")));
  result.cumulative = cumulative_from(sums);
  auto &ledger = result.cumulative;
  ledger.baseline.divB =
      decode_vertex(a.at("baseline_divB"), c.geometry, c.parameters.K);
  if (ledger.baseline.step > u.step ||
      ledger.baseline.time !=
          time_at_step(c.parameters, ledger.baseline.step) ||
      ledger.last_step != u.step || ledger.last_time != u.time ||
      ledger.accepted_intervals != u.step - ledger.baseline.step)
    throw std::runtime_error("restart baseline or interval identity mismatch");
  if (!ledger.accepted_intervals) {
    Cumulative zero;
    zero.baseline = ledger.baseline;
    zero.last_step = u.step;
    zero.last_time = u.time;
    if (cumulative_json(zero) != sums)
      throw std::runtime_error(
          "initial checkpoint contains nonzero cumulative ledger");
  } else if (ledger.last_trial == 0)
    throw std::runtime_error("restart interval trial is missing");
  result.trial = to_uint64(a.at("last_trial_id"));
  result.source = to_uint64(a.at("last_source_id"));
  if (result.trial == UINT64_MAX || result.source == UINT64_MAX ||
      result.trial < ledger.last_trial)
    throw std::runtime_error("restart counters are invalid");
  const auto hash = sha256_file(path);
  auto latest_path = path.parent_path() / "latest.json";
  if (std::filesystem::exists(latest_path)) {
    auto latest = read_json(latest_path);
    if (latest.at("file") == path.filename().string() &&
        latest.at("sha256") != hash)
      throw std::runtime_error(
          "checkpoint file hash does not match its durable pointer");
  }
  result.parent = {{"checkpoint", std::filesystem::absolute(path).string()},
                   {"sha256", hash},
                   {"original_run_id", to_utf8(a.at("original_run_id"))},
                   {"original_config_hash", to_utf8(a.at("config_hash"))}};
  return result;
}
Json inspect(const std::filesystem::path &dir) {
  auto result = read_json(dir / "run.json");
  auto history = read_text(dir / "history.csv", 64U << 20),
       checks = read_text(dir / "checks.jsonl", 512U << 20);
  bool interrupted = (!history.empty() && history.back() != '\n') ||
                     (!checks.empty() && checks.back() != '\n');
  std::size_t count = 0;
  std::istringstream lines(checks);
  std::string line;
  while (std::getline(lines, line)) {
    if (line.empty())
      continue;
    try {
      (void)parse_json(line);
      ++count;
    } catch (const std::exception &) {
      interrupted = true;
      break;
    }
  }
  std::istringstream rows(history);
  std::getline(rows, line);
  std::optional<std::uint64_t> last;
  while (std::getline(rows, line)) {
    if (line.empty())
      continue;
    std::size_t end = 0;
    auto step = std::stoull(line, &end);
    if (end >= line.size() || line[end] != ',' || (last && step != *last + 1))
      interrupted = true;
    last = step;
  }
  if (!result["durable_checkpoint"].is_null()) {
    const auto &cp = result["durable_checkpoint"];
    auto path = dir / "checkpoints" / cp.at("file").get<std::string>();
    if (sha256_file(path) != cp.at("sha256").get<std::string>())
      throw std::runtime_error("durable checkpoint hash mismatch");
  }
  for (const auto &entry : std::filesystem::recursive_directory_iterator(dir))
    if (entry.path().filename().string().find(".tmp.") != std::string::npos)
      interrupted = true;
  result["inspection"] = {
      {"interrupted_output", interrupted},
      {"diagnostic_records", count},
      {"last_history_step", last ? Json(*last) : Json(nullptr)}};
  return result;
}
} // namespace io
} // namespace fomhd
