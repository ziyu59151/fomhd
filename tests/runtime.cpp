#include "fomhd/io.hpp"
#include "support.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace fomhd;
void require(bool v, const char *s) {
  if (!v)
    throw std::runtime_error(s);
}
template <class F> void rejects(F f) {
  bool b = false;
  try {
    f();
  } catch (const std::exception &) {
    b = true;
  }
  require(b, "negative control accepted");
}
std::filesystem::path running_executable() {
  std::vector<char> buffer(4096);
  const auto size =
      ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  require(size > 0 && static_cast<std::size_t>(size) < buffer.size(),
          "cannot resolve running executable");
  buffer[static_cast<std::size_t>(size)] = '\0';
  return std::filesystem::path(buffer.data());
}
int main(int argc, char **argv) {
  try {
    std::string name = argc > 1 ? argv[1] : "state";
    if (name == "state") {
      auto p = test::parameters();
      Compute cpu;
      auto g = build_rectangle(4, 5, 1, 0, .2, .2);
      auto o = build_operators(g, p.K);
      Context c{p, g, o, cpu, p.Nphi};
      auto u = closed_reference(c, 1e12, 1e-9, 1e-9, 1e-5, .01);
      auto a = io::encode_state(u, g);
      auto v = io::decode_state(a, g, p.K);
      require(v.fluid[1].mom == u.fluid[1].mom && v.B == u.B,
              "state packing roundtrip");
      auto n = io::to_complex(a.at("n_i"));
      n[g.R.size() - 1] = 1;
      a["n_i"] = io::from_complex(n, a.at("n_i").shape);
      rejects([&] { (void)io::decode_state(a, g, p.K); });
    } else if (name == "json") {
      rejects([] { (void)io::parse_json("{\"a\":1,\"a\":2}"); });
      rejects([] { (void)io::parse_json("{\"a\":{\"b\":1,\"b\":2}}"); });
      require(io::parse_json("{\"a\":1,\"nested\":{\"a\":2}}")["a"] == 1,
              "valid duplicate across objects");
    } else if (name == "atomic") {
      auto root = std::filesystem::temp_directory_path() /
                  ("fomhd-atomic-" + std::to_string(getpid()));
      std::filesystem::create_directory(root);
      auto path = root / "result";
      rejects([&] {
        (void)io::atomic_publish(
            path, [](auto p) { io::write_text(p, "new"); }, false,
            [](auto phase) {
              if (phase == "written")
                throw std::runtime_error("injected");
            });
      });
      require(!std::filesystem::exists(path),
              "prepublication failure leaked output");
      bool retained = false;
      for (auto f : std::filesystem::directory_iterator(root))
        retained |=
            f.path().filename().string().find(".tmp.") != std::string::npos;
      require(retained, "failure temp discarded");
      io::atomic_publish(path, [](auto p) { io::write_text(p, "old"); });
      rejects([&] {
        io::atomic_publish(path, [](auto p) { io::write_text(p, "new"); });
      });
      require(io::read_text(path) == "old",
              "no-clobber failure changed target");
      bool after = false;
      try {
        io::atomic_publish(
            path, [](auto p) { io::write_text(p, "new"); }, true,
            [](auto phase) {
              if (phase == "renamed")
                throw std::runtime_error("injected");
            });
      } catch (const io::PublicationError &e) {
        after = e.published;
      }
      require(after && io::read_text(path) == "new",
              "published failure falsely claimed rollback");
      std::filesystem::remove_all(root);
    } else if (name == "checkpoint_reuse") {
      const auto executable = running_executable();
      const auto build = executable.parent_path();
      auto root = build / "testing" /
                  ("checkpoint-reuse-" + std::to_string(getpid()));
      std::filesystem::remove_all(root);
      std::filesystem::create_directories(root / "inputs");

      auto manifest = io::read_json(build / "build_manifest.json");
      manifest["binary_path"] = executable.string();
      manifest["binary_sha256"] = io::sha256_running_executable();
      const auto manifest_path = root / "build_manifest.json";
      io::write_text(manifest_path, manifest.dump(2) + "\n");

      auto parameters = test::parameters();
      Compute compute;
      auto case_config = io::Case{};
      case_config.parameters = parameters;
      case_config.geometry = build_rectangle(4, 4, 1, 0, .2, .2);
      case_config.operators =
          build_operators(case_config.geometry, parameters.K);
      case_config.limits = {};
      case_config.case_path = root / "inputs/case.toml";
      case_config.geometry_path = root / "inputs/geometry.npz";
      case_config.initial_path = root / "inputs/initial.npz";
      case_config.source_path = root / "inputs/source_manifest.json";
      case_config.output = root / "run";
      case_config.name = "checkpoint-reuse";
      case_config.backend = "cpu";
      case_config.initial_kind = "non_equilibrium";
      case_config.raw = "schema_version = 1\n";
      case_config.config_hash = io::sha256("checkpoint config");
      case_config.semantic_hash = io::sha256("checkpoint semantic");
      case_config.geometry_hash = io::sha256("checkpoint geometry");
      case_config.operator_hash = io::sha256("checkpoint operators");
      case_config.source = {{"provider", "runtime.cpp"}};
      case_config.resolved = {
          {"diagnostics", {{"mode_observations", io::Json::array()}}}};

      Context c{case_config.parameters, case_config.geometry,
                case_config.operators, compute, parameters.Nphi};
      auto initial =
          closed_reference(c, parameters.ref.ni, parameters.ref.pressure,
                           parameters.ref.pressure, parameters.ref.magnetic);
      io::write_npz(case_config.initial_path,
                    io::encode_state(initial, case_config.geometry));
      std::vector<double> lcfs_r, lcfs_z;
      for (const auto &point : case_config.geometry.polygon_input) {
        lcfs_r.push_back(point.R);
        lcfs_z.push_back(point.Z);
      }
      io::write_npz(
          case_config.geometry_path,
          {{"R_nodes",
            io::from_doubles(case_config.geometry.R,
                             {case_config.geometry.R.size()})},
           {"Z_nodes",
            io::from_doubles(case_config.geometry.Z,
                             {case_config.geometry.Z.size()})},
           {"lcfs_R", io::from_doubles(lcfs_r, {lcfs_r.size()})},
           {"lcfs_Z", io::from_doubles(lcfs_z, {lcfs_z.size()})}});
      io::write_text(case_config.source_path,
                     case_config.source.dump(2) + "\n");
      case_config.input_hashes = {
          {"geometry", io::sha256_file(case_config.geometry_path)},
          {"initial", io::sha256_file(case_config.initial_path)},
          {"source", io::sha256_file(case_config.source_path)}};

      Cumulative cumulative;
      auto committed = initialize(initial, c, cumulative, false);
      {
        io::Run run(case_config, manifest_path);
        run.checkpoint(committed, cumulative, c);

        const auto accepted = committed.accepted;
        committed.accepted = false;
        rejects([&] { run.checkpoint(committed, cumulative, c); });
        committed.accepted = accepted;

        const auto generation = committed.cache.tag.generation;
        ++committed.cache.tag.generation;
        rejects([&] { run.checkpoint(committed, cumulative, c); });
        committed.cache.tag.generation = generation;

        const auto pointer = case_config.output / "checkpoints/latest.json";
        const auto saved = pointer.string() + ".review-saved";
        std::filesystem::rename(pointer, saved);
        rejects([&] { run.checkpoint(committed, cumulative, c); });
        require(!std::filesystem::exists(pointer),
                "checkpoint reuse recreated a missing durable pointer");
        std::filesystem::rename(saved, pointer);

        const auto pointer_text = io::read_text(pointer);
        auto mismatched_pointer = io::parse_json(pointer_text);
        mismatched_pointer["step"] =
            mismatched_pointer["step"].get<std::uint64_t>() + 1;
        io::write_text(pointer, mismatched_pointer.dump(2) + "\n");
        rejects([&] { run.checkpoint(committed, cumulative, c); });
        io::write_text(pointer, pointer_text);

        const auto durable = io::parse_json(pointer_text);
        const auto checkpoint_file =
            case_config.output / "checkpoints" /
            durable.at("file").get<std::string>();
        const auto checkpoint_saved = checkpoint_file.string() + ".review-saved";
        std::filesystem::rename(checkpoint_file, checkpoint_saved);
        io::write_text(checkpoint_file, "corrupt checkpoint");
        rejects([&] { run.checkpoint(committed, cumulative, c); });
        std::filesystem::remove(checkpoint_file);
        std::filesystem::rename(checkpoint_saved, checkpoint_file);
      }
      std::filesystem::remove_all(root);
    } else
      throw std::runtime_error("unknown test");
    std::cout << "PASS " << name << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
