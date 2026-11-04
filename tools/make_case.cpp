#include "fomhd/backend.hpp"
#include "fomhd/io.hpp"
#include "support.hpp"
#include <iostream>
#include <toml.hpp>
namespace {
toml::table table(const nlohmann::json &j) {
  toml::table t;
  for (auto i = j.begin(); i != j.end(); ++i) {
    const auto &v = i.value();
    if (v.is_object())
      t.insert(i.key(), table(v));
    else if (v.is_string())
      t.insert(i.key(), v.get<std::string>());
    else if (v.is_boolean())
      t.insert(i.key(), v.get<bool>());
    else if (v.is_number_float())
      t.insert(i.key(), v.get<double>());
    else if (v.is_number_unsigned())
      t.insert(i.key(), static_cast<std::int64_t>(v.get<std::uint64_t>()));
    else if (v.is_number_integer())
      t.insert(i.key(), v.get<std::int64_t>());
    else if (v.is_array() && v.empty())
      t.insert(i.key(), toml::array{});
    else
      throw std::runtime_error("unsupported fixture config value");
  }
  return t;
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 6)
      throw std::runtime_error("Usage: fomhd_make_case DIR [STEPS=4] "
                               "[VELOCITY=0.01] [DT=1e-9] [K=1]");
    const auto root = std::filesystem::absolute(argv[1]);
    if (std::filesystem::exists(root))
      throw std::runtime_error("fixture directory already exists");
    std::filesystem::create_directories(root);
    auto p = fomhd::test::parameters(argc > 5 ? std::stoi(argv[5]) : 1);
    if (argc > 2)
      p.nsteps = std::stoull(argv[2]);
    if (argc > 4)
      p.dt = std::stod(argv[4]);
    double velocity = argc > 3 ? std::stod(argv[3]) : .01;
    auto g = fomhd::build_rectangle(4, 4, 1, 0, .2, .2);
    auto op = fomhd::build_operators(g, p.K, {p.rank_rtol, p.memory_limit});
    fomhd::Compute cpu;
    fomhd::Context c{p, g, op, cpu, p.Nphi};
    auto state = fomhd::closed_reference(
        c, p.ref.ni, p.ref.pressure, p.ref.pressure, p.ref.magnetic, velocity);
    fomhd::io::write_npz(root / "initial.npz",
                         fomhd::io::encode_state(state, g));
    std::vector<double> r, z;
    for (auto point : g.polygon_input) {
      r.push_back(point.R);
      z.push_back(point.Z);
    }
    fomhd::io::write_npz(
        root / "geometry.npz",
        {{"R_nodes", fomhd::io::from_doubles(g.R, {g.R.size()})},
         {"Z_nodes", fomhd::io::from_doubles(g.Z, {g.Z.size()})},
         {"lcfs_R", fomhd::io::from_doubles(r, {r.size()})},
         {"lcfs_Z", fomhd::io::from_doubles(z, {z.size()})}});
    using J = nlohmann::json;
    J source = {
        {"provider_name", "fomhd_closed_reference"},
        {"provider_version", "v2"},
        {"units", "SI"},
        {"conversion_method", "Discrete harmonic magnetic field and neutral "
                              "species; equal toroidal velocity perturbation."},
        {"continuous_residual_status",
         "Discrete reference; not experimental validation."},
        {"input_hashes",
         {{"geometry", fomhd::io::sha256_file(root / "geometry.npz")},
          {"initial", fomhd::io::sha256_file(root / "initial.npz")}}}};
    fomhd::io::write_text(root / "source_manifest.json", source.dump(2) + "\n");
    J j = {{"schema_version", 1},
           {"layout_version", 1},
           {"model", "A0"},
           {"units", "SI"},
           {"case_name", velocity == 0 ? "a0-equilibrium" : "a0-evolution"}};
    j["species"] = {{"ion_id", "ion"},
                    {"electron_id", "electron"},
                    {"qi", p.species[0].charge},
                    {"mi", p.species[0].mass},
                    {"qe", p.species[1].charge},
                    {"me", p.species[1].mass},
                    {"gamma_i", p.species[0].gamma}};
    j["grid"] = {{"R0", 1.0}, {"Z0", 0.0}, {"hR", .2}, {"hZ", .2},
                 {"NR", 4},   {"NZ", 4},   {"K", p.K}, {"Nphi", p.Nphi}};
    j["time"] = {
        {"start_time", p.start_time},
        {"dt", p.dt},
        {"nsteps", p.nsteps},
        {"initial_kind", velocity == 0 ? "equilibrium" : "non_equilibrium"}};
    j["boundary"] = {{"kind", "fixed_lcfs_static"},
                     {"potential", 0.0},
                     {"version", "staircase_relative_v1"}};
    j["solver"] = {{"backend", "cpu"},
                   {"precision", "fp64"},
                   {"complex_precision", "complex128"},
                   {"memory_limit_bytes", p.memory_limit},
                   {"device_memory_limit_bytes", p.device_memory_limit},
                   {"rank_rtol", p.rank_rtol},
                   {"device_ordinal", 0},
                   {"cpu_threads", 1}};
    j["solver"]["constants"] = {{"epsilon0_Fpm", p.epsilon0},
                                {"mu0_Hpm", p.mu0},
                                {"version", "CODATA-2022"}};
    auto r0 = p.ref;
    j["solver"]["references"] = {
        {"L_m", r0.length},   {"t_s", r0.time},       {"n_i_m3", r0.ni},
        {"n_e_m3", r0.ne},    {"u_mps", r0.velocity}, {"E_Vpm", r0.electric},
        {"B_T", r0.magnetic}, {"J_Apm2", r0.current}, {"rho_Cpm3", r0.charge},
        {"p_Pa", r0.pressure}};
    j["budgets"]["domain"] = {{"n_min_i", 0.0},          {"n_min_e", 0.0},
                              {"n_face_min_i", 0.0},     {"n_face_min_e", 0.0},
                              {"p_min_i", 0.0},          {"p_min_perp_e", 0.0},
                              {"p_min_parallel_e", 0.0}, {"B_min", p.B_min}};
    for (const auto &[name, b] : p.budgets)
      j["budgets"]["checks"][name] = {
          {"atol", b.atol},
          {"rtol", b.rtol},
          {"reference", b.reference},
          {"unit", b.unit},
          {"norm", b.norm == fomhd::Norm::linf          ? "linf"
                   : b.norm == fomhd::Norm::weighted_l2 ? "weighted_L2"
                                                        : "value"},
          {"gate", b.hard ? "hard" : "observe"}};
    j["io"] = {{"geometry_file", "geometry.npz"},
               {"initial_file", "initial.npz"},
               {"source_manifest_file", "source_manifest.json"},
               {"output_dir", "run"},
               {"snapshot_stride", 1},
               {"checkpoint_stride", 1},
               {"max_input_bytes", 32U << 20},
               {"max_uncompressed_bytes", 64U << 20},
               {"max_entries", 512},
               {"max_header_bytes", 1U << 20},
               {"write_residual_arrays", true}};
    j["diagnostics"] = {{"inner_distance_m", p.inner_distance},
                        {"alias_policy", "initial_midpoint_restart"},
                        {"timing_repetitions", 1},
                        {"timing_warmup", 0},
                        {"mode_observations", J::array()}};
    std::ostringstream toml;
    toml << table(j);
    fomhd::io::write_text(root / "case.toml", toml.str());
    std::cout << root / "case.toml" << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
