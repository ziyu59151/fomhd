#include "fomhd/io.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <toml.hpp>
namespace fomhd::io {
namespace {
void keys(const Json &object, std::initializer_list<const char *> expected,
          const std::string &name) {
  if (!object.is_object())
    throw std::runtime_error(name + " must be a table");
  std::set<std::string> wanted;
  for (auto s : expected)
    wanted.insert(s);
  for (auto i = object.begin(); i != object.end(); ++i)
    if (!wanted.contains(i.key()))
      throw std::runtime_error("unknown key " + name + "." + i.key());
  for (auto s : wanted)
    if (!object.contains(s))
      throw std::runtime_error("missing key " + name + "." + s);
}
double real(const Json &j, const char *key, bool positive = false) {
  const auto &x = j.at(key);
  if (!x.is_number_float())
    throw std::runtime_error(std::string(key) + " must be a TOML float");
  const double v = x.get<double>();
  if (!std::isfinite(v) || (positive && v <= 0))
    throw std::runtime_error(std::string(key) +
                             " is outside its finite domain");
  return v;
}
std::uint64_t integer(const Json &j, const char *key, bool positive = false) {
  const auto &x = j.at(key);
  if (!x.is_number_integer() || x.get<double>() < 0)
    throw std::runtime_error(std::string(key) +
                             " must be a nonnegative integer");
  const auto n = x.get<std::uint64_t>();
  if (positive && n == 0)
    throw std::runtime_error(std::string(key) + " must be positive");
  return n;
}
std::string str(const Json &j, const char *key) {
  return j.at(key).get<std::string>();
}
bool safe_name(const std::string &s) {
  return !s.empty() && s.size() <= 128 &&
         std::all_of(s.begin(), s.end(), [](unsigned char c) {
           return std::isalnum(c) || c == '-' || c == '_';
         });
}
std::map<std::string, Budget> registry(const Parameters &p) {
  std::map<std::string, Budget> specs;
  auto arr = [&](std::string key, std::string unit, bool hard = true) {
    specs[key + ".linf"] = {0, 0, 0, unit, Norm::linf, hard};
    specs[key + ".weighted_L2"] = {0, 0, 0, unit, Norm::weighted_l2, hard};
  };
  auto val = [&](std::string key, std::string unit, bool hard = true) {
    specs[key + ".value"] = {0, 0, 0, unit, Norm::value, hard};
  };
  for (auto key : {"poisson_linear", "gauss", "transverse_divergence"})
    arr(key, "V/m^2");
  arr("darwin_dynamic", "A/(m^2*s)");
  arr("longitudinal_projection", "A/m^3");
  for (auto key : {"initial_divB", "initial_ampere", "divB_update",
                   "long_term_divB_change"})
    arr(key, "T/m");
  for (auto key : {"current_divB", "current_ampere"})
    arr(key, "T/m", false);
  arr("CT_increment", "T/(m*s)");
  arr("current_response", "A/(m^2*s)");
  for (auto key : {"CT_update", "equilibrium_B", "alias_B"})
    arr(key, "T/s");
  for (auto key : {"RC_flux", "RC_mom", "deltaC"})
    arr(key, "C/(m^3*s)");
  arr("RL", "A/m^2");
  val("topology_increment", "Wb");
  val("total_charge_balance", "C");
  for (auto key : {"ion_internal", "electron_perp", "electron_parallel",
                   "field_energy_B", "field_energy_L", "total_energy"})
    val(key, "J", false);
  val("field_exchange", "W", false);
  for (auto key : {"momentum_x", "momentum_y", "momentum_z"})
    val(key, "kg*m/s", false);
  val("angular_momentum", "kg*m^2/s", false);
  for (int a = 0; a < 2; ++a) {
    const auto id = "." + p.species[a].id;
    for (auto key :
         {"stage_continuity", "species_continuity", "equilibrium_n", "alias_n"})
      arr(key + id, "1/(m^3*s)");
    for (auto key :
         {"wall_rhs", "wall_multiplier", "equilibrium_mom", "alias_mom"})
      arr(key + id, "kg/(m^2*s^2)");
    arr("wall_state" + id, "kg/(m^2*s)");
    arr("theta_identity" + id, "1/s");
    arr("wall_number_flux" + id, "1/(m^2*s)");
    arr("wall_perp_flux" + id, "Pa*m/s");
    if (a)
      arr("wall_parallel_flux" + id, "Pa*m/s");
    for (auto key : {"equilibrium_perp", "alias_perp"})
      arr(key + id, "Pa/s");
    if (a)
      for (auto key : {"cgl_source", "equilibrium_parallel", "alias_parallel"})
        arr(key + id, "Pa/s");
    for (auto key : {"electric_pairing", "pressure_pairing", "magnetic_work",
                     "wall_work", "convection_pairing"})
      val(key + id, "W");
    val("spectral_transfer_defect" + id, "W", false);
    val("total_number_balance" + id, "1");
    val("kinetic_energy" + id, "J", false);
  }
  return specs;
}
std::string matrix_fingerprint(const Operators &o) {
  std::string values = "A0-compatible-operators-v2\n";
  auto matrix = [&](const Matrix &m) {
    values += std::to_string(m.rows) + "," + std::to_string(m.cols) + ":" +
              sha256(from_complex(m.data, {m.rows, m.cols}).data) + "\n";
  };
  for (auto group : {&o.G, &o.C, &o.DB, &o.DJ, &o.Cdag, &o.Lphi})
    for (const auto &m : *group)
      matrix(m);
  for (auto m : {&o.I, &o.H, &o.QB, &o.T_full, &o.T, &o.Bc_full, &o.Bc, &o.DR,
                 &o.DZ, &o.topology_flux_row, &o.PR_r, &o.PZ_r, &o.PR_z,
                 &o.PZ_z, &o.PR_v, &o.PZ_v})
    matrix(*m);
  for (auto w : {&o.M0, &o.M1, &o.MB})
    values += sha256(from_doubles(*w, {w->size()}).data) + "\n";
  for (auto rows : {&o.selected_wall_rows, &o.selected_constraint_rows})
    for (auto row : *rows)
      values += std::to_string(row) + ",";
  return sha256(values);
}
Array scatter(const Vec &v, const EntityTable &e, std::size_t M, std::size_t ny,
              std::size_t nx, std::size_t components = 1) {
  const auto n = e.keys.size(), plane = checked_product(nx, ny, "disk plane");
  if (v.size() != M * components * n)
    throw std::runtime_error("compact state extent mismatch");
  Vec out(checked_product(checked_product(M, components, "disk components"),
                          plane, "disk state"));
  for (std::size_t c = 0; c < components; ++c)
    for (std::size_t m = 0; m < M; ++m)
      for (std::size_t j = 0; j < n; ++j)
        out[(c * M + m) * plane + e.keys[j].second * nx + e.keys[j].first] =
            v[(m * components + c) * n + j];
  auto shape = std::vector<std::size_t>{M, ny, nx};
  if (components != 1)
    shape.insert(shape.begin(), components);
  return from_complex(out, shape);
}
Vec gather(const Array &a, const EntityTable &e, std::size_t M, std::size_t ny,
           std::size_t nx, std::size_t components = 1) {
  auto shape = std::vector<std::size_t>{M, ny, nx};
  if (components != 1)
    shape.insert(shape.begin(), components);
  if (a.shape != shape)
    throw std::runtime_error("state array shape mismatch");
  auto data = to_complex(a);
  if (!is_finite(data))
    throw std::runtime_error("state contains nonfinite values");
  const auto plane = nx * ny, n = e.keys.size();
  Vec v(M * components * n);
  std::vector<bool> active(plane, false);
  for (auto k : e.keys)
    active[k.second * nx + k.first] = true;
  for (std::size_t c = 0; c < components; ++c)
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t j = 0; j < plane; ++j)
        if (!active[j] && data[(c * M + m) * plane + j] != Complex{})
          throw std::runtime_error("nonzero inactive slot in state");
      for (std::size_t j = 0; j < n; ++j)
        v[(m * components + c) * n + j] =
            data[(c * M + m) * plane + e.keys[j].second * nx + e.keys[j].first];
    }
  return v;
}
} // namespace
std::string read_text(const std::filesystem::path &p, std::size_t cap) {
  if (!std::filesystem::is_regular_file(p) ||
      std::filesystem::file_size(p) > cap)
    throw std::runtime_error("input is not a bounded regular file: " +
                             p.string());
  std::ifstream f(p, std::ios::binary);
  std::string s((std::istreambuf_iterator<char>(f)), {});
  if (f.bad() || s.size() > cap)
    throw std::runtime_error("cannot read bounded input: " + p.string());
  return s;
}
void write_text(const std::filesystem::path &p, const std::string &s) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(s.data(), s.size());
  f.close();
  if (!f)
    throw std::runtime_error("cannot write " + p.string());
}
void require_finite_json(const Json &j) {
  if (j.is_number_float() && !std::isfinite(j.get<double>()))
    throw std::runtime_error("nonfinite JSON number");
  if (j.is_structured())
    for (const auto &x : j)
      require_finite_json(x);
}
Json parse_json(const std::string &text) {
  std::vector<std::set<std::string>> stack;
  auto result = Json::parse(text, [&](int, Json::parse_event_t e, Json &v) {
    if (e == Json::parse_event_t::object_start)
      stack.emplace_back();
    if (e == Json::parse_event_t::key &&
        !stack.back().insert(v.get<std::string>()).second)
      throw std::runtime_error("duplicate JSON key");
    if (e == Json::parse_event_t::object_end)
      stack.pop_back();
    return true;
  });
  require_finite_json(result);
  return result;
}
Json read_json(const std::filesystem::path &p) {
  return parse_json(read_text(p));
}
Case load_case(const std::filesystem::path &path, Overrides overrides) {
  Case c;
  c.case_path = std::filesystem::absolute(path).lexically_normal();
  c.raw = read_text(path);
  Json j;
  try {
    std::ostringstream s;
    s << toml::json_formatter(
        toml::parse(c.raw, std::string_view(path.string())));
    j = parse_json(s.str());
  } catch (const std::exception &e) {
    throw std::runtime_error(std::string("TOML input: ") + e.what());
  }
  keys(j,
       {"schema_version", "model", "units", "layout_version", "case_name",
        "species", "grid", "time", "boundary", "solver", "budgets", "io",
        "diagnostics"},
       "case");
  if (integer(j, "schema_version") != 1 || integer(j, "layout_version") != 1 ||
      str(j, "model") != "A0" || str(j, "units") != "SI")
    throw std::runtime_error("expected A0 schema/layout 1; units must be SI");
  c.name = str(j, "case_name");
  if (!safe_name(c.name))
    throw std::runtime_error("invalid case_name");
  auto &p = c.parameters;
  auto &s = j["species"];
  keys(s, {"ion_id", "electron_id", "qi", "mi", "qe", "me", "gamma_i"},
       "species");
  p.species[0] = {str(s, "ion_id"), real(s, "mi", true), real(s, "qi", true),
                  real(s, "gamma_i", true)};
  p.species[1] = {str(s, "electron_id"), real(s, "me", true), real(s, "qe"), 0};
  if (!safe_name(p.species[0].id) || !safe_name(p.species[1].id) ||
      p.species[0].id == p.species[1].id || p.species[1].charge >= 0 ||
      p.species[0].gamma <= 1)
    throw std::runtime_error("invalid species identifiers or constants");
  auto &solver = j["solver"];
  keys(solver,
       {"backend", "precision", "complex_precision", "memory_limit_bytes",
        "device_memory_limit_bytes", "rank_rtol", "device_ordinal",
        "cpu_threads", "constants", "references"},
       "solver");
  c.backend = str(solver, "backend");
  if ((c.backend != "cpu" && c.backend != "cuda") ||
      str(solver, "precision") != "fp64" ||
      str(solver, "complex_precision") != "complex128" ||
      integer(solver, "cpu_threads", true) != 1)
    throw std::runtime_error("unsupported backend/precision/thread policy");
  c.device = static_cast<int>(integer(solver, "device_ordinal"));
  if (c.device < 0 || (c.backend == "cpu" && c.device != 0))
    throw std::runtime_error("invalid device ordinal");
  p.memory_limit = integer(solver, "memory_limit_bytes", true);
  p.device_memory_limit = integer(solver, "device_memory_limit_bytes", true);
  p.rank_rtol = real(solver, "rank_rtol", true);
  if (p.rank_rtol >= 1)
    throw std::runtime_error("rank_rtol must be below one");
  auto &constants = solver["constants"];
  keys(constants, {"epsilon0_Fpm", "mu0_Hpm", "version"}, "constants");
  p.epsilon0 = real(constants, "epsilon0_Fpm", true);
  p.mu0 = real(constants, "mu0_Hpm", true);
  if (str(constants, "version").empty())
    throw std::runtime_error("constants version is required");
  auto &r = solver["references"];
  keys(r,
       {"L_m", "t_s", "n_i_m3", "n_e_m3", "u_mps", "E_Vpm", "B_T", "J_Apm2",
        "rho_Cpm3", "p_Pa"},
       "references");
  p.ref = {real(r, "L_m", true),      real(r, "t_s", true),
           real(r, "n_i_m3", true),   real(r, "n_e_m3", true),
           real(r, "u_mps", true),    real(r, "E_Vpm", true),
           real(r, "B_T", true),      real(r, "J_Apm2", true),
           real(r, "rho_Cpm3", true), real(r, "p_Pa", true)};
  auto &t = j["time"];
  keys(t, {"start_time", "dt", "nsteps", "initial_kind"}, "time");
  if (overrides.nsteps)
    t["nsteps"] = *overrides.nsteps;
  p.dt = real(t, "dt", true);
  p.start_time = real(t, "start_time");
  p.nsteps = integer(t, "nsteps");
  if (p.nsteps > (1ULL << 52))
    throw std::runtime_error("too many time steps");
  c.initial_kind = str(t, "initial_kind");
  if (c.initial_kind != "equilibrium" && c.initial_kind != "non_equilibrium")
    throw std::runtime_error("unknown initial kind");
  (void)time_at_step(p, p.nsteps);
  if (p.nsteps)
    (void)time_at_midpoint(p, p.nsteps - 1);
  const auto &b = j["boundary"];
  keys(b, {"kind", "potential", "version"}, "boundary");
  if (str(b, "kind") != "fixed_lcfs_static" || real(b, "potential") != 0 ||
      str(b, "version") != "staircase_relative_v1")
    throw std::runtime_error("unsupported boundary");
  auto &grid = j["grid"];
  keys(grid, {"R0", "Z0", "hR", "hZ", "NR", "NZ", "K", "Nphi"}, "grid");
  auto nr = integer(grid, "NR", true), nz = integer(grid, "NZ", true),
       K = integer(grid, "K"), N = integer(grid, "Nphi", true);
  if (nr < 2 || nz < 2 || K > 1000 || N > 100000 || N % 2 != 1 || N < 4 * K + 1)
    throw std::runtime_error("invalid grid or Fourier resolution");
  p.K = static_cast<int>(K);
  p.Nphi = static_cast<int>(N);
  auto &bounds = j["budgets"];
  keys(bounds, {"domain", "checks"}, "budgets");
  auto &d = bounds["domain"];
  keys(d,
       {"n_min_i", "n_min_e", "n_face_min_i", "n_face_min_e", "p_min_i",
        "p_min_perp_e", "p_min_parallel_e", "B_min"},
       "domain");
  for (auto x = d.begin(); x != d.end(); ++x)
    if (real(d, x.key().c_str()) < 0)
      throw std::runtime_error("negative domain floor");
  p.species[0].n_min = real(d, "n_min_i");
  p.species[1].n_min = real(d, "n_min_e");
  p.species[0].face_min = real(d, "n_face_min_i");
  p.species[1].face_min = real(d, "n_face_min_e");
  p.species[0].p_min = real(d, "p_min_i");
  p.species[1].p_min = real(d, "p_min_perp_e");
  p.species[1].parallel_min = real(d, "p_min_parallel_e");
  p.B_min = real(d, "B_min");
  const auto specs = registry(p);
  auto &checks = bounds["checks"];
  if (!checks.is_object())
    throw std::runtime_error("checks must be a table");
  for (auto x = checks.begin(); x != checks.end(); ++x)
    if (!specs.contains(x.key()))
      throw std::runtime_error("unknown budget " + x.key());
  for (const auto &[name, spec] : specs) {
    if (!checks.contains(name))
      throw std::runtime_error("missing budget " + name);
    const auto &x = checks[name];
    keys(x, {"atol", "rtol", "reference", "unit", "norm", "gate"}, name);
    Budget budget = spec;
    budget.atol = real(x, "atol");
    budget.rtol = real(x, "rtol");
    budget.reference = real(x, "reference", true);
    auto norm = str(x, "norm"), gate = str(x, "gate");
    if (budget.atol < 0 || budget.rtol < 0 || str(x, "unit") != spec.unit ||
        !name.ends_with("." + norm) || (gate != "hard" && gate != "observe") ||
        (spec.hard && gate != "hard"))
      throw std::runtime_error("invalid budget unit/norm/gate: " + name);
    budget.hard = gate == "hard";
    p.budgets[name] = budget;
  }
  auto &diag = j["diagnostics"];
  keys(diag,
       {"inner_distance_m", "alias_policy", "timing_repetitions",
        "timing_warmup", "mode_observations"},
       "diagnostics");
  p.inner_distance = real(diag, "inner_distance_m", true);
  if (str(diag, "alias_policy") != "initial_midpoint_restart")
    throw std::runtime_error("unsupported alias policy");
  (void)integer(diag, "timing_repetitions", true);
  (void)integer(diag, "timing_warmup");
  if (!diag["mode_observations"].is_array())
    throw std::runtime_error("mode_observations must be an array");
  for (const auto &ob : diag["mode_observations"]) {
    keys(ob, {"field", "component", "mode", "entity_key"}, "observation");
    if (!ob["mode"].is_number_integer() ||
        std::abs(ob["mode"].get<int>()) > p.K || !ob["entity_key"].is_array() ||
        ob["entity_key"].size() != 2)
      throw std::runtime_error("invalid mode observation");
  }
  auto &paths = j["io"];
  keys(paths,
       {"geometry_file", "initial_file", "output_dir", "snapshot_stride",
        "checkpoint_stride", "max_input_bytes", "max_uncompressed_bytes",
        "max_entries", "max_header_bytes", "source_manifest_file",
        "write_residual_arrays"},
       "io");
  const auto resolve = [&](const char *key) {
    return std::filesystem::absolute(c.case_path.parent_path() /
                                     str(paths, key))
        .lexically_normal();
  };
  if (overrides.output)
    paths["output_dir"] = std::filesystem::absolute(*overrides.output).string();
  c.output = resolve("output_dir");
  c.geometry_path = resolve("geometry_file");
  c.initial_path = resolve("initial_file");
  c.source_path = resolve("source_manifest_file");
  c.snapshot_stride = integer(paths, "snapshot_stride", true);
  c.checkpoint_stride = integer(paths, "checkpoint_stride", true);
  c.residual_arrays = paths["write_residual_arrays"].get<bool>();
  c.limits = {integer(paths, "max_input_bytes", true),
              integer(paths, "max_uncompressed_bytes", true),
              integer(paths, "max_entries", true),
              integer(paths, "max_header_bytes", true)};
  if (c.limits.max_expanded_bytes > p.memory_limit / 4)
    throw std::runtime_error(
        "input expanded cap exceeds one quarter of the host memory budget");
  c.source = read_json(c.source_path);
  for (auto key :
       {"provider_name", "provider_version", "units", "conversion_method",
        "input_hashes", "continuous_residual_status"})
    if (!c.source.contains(key))
      throw std::runtime_error(std::string("source manifest missing ") + key);
  if (c.source["units"] != "SI")
    throw std::runtime_error("source manifest units must be SI");
  c.input_hashes = {{"case", sha256(c.raw)},
                    {"geometry", sha256_file(c.geometry_path)},
                    {"initial", sha256_file(c.initial_path)},
                    {"source", sha256_file(c.source_path)}};
  for (auto i = c.source["input_hashes"].begin();
       i != c.source["input_hashes"].end(); ++i)
    if (!c.input_hashes.contains(i.key()) ||
        i.value() != c.input_hashes[i.key()])
      throw std::runtime_error("source input hash mismatch");
  auto geometry = read_npz(c.geometry_path, c.limits,
                           Keys{"R_nodes", "Z_nodes", "lcfs_R", "lcfs_Z"});
  if (geometry.size() != 4)
    throw std::runtime_error("missing geometry array");
  auto R = to_doubles(geometry.at("R_nodes")),
       Z = to_doubles(geometry.at("Z_nodes")),
       lr = to_doubles(geometry.at("lcfs_R")),
       lz = to_doubles(geometry.at("lcfs_Z"));
  if (R.size() != nr || Z.size() != nz || lr.size() != lz.size())
    throw std::runtime_error("geometry dimensions disagree with configuration");
  for (auto [name, values] :
       std::initializer_list<std::pair<const char *, const RealVec *>>{
           {"R_nodes", &R}, {"Z_nodes", &Z}, {"lcfs_R", &lr}, {"lcfs_Z", &lz}})
    if (geometry.at(name).shape != std::vector<std::size_t>{values->size()})
      throw std::runtime_error("geometry array must be one dimensional");
  const double r0 = real(grid, "R0"), z0 = real(grid, "Z0"),
               hr = real(grid, "hR", true), hz = real(grid, "hZ", true);
  for (std::size_t i = 0; i < nr; ++i)
    if (std::abs(R[i] - std::fma(double(i), hr, r0)) >
        64 * std::numeric_limits<double>::epsilon() *
            std::max(std::abs(R[i]), hr))
      throw std::runtime_error("R grid configuration mismatch");
  for (std::size_t i = 0; i < nz; ++i)
    if (std::abs(Z[i] - std::fma(double(i), hz, z0)) >
        64 * std::numeric_limits<double>::epsilon() *
            std::max({std::abs(Z[i]), std::abs(z0), hz}))
      throw std::runtime_error("Z grid configuration mismatch");
  std::vector<Point2> polygon;
  for (std::size_t i = 0; i < lr.size(); ++i)
    polygon.push_back({lr[i], lz[i]});
  c.geometry = build_geometry(R, Z, polygon, {p.memory_limit, 1e-12});
  c.operators = build_operators(c.geometry, p.K, {p.rank_rtol, p.memory_limit});
  c.geometry_hash = sha256(c.input_hashes["geometry"].get<std::string>() +
                           "half_open_ray_certified_v2");
  c.operator_hash = matrix_fingerprint(c.operators);
  c.resolved = j;
  c.resolved["resolved_paths"] = {{"geometry", c.geometry_path.string()},
                                  {"initial", c.initial_path.string()},
                                  {"source", c.source_path.string()},
                                  {"output", c.output.string()}};
  c.resolved["input_hashes"] = c.input_hashes;
  c.config_hash = sha256(c.resolved.dump());
  auto semantic = j;
  semantic["time"].erase("nsteps");
  for (auto key : {"output_dir", "snapshot_stride", "checkpoint_stride",
                   "geometry_file", "initial_file", "source_manifest_file"})
    semantic["io"].erase(key);
  for (auto key : {"timing_repetitions", "timing_warmup"})
    semantic["diagnostics"].erase(key);
  semantic["input_hashes"] = c.input_hashes;
  semantic["input_hashes"].erase("case");
  semantic["operator_hash"] = c.operator_hash;
  semantic["geometry_hash"] = c.geometry_hash;
  c.semantic_hash = sha256(semantic.dump());
  return c;
}
Arrays encode_state(const State &u, const Geometry &g) {
  auto nc = g.cells.keys.size(), M = u.fluid[0].n.size() / nc, ny = g.Z.size(),
       nx = g.R.size();
  Arrays out;
  for (int a = 0; a < 2; ++a) {
    out[a ? "n_e" : "n_i"] = scatter(u.fluid[a].n, g.cells, M, ny, nx);
    out[a ? "mom_e" : "mom_i"] = scatter(u.fluid[a].mom, g.cells, M, ny, nx, 3);
    out[a ? "p_perp_e" : "p_i"] = scatter(u.fluid[a].perp, g.cells, M, ny, nx);
    if (a)
      out["p_parallel_e"] = scatter(u.fluid[a].parallel, g.cells, M, ny, nx);
  }
  const auto nz = g.z_faces.keys.size(), nv = g.vertices.keys.size(),
             nr = g.r_faces.keys.size(), nb = nz + nv + nr;
  Vec r(M * nz), phi(M * nv), z(M * nr);
  for (std::size_t k = 0; k < M; ++k) {
    put(r, k * nz, slice(u.B, k * nb, nz));
    put(phi, k * nv, slice(u.B, k * nb + nz, nv));
    put(z, k * nr, slice(u.B, k * nb + nz + nv, nr));
  }
  out["B_R"] = scatter(r, g.z_faces, M, ny + 1, nx);
  out["B_phi"] = scatter(phi, g.vertices, M, ny + 1, nx + 1);
  out["B_Z"] = scatter(z, g.r_faces, M, ny, nx + 1);
  return out;
}
State decode_state(const Arrays &arrays, const Geometry &g, int K) {
  const auto M = static_cast<std::size_t>(2 * K + 1), ny = g.Z.size(),
             nx = g.R.size();
  State u;
  u.generation = 1;
  for (int a = 0; a < 2; ++a) {
    u.fluid[a].n = gather(arrays.at(a ? "n_e" : "n_i"), g.cells, M, ny, nx);
    u.fluid[a].mom =
        gather(arrays.at(a ? "mom_e" : "mom_i"), g.cells, M, ny, nx, 3);
    u.fluid[a].perp =
        gather(arrays.at(a ? "p_perp_e" : "p_i"), g.cells, M, ny, nx);
    if (a)
      u.fluid[a].parallel =
          gather(arrays.at("p_parallel_e"), g.cells, M, ny, nx);
  }
  auto r = gather(arrays.at("B_R"), g.z_faces, M, ny + 1, nx),
       phi = gather(arrays.at("B_phi"), g.vertices, M, ny + 1, nx + 1),
       z = gather(arrays.at("B_Z"), g.r_faces, M, ny, nx + 1);
  const auto nz = g.z_faces.keys.size(), nv = g.vertices.keys.size(),
             nr = g.r_faces.keys.size(), nb = nz + nv + nr;
  u.B.resize(M * nb);
  for (std::size_t k = 0; k < M; ++k) {
    put(u.B, k * nb, slice(r, k * nz, nz));
    put(u.B, k * nb + nz, slice(phi, k * nv, nv));
    put(u.B, k * nb + nz + nv, slice(z, k * nr, nr));
  }
  return u;
}
State load_initial(const Case &c) {
  auto a = read_npz(c.initial_path, c.limits,
                    Keys{"n_i", "mom_i", "p_i", "n_e", "mom_e", "p_perp_e",
                         "p_parallel_e", "B_R", "B_phi", "B_Z"});
  if (a.size() != 10)
    throw std::runtime_error("initial state must have exactly ten arrays");
  auto u = decode_state(a, c.geometry, c.parameters.K);
  u.time = c.parameters.start_time;
  return u;
}
Array encode_vertex(const Vec &x, const Geometry &g, int K) {
  return scatter(x, g.vertices, 2 * K + 1, g.Z.size() + 1, g.R.size() + 1);
}
Vec decode_vertex(const Array &a, const Geometry &g, int K) {
  return gather(a, g.vertices, 2 * K + 1, g.Z.size() + 1, g.R.size() + 1);
}
} // namespace fomhd::io
