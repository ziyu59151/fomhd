#include "fomhd/diagnostics.hpp"
#include "support.hpp"
#include <algorithm>
#include <iostream>
using namespace fomhd;
void require(bool b, const char *s) {
  if (!b)
    throw std::runtime_error(s);
}
int main(int argc, char **argv) {
  try {
    auto p = test::parameters();
    p.inner_distance = .21;
    Compute cpu;
    auto g = build_rectangle(6, 6, 1, 0, .2, .2);
    auto o = build_operators(g, 0);
    p.K = 0;
    p.Nphi = 1;
    Context c{p, g, o, cpu, 1};
    StageTag t{0, 1, 1, 0, 0, "initial"};
    Vec r(o.Nc);
    for (std::size_t j = 0; j < o.Nc; ++j)
      r[j] = g.cell_wall_distance[j] < p.inner_distance ? 2 : 1;
    std::vector<Check> records;
    measure(records, "gauss", r, o.M0, "V/m^2", t, c);
    for (auto region : {"all", "inner", "wall"}) {
      auto i = std::find_if(records.begin(), records.end(), [&](const auto &x) {
        return x.region == region && x.norm == Norm::linf;
      });
      require(i != records.end(), "regional diagnostic missing");
      require(i->raw == (std::string(region) == "inner" ? 1 : 2),
              "wrong regional mask");
    }
    records.clear();
    measure(records, "wall_state", Vec(o.T_full.rows),
            RealVec(o.T_full.rows, 1), "kg/(m^2*s)", t, c, "ion", true);
    require(records[1].value_unit == "kg/(m^2*s)",
            "dimensionless wall weights labeled as volume");
    auto local_geometry = g;
    local_geometry.walls.clear();
    auto local_operators = o;
    local_operators.T = Matrix(0, 3 * o.Nc);
    local_operators.T_full = Matrix(0, 3 * o.Nc);
    Context local{p, local_geometry, local_operators, cpu, 1};
    std::vector<Check> absent_wall;
    measure(absent_wall, "wall_state", {}, {}, "kg/(m^2*s)", t, local,
            "ion", true);
    require(absent_wall.size() == 6, "absent wall records were omitted");
    for (const auto &record : absent_wall)
      require(!record.available && !record.passed && !record.hard &&
                  record.selected == 0 && !record.reason.empty(),
              "absent wall reported a synthetic pass");
    for (const auto *context : {&c, &local}) {
      bool rejected_empty = false;
      try {
        measure(absent_wall, "gauss", {}, {}, "V/m^2", t, *context);
      } catch (const Failure &error) {
        rejected_empty = error.code == "DiagnosticShape";
      }
      require(rejected_empty, "empty physical residual was accepted");
    }
    bool rejected_wall = false;
    try {
      measure(absent_wall, "wall_state", {}, {}, "kg/(m^2*s)", t, c,
              "ion", true);
    } catch (const Failure &error) {
      rejected_wall = error.code == "DiagnosticShape";
    }
    require(rejected_wall, "missing closed-wall residual was accepted");
    auto expected = stage_requirements(c, t);
    auto complete = expected;
    for (auto &x : complete) {
      x.available = true;
      x.passed = true;
    }
    gate_records(complete, expected);
    auto omitted =
        std::find_if(complete.begin(), complete.end(), [](const auto &x) {
          return x.name == "stage_continuity" && x.population == "electron" &&
                 x.region == "inner";
        });
    require(omitted != complete.end(),
            "expected registry lacks electron inner continuity");
    complete.erase(omitted);
    bool rejected = false;
    try {
      gate_records(complete, expected);
    } catch (const Failure &) {
      rejected = true;
    }
    require(rejected, "gate accepted a missing population/region tuple");
    Vec tiny(o.Ne, Complex{0, 1e-30});
    require(max_abs(inverse(tiny, o.Ne, c, true, p.ref.electric)) == 0,
            "field roundoff was not checked before taking real samples");
    tiny[0] = Complex{0, 1e-2};
    rejected = false;
    try {
      (void)inverse(tiny, o.Ne, c, true, p.ref.electric);
    } catch (const Failure &) {
      rejected = true;
    }
    require(rejected, "macroscopic imaginary field was accepted");
    std::cout << "PASS regional masks and weighted units\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "FAIL " << e.what() << '\n';
    return 1;
  }
}
