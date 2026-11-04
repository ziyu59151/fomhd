#pragma once
#include "fomhd/types.hpp"

namespace fomhd {
const Budget &budget(const Context &, const std::string &name);
void measure(std::vector<Check> &, const std::string &, const Vec &,
             const RealVec &weights, const std::string &quantity_unit,
             const StageTag &, const Context &,
             const std::string &population = "", bool samples = false,
             const std::string &component = "");
void scalar_check(std::vector<Check> &, const std::string &, double,
                  const std::string &unit, const StageTag &, const Context &,
                  const std::string &population = "");
void gate(const std::vector<Check> &,
          const std::vector<std::string> &required = {});
void gate_records(const std::vector<Check> &,
                  const std::vector<Check> &required);
std::vector<Check> stage_requirements(const Context &, const StageTag &);
void check_stage(const State &, StageResult &, const Context &);
StepResult check_step(const State &, const StageResult &, const State &,
                      const StageResult &, const State &, const StageResult &,
                      const Cumulative &, const Context &);
StepResult check_step(const State &, const StageResult &, const State &,
                      const StageResult &, const State &, const StageResult &,
                      const Cumulative &, const Context &, const State *prescribed_source);
} // namespace fomhd
