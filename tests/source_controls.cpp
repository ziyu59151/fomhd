#include "fomhd/verification.hpp"
#include "fomhd/diagnostics.hpp"
#include "support.hpp"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>
using namespace fomhd;
using Json = nlohmann::json;
namespace {
void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
void identical(const State &a, const State &b) {
  require(a.B == b.B && a.step == b.step && a.time == b.time &&
              a.generation == b.generation, "zero source changed state identity or B");
  for (int s = 0; s < 2; ++s)
    require(a.fluid[s].n == b.fluid[s].n && a.fluid[s].mom == b.fluid[s].mom &&
                a.fluid[s].perp == b.fluid[s].perp &&
                a.fluid[s].parallel == b.fluid[s].parallel,
            "zero source changed fluid state");
}
State reference_fixture(const Context &c) {
  auto u = closed_reference(c, c.p.ref.ni, c.p.ref.pressure,
                            c.p.ref.pressure, c.p.ref.magnetic, 1);
  for (std::size_t j = 0; j < c.op.Nc; ++j)
    u.fluid[0].n[c.p.K * c.op.Nc + j] *= 1.0001;
  return u;
}
State source_fixture(const State &u, const Context &c) {
  auto source = allocate_state(c);
  const auto nc = c.op.Nc;
  for (int a = 0; a < 2; ++a) {
    for (std::size_t j = 0; j < nc; ++j) {
      source.fluid[a].n[c.p.K * nc + j] = (a ? .03 : .01) * c.p.ref.ni / c.p.ref.time;
      source.fluid[a].perp[c.p.K * nc + j] = (a ? .2 : .1) * c.p.ref.pressure / c.p.ref.time;
      if (a) source.fluid[a].parallel[c.p.K * nc + j] = -.1 * c.p.ref.pressure / c.p.ref.time;
      const double force = c.p.species[a].mass * c.p.ref.ni / c.p.ref.time;
      source.fluid[a].mom[(c.p.K * 3 + 1) * nc + j] = .03 * force;
      for (int k : {-1, 1})
        source.fluid[a].mom[((c.p.K + k) * 3 + 1) * nc + j] =
            .05 * force * Complex(1, k);
    }
  }
  source.B = scaled(u.B, .01 / c.p.ref.time);
  return source;
}
verification::SourceRates independent_rates(const State &u, const State &source,
                                             const StageTag &tag, Context &c) {
  const double h = c.p.ref.time * 1e-4;
  auto plus_tag = tag, minus_tag = tag;
  plus_tag.generation = 2; minus_tag.generation = 3;
  const auto plus = combine(u, source, h, plus_tag, c);
  const auto minus = combine(u, source, -h, minus_tag, c);
  const auto positive = evaluate_stage(plus, plus_tag, c, false);
  const auto negative = evaluate_stage(minus, minus_tag, c, false);
  const auto difference = [&](double upper, double lower) { return (upper-lower)/(2*h); };
  verification::SourceRates expected;
  for (int a = 0; a < 2; ++a) {
    expected.totals.number[a] = difference(positive.totals.number[a], negative.totals.number[a]);
    expected.totals.kinetic[a] = difference(positive.totals.kinetic[a], negative.totals.kinetic[a]);
  }
  for (auto field : {&Totals::ion_internal, &Totals::electron_perp,
       &Totals::electron_parallel, &Totals::magnetic, &Totals::longitudinal,
       &Totals::energy, &Totals::topology, &Totals::physical_flux})
    expected.totals.*field = difference(positive.totals.*field, negative.totals.*field);
  for (int d = 0; d < 3; ++d)
    expected.momentum.value[d] = difference(positive.momentum.value[d], negative.momentum.value[d]);
  expected.momentum.angular = difference(positive.momentum.angular, negative.momentum.angular);
  return expected;
}
Json chain_rule(Context &c) {
  const auto u = reference_fixture(c);
  StageTag tag{u.step, u.generation, 1, c.spectral_id, u.time, "chain_rule"};
  const auto frame = evaluate_stage(u, tag, c, false);
  const auto source = source_fixture(u, c);
  const auto rate = verification::source_rates(source, u, frame, c);
  const auto expected = independent_rates(u, source, tag, c);
  Json records = Json::array();
  bool passed = true;
  auto check = [&](const char *name, double actual, double independent, double reference) {
    const double expected = independent;
    const double scale = std::max(std::abs(expected), reference * 1e-6);
    const double error = std::abs(actual - expected) / scale;
    const bool active = std::abs(expected) > reference * 1e-6;
    const bool good = active && std::isfinite(error) && error < 2e-5;
    passed &= good;
    records.push_back({{"name",name},{"analytic_rate",actual},
                      {"finite_difference_rate",expected},{"scaled_error",error},
                      {"signal_present",active},{"passed",good}});
  };
  double volume = 0;
  for (double w : c.op.M0) volume += 2 * std::acos(-1.) * w;
  for (int a = 0; a < 2; ++a) {
    check(a ? "number_e" : "number_i", rate.totals.number[a], expected.totals.number[a], c.p.ref.ni * volume / c.p.ref.time);
    check(a ? "kinetic_e" : "kinetic_i", rate.totals.kinetic[a], expected.totals.kinetic[a], c.p.species[a].mass * c.p.ref.ni * volume / c.p.ref.time);
  }
  const double power = frame.totals.energy / c.p.ref.time;
  for (auto [name, field] : std::initializer_list<std::pair<const char*, double Totals::*>>{
       {"ion_internal", &Totals::ion_internal}, {"electron_perp", &Totals::electron_perp},
       {"electron_parallel", &Totals::electron_parallel}, {"magnetic", &Totals::magnetic},
       {"longitudinal", &Totals::longitudinal}, {"energy", &Totals::energy}})
    check(name, rate.totals.*field, expected.totals.*field, power);
  for (auto [name, field] : std::initializer_list<std::pair<const char*, double Totals::*>>{
       {"topology", &Totals::topology}, {"physical_flux", &Totals::physical_flux}})
    check(name, rate.totals.*field, expected.totals.*field,
          c.p.ref.magnetic * c.p.ref.length * c.p.ref.length / c.p.ref.time);
  const double force = c.p.species[0].mass * c.p.ref.ni * volume / c.p.ref.time;
  for (int d = 0; d < 3; ++d)
    check(d == 0 ? "momentum_x" : d == 1 ? "momentum_y" : "momentum_z",
          rate.momentum.value[d], expected.momentum.value[d], force);
  check("angular_momentum", rate.momentum.angular, expected.momentum.angular, force * c.p.ref.length);
  return {{"case","source_chain_rule"},{"difference_step_s",c.p.ref.time * 1e-4},
          {"records",records},{"passed",passed}};
}
void zero_source(Context &c) {
  Context a{c.p,c.geometry,c.op,c.compute,c.N}, b{c.p,c.geometry,c.op,c.compute,c.N};
  const auto initial = closed_reference(c,c.p.ref.ni,c.p.ref.pressure,c.p.ref.pressure,c.p.ref.magnetic,1);
  Cumulative la, lb;
  auto ua = initialize(initial,a,la,false), ub = initialize(initial,b,lb,false);
  advance(ua,la,a);
  verification::advance(ub,lb,b,[](const State&,const StageTag&,Context &ctx){return allocate_state(ctx);});
  identical(ua.state,ub.state);
  require(ua.cache.E == ub.cache.E && ua.cache.EL == ub.cache.EL && ua.cache.ET == ub.cache.ET,
          "zero source changed field cache");
  require(la.accepted_intervals == lb.accepted_intervals && la.last_trial == lb.last_trial &&
              la.boundary_integrals.total_energy == lb.boundary_integrals.total_energy,
          "zero source changed ledger");
}
Json omitted_accounting(Context &c) {
  Cumulative ledger;
  auto u = initialize(reference_fixture(c), c, ledger, false);
  const State left = u.state;
  const auto baseline = ledger;
  const auto source = source_fixture(left, c);
  State middle;
  StageTag base_tag, middle_tag;
  const auto callback = [&](const State &state,const StageTag &tag,Context&){
    if(tag.stage=="midpoint") {middle=state;middle_tag=tag;} else base_tag=tag;
    return source;
  };
  const auto accounted = verification::advance(u,ledger,c,callback);
  const auto base=evaluate_stage(left,base_tag,c,false);
  const auto mid=evaluate_stage(middle,middle_tag,c,false);
  const auto end=evaluate_stage(u.state,u.cache.tag,c,false);
  const auto expected=independent_rates(middle,source,middle_tag,c);
  std::vector<Check> omitted;
  try {(void)check_step(left,base,middle,mid,u.state,end,baseline,c);}
  catch(const Failure &f) {
    if (f.code != "HardGate") throw;
    omitted=f.checks;
  }
  require(!omitted.empty(),"omitted source accounting escaped hard balance checks");
  const auto record = [](const std::vector<Check> &checks, const std::string &name,
                         const std::string &population) -> const Check& {
    const Check *found=nullptr;
    for (const auto &check : checks)
      if (check.name==name && check.population==population &&
          check.tag.stage=="interval" && check.norm==Norm::value) {
        require(!found,"duplicate scalar interval record");
        found=&check;
      }
    require(found && found->available,"missing scalar interval record");
    return *found;
  };
  Json records=Json::array();
  bool passed=true;
  const auto check = [&](const char *name, double rate, const std::string &population="") {
    const auto &aware=record(accounted.checks,name,population);
    const auto &unforced=record(omitted,name,population);
    const double observed=unforced.raw-aware.raw, integral=c.p.dt*rate;
    const double error=std::abs(observed-integral)/std::abs(integral);
    const bool good=std::isfinite(error) && integral!=0 && error<2e-5;
    passed &= good;
    records.push_back({{"name",name},{"population",population},{"unit",aware.unit},
        {"independent_rate",rate},{"expected_integral",integral},
        {"observed_accounting_difference",observed},{"relative_error",error},
        {"production_hard",aware.hard},{"passed",good}});
  };
  for (int a=0;a<2;++a) {
    check("total_number_balance",expected.totals.number[a],c.p.species[a].id);
    check("kinetic_energy",expected.totals.kinetic[a],c.p.species[a].id);
  }
  check("total_charge_balance",c.p.species[0].charge*expected.totals.number[0]+
        c.p.species[1].charge*expected.totals.number[1]);
  for (auto [name, field] : std::initializer_list<std::pair<const char*, double Totals::*>>{
       {"ion_internal", &Totals::ion_internal}, {"electron_perp", &Totals::electron_perp},
       {"electron_parallel", &Totals::electron_parallel}, {"field_energy_B", &Totals::magnetic},
       {"field_energy_L", &Totals::longitudinal}, {"total_energy", &Totals::energy},
       {"topology_increment", &Totals::topology}})
    check(name,expected.totals.*field);
  for (int d=0;d<3;++d)
    check(d==0?"momentum_x":d==1?"momentum_y":"momentum_z",expected.momentum.value[d]);
  check("angular_momentum",expected.momentum.angular);
  require(record(accounted.checks,"field_exchange","").raw==record(omitted,"field_exchange","").raw,
          "source accounting changed physical field exchange");
  return {{"records",records},{"passed",passed},{"omitted_source_rejected",true},
          {"physical_field_exchange_unchanged",true},{"midpoint_time_s",middle_tag.time},
          {"weight_s",c.p.dt}};
}
}
int main() {
  try {
    auto p=test::parameters();p.Nphi=9;p.dt=1e-10;
    auto g=build_rectangle(4,4,1,0,.2,.2);auto op=build_operators(g,p.K);Compute cpu;
    Context c{p,g,op,cpu,p.Nphi};
    auto result=chain_rule(c);
    zero_source(c);
    result["interval_accounting"]=omitted_accounting(c);
    result["passed"]=result["passed"].get<bool>() && result["interval_accounting"]["passed"].get<bool>();
    result["zero_source_identical"]=true;result["omitted_accounting_rejected"]=true;
    std::cout<<result.dump(2)<<'\n';
    return result["passed"].get<bool>()?0:1;
  }catch(const std::exception &error){std::cerr<<"Source controls failed: "<<error.what()<<'\n';return 2;}
}
