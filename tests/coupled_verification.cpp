#include "fomhd/verification.hpp"
#include "fomhd/diagnostics.hpp"
#include "fomhd/io.hpp"
#include "support.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <iostream>
#include <numbers>
using namespace fomhd;
using Json = nlohmann::json;
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
State difference(const State& a, const State& b) {
  State r=a;
  for (int s=0;s<2;++s) {
    r.fluid[s].n=subtract(a.fluid[s].n,b.fluid[s].n);
    r.fluid[s].mom=subtract(a.fluid[s].mom,b.fluid[s].mom);
    r.fluid[s].perp=subtract(a.fluid[s].perp,b.fluid[s].perp);
    if(s)r.fluid[s].parallel=subtract(a.fluid[s].parallel,b.fluid[s].parallel);
  }
  r.B=subtract(a.B,b.B);return r;
}
struct Trajectory {
  const Context& c;
  State equilibrium;
  double duration{1e-8}, speed{.1};
  State at(const StageTag& tag, bool derivative=false) const {
    State u=derivative?allocate_state(c):equilibrium;
    u.step=tag.step;u.generation=tag.generation;u.time=tag.time;
    const double t=tag.time/duration;
    auto wave=[&](double frequency){return derivative?frequency*std::cos(frequency*t)/duration:std::sin(frequency*t);};
    const auto nc=c.op.Nc;
    for(int s=0;s<2;++s) {
      const double mass=c.p.species[s].mass;
      const double pressure=c.p.ref.pressure*mass/c.p.species[0].mass;
      for(std::size_t j=0;j<nc;++j) {
        const double f=.6+.2*std::cos(c.geometry.cells.R[j])*std::cos(c.geometry.cells.Z[j]);
        if(!derivative) {
          u.fluid[s].perp[c.p.K*nc+j]=pressure;
          if(s)u.fluid[s].parallel[c.p.K*nc+j]=pressure;
        }
        for(int k:{-1,1}) {
          // A small charge separation keeps forcing and target momentum well conditioned.
          u.fluid[s].n[(c.p.K+k)*nc+j]=.01*(s?.9999:1.)*c.p.ref.ni*f*wave(1);
          u.fluid[s].mom[((c.p.K+k)*3+1)*nc+j]=.5*(s?1.3:1.)*mass*c.p.ref.ni*speed*f*wave(1.2);
          u.fluid[s].perp[(c.p.K+k)*nc+j]=.015*pressure*f*wave(.7);
          if(s)u.fluid[s].parallel[(c.p.K+k)*nc+j]=.011*pressure*f*wave(.7);
        }
      }
    }
    for(std::size_t j=0;j<u.B.size();++j)
      u.B[j]=equilibrium.B[j]*((derivative?0.:1.)+.01*wave(.8));
    return u;
  }
};
Json errors(const State& u,const State& exact,const Context& c) {
  Json out;
  auto component=[&](const char* name,const Vec& a,const Vec& b,double scale){
    out[name]=norm(subtract(a,b))/(scale*std::sqrt(double(a.size())));
  };
  for(int s=0;s<2;++s) {
    component(s?"n_e":"n_i",u.fluid[s].n,exact.fluid[s].n,c.p.ref.ni*.02);
    component(s?"mom_e":"mom_i",u.fluid[s].mom,exact.fluid[s].mom,c.p.species[s].mass*c.p.ref.ni*.1);
    double pressure=c.p.ref.pressure*c.p.species[s].mass/c.p.species[0].mass;
    component(s?"p_perp_e":"p_i",u.fluid[s].perp,exact.fluid[s].perp,pressure*.03);
    if(s)component("p_parallel_e",u.fluid[s].parallel,exact.fluid[s].parallel,pressure*.03);
  }
  component("B",u.B,exact.B,c.p.ref.magnetic*.01);
  double combined=0;
  for (auto value : out) combined=std::hypot(combined,value.get<double>());
  out["combined"]=combined;
  return out;
}
Json rates(const verification::SourceRates &r) {
  return {{"number_i_s^-1",r.totals.number[0]},{"number_e_s^-1",r.totals.number[1]},
          {"kinetic_i_W",r.totals.kinetic[0]},{"kinetic_e_W",r.totals.kinetic[1]},
          {"ion_internal_W",r.totals.ion_internal},{"electron_perp_W",r.totals.electron_perp},
          {"electron_parallel_W",r.totals.electron_parallel},{"magnetic_W",r.totals.magnetic},
          {"longitudinal_W",r.totals.longitudinal},{"energy_W",r.totals.energy},
          {"topology_Wb_s^-1",r.totals.topology},{"physical_flux_Wb_s^-1",r.totals.physical_flux},
          {"momentum_x_N",r.momentum.value[0]},{"momentum_y_N",r.momentum.value[1]},
          {"momentum_z_N",r.momentum.value[2]},{"angular_momentum_Nm",r.momentum.angular}};
}
Json run(bool omit) {
  auto p=test::parameters();p.Nphi=9;
  auto g=build_rectangle(4,4,1,0,.2,.2);auto op=build_operators(g,p.K);Compute cpu;
  Json levels=Json::array();
  for(int steps:{8,16,32}) {
    p.dt=1e-8/steps;p.nsteps=steps;Context c{p,g,op,cpu,p.Nphi};
    Trajectory path{c,closed_reference(c,p.ref.ni,p.ref.pressure,
      p.ref.pressure*p.species[1].mass/p.species[0].mass,p.ref.magnetic)};
    Cumulative ledger;auto initial=path.at({0,1,0,0,0,"reference"});
    auto u=initialize(initial,c,ledger,true);
    Json pending_rates,integrals,source_intervals=Json::array();
    double max_EL=0,max_ET=0,max_physical_B_rate=0;
    auto source=[&](const State& actual,const StageTag& tag,Context& context){
      auto reference=path.at(tag);auto physical=evaluate_stage(reference,tag,context,false);
      auto result=difference(path.at(tag,true),physical.rhs);
      if(omit)std::fill(result.fluid[0].perp.begin(),result.fluid[0].perp.end(),Complex{});
      if(tag.stage=="midpoint") {
        const auto frame=evaluate_stage(actual,tag,context,false);
        pending_rates=rates(verification::source_rates(result,actual,frame,context));
        max_EL=std::max(max_EL,max_abs(frame.cache.EL));
        max_ET=std::max(max_ET,max_abs(frame.cache.ET));
        max_physical_B_rate=std::max(max_physical_B_rate,max_abs(frame.rhs.B));
      }
      return result;
    };
    Json checks=Json::array();
    for(int step=0;step<steps;++step) {
      auto r=verification::advance(u,ledger,c,source);
      for(auto it=pending_rates.begin();it!=pending_rates.end();++it) {
        if(!integrals.contains(it.key()))integrals[it.key()]=0.;
        integrals[it.key()]=integrals[it.key()].get<double>()+p.dt*it.value().get<double>();
      }
      source_intervals.push_back({{"step",u.state.step},{"midpoint_time_s",(step+.5)*p.dt},
          {"weight_s",p.dt},{"accepted",r.interval.accepted},{"rates",pending_rates}});
      for(const auto& q:r.checks) {
        require(!q.hard||(q.available&&q.passed),"failed hard diagnostic escaped advancement");
        require(!q.available||(std::isfinite(q.raw)&&std::isfinite(q.threshold)&&std::isfinite(q.reference)),
                "nonfinite populated diagnostic");
        if(step==steps-1)checks.push_back({{"name",q.name},{"population",q.population},{"region",q.region},
          {"component",q.component},{"producer",q.producer},{"norm",static_cast<int>(q.norm)},
          {"quantity_unit",q.unit},{"value_unit",q.value_unit},{"stage",q.tag.stage},{"time_s",q.tag.time},
          {"step",q.tag.step},{"generation",q.tag.generation},{"trial",q.tag.trial},
          {"raw",q.raw},{"reference",q.reference},{"threshold",q.threshold},{"hard",q.hard},
          {"available",q.available},{"reason",q.reason},{"passed",q.passed}});
      }
    }
    require(max_EL>1e-20&&max_ET>1e-20&&max_physical_B_rate>1e-20,
            "manufactured trajectory did not excite coupled field response");
    auto exact=path.at(u.cache.tag);
    levels.push_back({{"steps",steps},{"dt_s",p.dt},{"errors",errors(u.state,exact,c)},
                      {"checks",checks},{"accepted_intervals",ledger.accepted_intervals},
                      {"source_intervals",source_intervals},{"source_integrals_rate_times_seconds",integrals},
                      {"maximum_response",{{"EL_Vpm",max_EL},{"ET_Vpm",max_ET},{"physical_B_rate_Tps",max_physical_B_rate}}}});
  }
  Json orders;bool passed=true;
  for(auto it=levels[0]["errors"].begin();it!=levels[0]["errors"].end();++it) {
    std::string key=it.key();double e0=it.value(),e1=levels[1]["errors"][key],e2=levels[2]["errors"][key];
    double a=std::log2(e0/e1),b=std::log2(e1/e2);
    orders[key]={a,b};passed&=std::isfinite(a)&&std::isfinite(b)&&a>1.7&&a<2.3&&b>1.7&&b<2.3;
  }
  return {{"case","coupled_discrete_temporal_MMS"},{"levels",levels},{"orders",orders},{"passed",passed},
          {"parameters",{{"K",p.K},{"Nphi",p.Nphi},{"n_ref_m^-3",p.ref.ni},
              {"duration_s",1e-8},{"density_harmonic_fraction",.01},{"electron_density_factor",.9999},
              {"electron_momentum_factor",1.3},{"ion_mass_kg",p.species[0].mass},{"electron_mass_kg",p.species[1].mass}}},
          {"source_accounting","Test-only accepted midpoint source integrals; production cumulative ledger contains physical terms only; no manufactured restart is exposed."},
          {"reference","independent analytic time trajectory; prescribed source evaluated only at reference state"},
          {"scope","Verification under known sources; not physical validation"}};
}
}
int main(int argc,char** argv) {
  try {
    const bool negative=argc>1&&std::string(argv[1])=="negative";
    auto result=run(negative);
    Json identity={{"compiler",__VERSION__},{"precision","fp64/complex128"},
                   {"executable_sha256",io::sha256_file(std::filesystem::absolute(argv[0]))}};
    for(const auto *relative : {"tests/coupled_verification.cpp","src/time/midpoint.cpp",
                               "src/diagnostics/source.cpp","src/diagnostics/diagnostics.cpp"})
      identity["source_hashes"][relative]=io::sha256_file(std::filesystem::path(FOMHD_SOURCE_ROOT)/relative);
    result["build_identity"]=identity;
    const bool accuracy=result["passed"].get<bool>();
    result["accuracy_passed"]=accuracy;
    if(negative) {result["negative_control_detected"]=!accuracy;result["passed"]=!accuracy;}
    std::cout<<result.dump(2)<<'\n';
    return result["passed"].get<bool>()?0:1;
  } catch(const std::exception& e) {std::cerr<<"Coupled verification failed: "<<e.what()<<'\n';return 2;}
}
