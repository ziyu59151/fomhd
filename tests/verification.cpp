#include "fomhd/diagnostics.hpp"
#include "support.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numbers>
using namespace fomhd;
using Json = nlohmann::json;
namespace {
void require(bool pass, const char *message) {
  if (!pass)
    throw NumericalError(message);
}
double error(const State &a, const State &b, const Parameters &p) {
  double total = 0;
  const auto add_error = [&](const Vec &x, const Vec &y, double ref) {
    total = std::hypot(total, norm(subtract(x, y)) /
                                  (ref * std::sqrt(double(x.size()))));
  };
  for (int s = 0; s < 2; ++s) {
    add_error(a.fluid[s].n, b.fluid[s].n, p.ref.ni);
    add_error(a.fluid[s].mom, b.fluid[s].mom,
              p.species[s].mass * p.ref.ni * 10);
    add_error(a.fluid[s].perp, b.fluid[s].perp, p.ref.pressure);
    if (s)
      add_error(a.fluid[s].parallel, b.fluid[s].parallel, p.ref.pressure);
  }
  add_error(a.B, b.B, p.ref.magnetic);
  return total;
}
Json temporal() {
  auto p = test::parameters();
  p.Nphi = 9;
  Compute cpu;
  const auto g = build_rectangle(4, 4, 1, 0, .2, .2);
  const auto op = build_operators(g, p.K);
  const double end_time = 1.6e-8;
  std::vector<State> solutions;
  Json runs = Json::array();
  for (int steps : {8, 16, 32, 128}) {
    p.dt = end_time / steps;
    p.nsteps = steps;
    Context c{p, g, op, cpu, p.Nphi};
    auto initial = closed_reference(c, p.ref.ni, p.ref.pressure, p.ref.pressure,
                                    p.ref.magnetic, 10);
    Cumulative ledger;
    auto state = initialize(initial, c, ledger, false);
    double max_gate = 0;
    const auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < steps; ++n) {
      auto result = advance(state, ledger, c);
      for (const auto &r : result.checks)
        if (r.hard && r.threshold > 0 && r.name.find("minimum_") != 0)
          max_gate = std::max(max_gate, std::abs(r.raw) / r.threshold);
    }
    solutions.push_back(state.state);
    runs.push_back(
        {{"steps", steps},
         {"dt_s", p.dt},
         {"end_time_s", state.state.time},
         {"max_hard_gate_ratio", max_gate},
         {"advance_elapsed_s", std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count()}});
    std::cerr << "Time refinement: " << steps << " accepted steps\n";
  }
  RealVec errors;
  for (int i = 0; i < 3; ++i) {
    errors.push_back(error(solutions[i], solutions.back(), p));
    runs[i]["normalized_state_error"] = errors.back();
  }
  RealVec orders{std::log2(errors[0] / errors[1]),
                 std::log2(errors[1] / errors[2])};
  Json result = {{"case", "unforced_closed_torus_time_refinement"},
                 {"reference_steps", 128},
                 {"runs", runs},
                 {"orders", orders},
                 {"acceptance", "Both orders must be in [1.7, 2.4]; all "
                                "production hard diagnostics must pass."}};
  result["passed"] = orders[0] >= 1.7 && orders[0] <= 2.4 && orders[1] >= 1.7 &&
                     orders[1] <= 2.4;
  return result;
}
Json manufactured_time() {
  auto p=test::parameters(0);Compute cpu;const auto g=build_rectangle(4,4,1,0,.2,.2);const auto op=build_operators(g,0);
  const double T=1e-7;Json levels=Json::array();RealVec errors;
  for(int steps:{4,8,16}) {
    p.dt=T/steps;p.nsteps=steps;Context c{p,g,op,cpu,p.Nphi};Cumulative ledger;
    auto state=initialize(closed_reference(c,p.ref.ni,p.ref.pressure,p.ref.pressure,p.ref.magnetic),c,ledger,true).state;
    const auto forced=[&](const State& u,const StageTag& tag){
      auto frame=evaluate_stage(u,tag,c,false);
      // Test-only, prescribed uniform heating; no momentum source enters fJ0.
      for(std::size_t j=0;j<op.Nc;++j){
        frame.rhs.fluid[0].perp[j]+=p.ref.pressure*std::exp(u.time/T)/T;
        const double q=.5*p.ref.pressure*std::exp(.5*u.time/T)/T;
        frame.rhs.fluid[1].perp[j]+=q;frame.rhs.fluid[1].parallel[j]+=q;
      }
      return frame.rhs;
    };
    for(int n=0;n<steps;++n){
      StageTag base{state.step,state.generation,std::uint64_t(n+1),0,state.time,"mms_base"};
      auto f=forced(state,base);auto mt=base;mt.generation++;mt.time=time_at_midpoint(p,n);mt.stage="mms_midpoint";
      auto mid=combine(state,f,p.dt/2,mt,c);auto fm=forced(mid,mt);auto et=base;et.step++;et.generation+=2;et.time=time_at_step(p,n+1);et.stage="mms_endpoint";
      state=combine(state,fm,p.dt,et,c);(void)evaluate_stage(state,et,c,false);
    }
    double e=0;for(std::size_t j=0;j<op.Nc;++j){
      e=std::max(e,std::abs(state.fluid[0].perp[j]/p.ref.pressure-std::exp(1.0)));
      e=std::max(e,std::abs(state.fluid[1].perp[j]/p.ref.pressure-std::exp(.5)));
      e=std::max(e,std::abs(state.fluid[1].parallel[j]/p.ref.pressure-std::exp(.5)));
    }
    errors.push_back(e);levels.push_back({{"steps",steps},{"dt_s",p.dt},{"normalized_pressure_linf",e}});
  }
  RealVec orders{std::log2(errors[0]/errors[1]),std::log2(errors[1]/errors[2])};
  return {{"case","uniform_pressure_temporal_MMS"},{"scope","Manufactured pressure heating in test harness; real stage, field, combination and hard diagnostics. No production forcing option."},
    {"levels",levels},{"orders",orders},{"passed",orders[0]>1.8&&orders[0]<2.2&&orders[1]>1.8&&orders[1]<2.2}};
}
Json compression() {
  auto p=test::parameters(0);Compute cpu;const auto g=build_rectangle(7,7,1,-.3,.1,.1);const auto op=build_operators(g,0);Context c{p,g,op,cpu,1};
  const auto nc=op.Nc,j=g.cell_id(3,3);Vec b(3*nc),B(3*nc);for(std::size_t i=0;i<nc;++i){b[2*nc+i]=1;B[2*nc+i]=p.ref.magnetic;}
  Json cases=Json::array();bool passed=true;
  for(bool parallel:{true,false}){
    const double theta=1e5,strain=parallel?theta:0,a=parallel?0:theta/2,az=parallel?theta:0,T=1e-6;
    const auto rate=[&](const Vec& y){
      FluidState u;u.n=Vec(nc,y[0]);u.perp=Vec(nc,y[1]);u.parallel=Vec(nc,y[2]);u.mom=Vec(3*nc);
      for(std::size_t i=0;i<nc;++i){u.mom[i]=p.species[1].mass*y[0]*a*g.cells.R[i];u.mom[2*nc+i]=p.species[1].mass*y[0]*az*g.cells.Z[i];}
      auto f=build_fluid(u,1,B,b,{0,1,1,0,0,"compression_local"},c);build_non_electric(f,1,B,b,c);
      return Vec{-f.divergence[j],f.pdot_perp[j],f.pdot_parallel[j]};
    };
    Vec initial{p.ref.ne,p.ref.pressure,2*p.ref.pressure};
    Vec exact{initial[0]*std::exp(-theta*T),initial[1]*std::exp((strain-2*theta)*T),initial[2]*std::exp(-(theta+2*strain)*T)};
    RealVec errors;Json levels=Json::array();
    for(int steps:{8,16,32}){
      Vec u=initial;const double dt=T/steps;
      for(int n=0;n<steps;++n){auto midpoint=cpu.axpy(u,rate(u),dt/2);u=cpu.axpy(u,rate(midpoint),dt);}
      double e=0;for(int d=0;d<3;++d)e=std::max(e,std::abs((u[d]-exact[d])/initial[d]));
      errors.push_back(e);levels.push_back({{"steps",steps},{"normalized_linf",e}});
    }
    RealVec orders{std::log2(errors[0]/errors[1]),std::log2(errors[1]/errors[2])};
    passed&=orders[0]>1.8&&orders[0]<2.2&&orders[1]>1.8&&orders[1]<2.2;
    cases.push_back({{"orientation",parallel?"parallel":"perpendicular"},{"theta_per_s",theta},{"strain_per_s",strain},{"levels",levels},{"orders",orders}});
  }
  return {{"case","CGL_B_prescribed_constant_strain"},{"scope","Local material response using production fluid routines at an interior cell, prescribed velocity and B direction. Uniform compression is not a closed-wall global solution."},{"cases",cases},{"passed",passed}};
}
Json spatial(int maximum) {
  Json levels = Json::array();
  RealVec errors;
  const double pi = std::numbers::pi;
  require(maximum==32||maximum==64,"spatial maximum must be 32 or 64");
  for (int n : {maximum/4,maximum/2,maximum}) {
    const double hr = 1.0 / n, hz = 2.0 / n;
    const auto g =
        build_rectangle(n + 1, n / 2 + 1, 1 + hr / 2, hz / 2, hr, hz);
    const auto G = build_gradient_operator(g, 1);
    const auto nc = g.cells.keys.size(), nr = g.r_faces.keys.size();
    Vec phi(nc);
    for (std::size_t i = 0; i < nc; ++i)
      phi[i] = std::sin(pi * (g.cells.R[i] - 1)) * std::sin(pi * g.cells.Z[i]);
    auto grad = matvec(G, phi);
    std::array<double, 3> linf{}, l2{};
    std::array<std::size_t, 3> count{};
    const auto accumulate = [&](const EntityTable &e, std::size_t offset,
                                int component) {
      for (std::size_t i = 0; i < e.keys.size(); ++i) {
        double R = e.R[i], Z = e.Z[i], x = R - 1;
        Complex exact =
            component == 0 ? Complex(pi * std::cos(pi * x) * std::sin(pi * Z))
            : component == 2
                ? Complex(pi * std::sin(pi * x) * std::cos(pi * Z))
                : Complex(0, std::sin(pi * x) * std::sin(pi * Z) / R);
        const double defect = std::abs(grad[offset + i] - exact);
        const double distance = std::min({x, 1 - x, Z, 1 - Z});
        const int region = distance >= .20 ? 1 : 2;
        for (int r : {0, region}) {
          linf[r] = std::max(linf[r], defect);
          l2[r] = std::hypot(l2[r], std::sqrt(g.M1[offset + i]) * defect);
          ++count[r];
        }
      }
    };
    accumulate(g.r_faces, 0, 0);
    accumulate(g.cells, nr, 1);
    accumulate(g.z_faces, nr + nc, 2);
    require(count[1] > 0, "empty manufactured interior");
    errors.push_back(linf[1]);
    levels.push_back({{"NR", n},
                      {"NZ", n / 2},
                      {"hR_m", hr},
                      {"hZ_m", hz},
                      {"linf_all_inner_wall", linf},
                      {"L2_per_radian_all_inner_wall", l2},
                      {"selected_all_inner_wall", count}});
  }
  RealVec order{std::log2(errors[0] / errors[1]),
                std::log2(errors[1] / errors[2])};
  return {
      {"case", "continuum_Dirichlet_gradient_MMS"},
      {"formula",
       "phi=sin(pi*(R-1))*sin(pi*Z)*exp(i*phi_angle); R in [1,2], Z in [0,1]"},
      {"levels", levels},
      {"inner_distance_m", .20},
      {"inner_orders", order},
      {"acceptance", "Both interior orders above 3.5; global and wall errors "
                     "are reported without a fourth-order claim."},
      {"passed", order[0] > 3.5 && order[1] > 3.5}};
}
} // namespace
int main(int argc, char **argv) {
  try {
    const std::string which = argc > 1 ? argv[1] : "spatial";
    Json result;
    if (which == "temporal")
      result = temporal();
    else if (which == "mms_time")result=manufactured_time();
    else if (which == "compression")result=compression();
    else if (which == "spatial")
      result = spatial(argc>2?std::stoi(argv[2]):64);
    else
      throw NumericalError("unknown verification case");
    std::cout << result.dump(2) << '\n';
    return result.at("passed").get<bool>() ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "Verification failed: " << e.what() << '\n';
    return 1;
  }
}
