#pragma once
#include "fomhd/physics.hpp"
#include "fomhd/backend.hpp"
#include <cmath>
namespace fomhd::test {
inline Parameters parameters(int K=1) {
    Parameters p;
    p.species={Species{"ion",1.67262192595e-27,1.602176634e-19,5.0/3,0,0,0,0},
               Species{"electron",9.1093837139e-31,-1.602176634e-19,0,0,0,0,0}};
    p.epsilon0=8.8541878188e-12;p.mu0=1.25663706127e-6;
    p.K=K;p.Nphi=4*K+1;p.rank_rtol=1e-12;p.dt=1e-9;p.nsteps=4;
    p.memory_limit=512ULL<<20;p.device_memory_limit=512ULL<<20;p.B_min=1e-12;
    p.inner_distance=.05;
    p.ref={.2,1e-6,1e12,1e12,1,1,1e-5,1,1e-7,1e-9};
    double response=0;for(auto s:p.species)response+=s.charge*s.charge*p.ref.ni/s.mass;
    p.ref.current=p.ref.time*p.ref.electric*(response+1/(p.mu0*p.ref.length*p.ref.length));
    const double L=p.ref.length,t=p.ref.time,n=p.ref.ni,E=p.ref.electric,B=p.ref.magnetic,P=p.ref.pressure;
    const double force=p.species[0].mass*n*p.ref.velocity/t;
    const double power=P*L*L*L/t,energy=P*L*L*L+B*B*L*L*L/p.mu0;
    const auto array=[&](const std::string& key,const std::string& unit,double ref,bool hard=true){
        p.budgets[key+".linf"]={0,1e-9,ref,unit,Norm::linf,hard};
        p.budgets[key+".weighted_L2"]={0,1e-9,ref,unit,Norm::weighted_l2,hard};
    };
    const auto scalar=[&](const std::string& key,const std::string& unit,double ref,bool hard=true){
        p.budgets[key+".value"]={0,1e-9,ref,unit,Norm::value,hard};
    };
    for(auto name:{"gauss","poisson_linear","transverse_divergence"})array(name,"V/m^2",E/L);
    array("longitudinal_projection","A/m^3",p.ref.current/L);
    array("darwin_dynamic","A/(m^2*s)",p.ref.current/t);
    for(auto name:{"current_divB","current_ampere"})array(name,"T/m",B/L,false);
    for(auto name:{"initial_divB","initial_ampere","divB_update","long_term_divB_change"})array(name,"T/m",B/L);
    array("CT_increment","T/(m*s)",B/(L*t));
    for(auto name:{"CT_update","equilibrium_B","alias_B"})array(name,"T/s",B/t);
    array("current_response","A/(m^2*s)",p.ref.current/t);
    for(auto name:{"RC_flux","RC_mom","deltaC"})array(name,"C/(m^3*s)",p.ref.charge/t);
    array("RL","A/m^2",p.ref.current);
    for(auto name:{"ion_internal","electron_perp","electron_parallel","field_energy_B","field_energy_L","total_energy"})scalar(name,"J",energy,false);
    scalar("field_exchange","W",energy/t,false);
    scalar("topology_increment","Wb",B*L*L);
    scalar("total_charge_balance","C",p.ref.charge*L*L*L);
    for(auto name:{"momentum_x","momentum_y","momentum_z"})scalar(name,"kg*m/s",p.species[0].mass*n*L*L*L*p.ref.velocity,false);
    scalar("angular_momentum","kg*m^2/s",p.species[0].mass*n*std::pow(L,4)*p.ref.velocity,false);
    for(int a=0;a<2;++a) {
        const auto id="."+p.species[a].id;
        for(auto name:{"stage_continuity","species_continuity","alias_n","equilibrium_n"})array(name+id,"1/(m^3*s)",n/t);
        for(auto name:{"wall_rhs","wall_multiplier","alias_mom","equilibrium_mom"})array(name+id,"kg/(m^2*s^2)",force);
        array("wall_state"+id,"kg/(m^2*s)",force*t);
        array("theta_identity"+id,"1/s",1/t);
        array("wall_number_flux"+id,"1/(m^2*s)",n*p.ref.velocity);
        array("wall_perp_flux"+id,"Pa*m/s",P*p.ref.velocity);
        if(a)array("wall_parallel_flux"+id,"Pa*m/s",P*p.ref.velocity);
        for(auto name:{"alias_perp","equilibrium_perp"})array(name+id,"Pa/s",P/t);
        if(a)for(auto name:{"cgl_source","alias_parallel","equilibrium_parallel"})array(name+id,"Pa/s",P/t);
        for(auto name:{"electric_pairing","pressure_pairing","magnetic_work","wall_work","convection_pairing"})scalar(name+id,"W",power);
        scalar("spectral_transfer_defect"+id,"W",power,false);
        scalar("total_number_balance"+id,"1",n*L*L*L);
        scalar("kinetic_energy"+id,"J",energy,false);
    }
    return p;
}
} // namespace fomhd::test
