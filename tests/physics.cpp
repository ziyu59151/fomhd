#include "fomhd/backend.hpp"
#include "fomhd/physics.hpp"
#include "fomhd/diagnostics.hpp"
#include <cmath>
#include <iostream>
#include <limits>

namespace {
void require(bool b,const char* why) { if(!b) throw std::runtime_error(why); }
fomhd::Parameters params() {
    fomhd::Parameters p;
    p.species={fomhd::Species{"ion",1.67262192595e-27,1.602176634e-19,5.0/3,0,0,0,0},
               fomhd::Species{"electron",9.1093837139e-31,-1.602176634e-19,0,0,0,0,0}};
    p.epsilon0=8.8541878188e-12; p.mu0=1.25663706127e-6;
    p.ref={1,1e-6,1e12,1e12,1,1,1e-5,1e-6,1e-7,1e-6};
    p.K=1;p.Nphi=9;p.rank_rtol=1e-12;p.dt=1e-10;p.nsteps=3;
    p.memory_limit=512ULL<<20;p.device_memory_limit=512ULL<<20;p.B_min=1e-10;
    return p;
}
void wall_projection() {
    auto p=params();fomhd::Compute cpu;
    const auto g=fomhd::build_rectangle(3,3,1,0,0.2,0.2);
    const auto op=fomhd::build_operators(g,1);
    fomhd::Context c{p,g,op,cpu,p.Nphi};
    fomhd::Vec n(p.Nphi*op.Nc,1e12),r(p.Nphi*3*op.Nc);
    for(std::size_t j=0;j<r.size();++j)r[j]={std::sin(double(j)+0.2),std::cos(double(j))};
    fomhd::StageTag tag{0,1,1,0,0,"initial"};
    auto w=fomhd::build_wall(n,p.species[0],tag,c);
    auto v=fomhd::project_wall(w,r,c);
    require(fomhd::max_abs(fomhd::sample_apply(op.T_full,v.value,p.Nphi,cpu))<2e-12,"full wall rows");
    auto twice=fomhd::project_wall(w,v.value,c);
    require(fomhd::max_abs(fomhd::subtract(twice.value,v.value))<2e-12,"projection idempotence");
    fomhd::Complex pairing{};
    for(int l=0;l<p.Nphi;++l)for(std::size_t j=0;j<3*op.Nc;++j)
        pairing+=std::conj(v.value[l*3*op.Nc+j])*v.reaction[l*3*op.Nc+j]/w.samples[l].winv[j];
    require(std::abs(pairing)<1e-12*fomhd::norm(r)*fomhd::norm(r)/w.samples[0].winv[0],"weighted wall work");
}
void gate_negative() {
    fomhd::Check x; x.name="observed";x.available=true;x.passed=false;x.hard=false;
    fomhd::gate({x},{"observed"});
    bool missing=false;try{fomhd::gate({x},{"required"});}catch(const fomhd::Failure&){missing=true;}
    require(missing,"missing required observe must fail");
    x.hard=true;bool rejected=false;try{fomhd::gate({x});}catch(const fomhd::Failure&){rejected=true;}
    require(rejected,"failed hard gate must fail");
}
void scheduling() {
    auto p=params();p.start_time=3.0;p.dt=0.125;
    require(fomhd::time_at_step(p,3)==3.375,"step label");
    require(fomhd::time_at_midpoint(p,3)==3.4375,"mid label");
}
}
int main(int argc,char**argv) {
 try {
    const std::string which=argc>1?argv[1]:"all";
    if(which=="wall"||which=="all")wall_projection();
    if(which=="gate"||which=="all")gate_negative();
    if(which=="time"||which=="all")scheduling();
    std::cout<<"PASS "<<which<<'\n';return 0;
 } catch(const std::exception&e) {std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
