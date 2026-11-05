#include "fomhd/io.hpp"
#include "fomhd/diagnostics.hpp"
#include "fomhd/backend.hpp"
#include "support.hpp"
#include <iostream>
using namespace fomhd;
int main(int argc, char** argv) {
  try {
    if(argc<2||argc>6)throw std::runtime_error("Usage: fomhd_export_native NEW.npz [POINTS=4] [K=1] [VELOCITY=0.01] [STRESS=0]");
    const auto path=std::filesystem::absolute(argv[1]);
    if(std::filesystem::exists(path))throw std::runtime_error("output already exists");
    const int points=argc>2?std::stoi(argv[2]):4;
    if(points<4||points>20)throw std::runtime_error("point count must be 4..20");
    auto p=test::parameters(argc>3?std::stoi(argv[3]):1);
    const double velocity=argc>4?std::stod(argv[4]):.01;
    const bool stress=argc>5?std::stoi(argv[5])!=0:false;
    if(stress)p.dt=1e-12;
    auto g=build_rectangle(points,points,1,0,.2,.2);
    auto op=build_operators(g,p.K,{p.rank_rtol,p.memory_limit});
    Compute cpu; Context c{p,g,op,cpu,p.Nphi};
    auto u=closed_reference(c,p.ref.ni,p.ref.pressure,p.ref.pressure,p.ref.magnetic,velocity);
    if(stress) {
      for(int a=0;a<2;++a)for(std::size_t j=0;j<op.Nc;++j) {
        const double shape=std::sin(2.3*g.cells.R[j]+1.7*g.cells.Z[j]);
        u.fluid[a].n[p.K*op.Nc+j]*=1+.2*shape;
        u.fluid[a].perp[p.K*op.Nc+j]*=1+.1*shape;
        if(a)u.fluid[a].parallel[p.K*op.Nc+j]*=.8+.15*shape;
        if(p.K)for(int k:{-1,1})u.fluid[a].perp[(p.K+k)*op.Nc+j]=p.ref.pressure*1e-5*(1+.1*shape);
      }
    }
    Cumulative cumulative; StageResult stage;
    auto committed=initialize(u,c,cumulative,velocity==0&&!stress,&stage);
    io::Arrays out;
    const auto vector=[&](const std::string& name,const Vec& x){out[name]=io::from_complex(x,{x.size()});};
    const auto real=[&](const std::string& name,const RealVec& x){out[name]=io::from_doubles(x,{x.size()});};
    const auto matrix=[&](const std::string& name,const Matrix& A){
      Vec row_major(A.data.size());
      for(std::size_t r=0;r<A.rows;++r)for(std::size_t j=0;j<A.cols;++j)row_major[r*A.cols+j]=A(r,j);
      out[name]=io::from_complex(row_major,{A.rows,A.cols});
    };
    const auto state=[&](const std::string& prefix,const State& x){
      vector(prefix+"_B",x.B);
      for(int a=0;a<2;++a){const auto n=prefix+"_"+std::to_string(a);vector(n+"_n",x.fluid[a].n);vector(n+"_mom",x.fluid[a].mom);vector(n+"_perp",x.fluid[a].perp);vector(n+"_parallel",x.fluid[a].parallel);}
    };
    real("R",g.R);real("Z",g.Z);real("M0",g.M0);real("M1",g.M1);real("MB",g.MB);real("wv",g.wv);real("cell_R",g.cells.R);real("cell_Z",g.cells.Z);
    for(const auto& item:std::vector<std::pair<std::string,const Matrix*>>{{"I",&op.I},{"H",&op.H},{"QB",&op.QB},{"DR",&op.DR},{"DZ",&op.DZ},{"T",&op.T},{"T_full",&op.T_full},{"Bc",&op.Bc},{"Bc_full",&op.Bc_full},{"topology_flux_row",&op.topology_flux_row},{"PR_r",&op.PR_r},{"PZ_r",&op.PZ_r},{"PR_z",&op.PR_z},{"PZ_z",&op.PZ_z},{"PR_v",&op.PR_v},{"PZ_v",&op.PZ_v}})matrix(item.first,*item.second);
    for(const auto& item:std::vector<std::pair<std::string,const std::vector<Matrix>*>>{{"G",&op.G},{"C",&op.C},{"DB",&op.DB},{"DJ",&op.DJ},{"Cdag",&op.Cdag},{"Lphi",&op.Lphi}})
      for(std::size_t k=0;k<item.second->size();++k)matrix(item.first+"_"+std::to_string(k),(*item.second)[k]);
    RealVec wall_edges,wall_signs;
    for(auto w:g.walls){wall_edges.push_back(w.component==WallComponent::r_face?w.face:g.r_faces.keys.size()+g.cells.keys.size()+w.face);wall_signs.push_back(w.sign);}
    real("wall_edges",wall_edges);real("wall_signs",wall_signs);
    state("state",u);state("rhs",stage.rhs);
    vector("E",stage.cache.E);vector("EL",stage.cache.EL);vector("ET",stage.cache.ET);vector("phi",stage.cache.phi);vector("current",stage.current);vector("rho",stage.rho);vector("fJ0",stage.fJ0);vector("BN",stage.BN);vector("bhat",stage.bhat);vector("ELdot",stage.ELdot);matrix("AE",stage.AE);
    for(const auto& [key,value]:stage.residuals)vector("residual_"+key,value);
    for(int a=0;a<2;++a){auto prefix="fluid_"+std::to_string(a)+"_";auto& f=stage.fluid[a];
      for(const auto& item:std::vector<std::pair<std::string,const Vec*>>{{"u",&f.u},{"flux",&f.flux},{"flux_hat",&f.flux_hat},{"nf",&f.nf},{"tensor",&f.tensor},{"gradient",&f.gradient},{"theta",&f.theta},{"strain",&f.strain},{"convection",&f.convection},{"pressure_force",&f.pressure_force},{"non_electric",&f.non_electric},{"pdot_perp",&f.pdot_perp},{"pdot_parallel",&f.pdot_parallel}})vector(prefix+item.first,*item.second);
      auto& l=stage.ledger[a];real("ledger_"+std::to_string(a),{l.electric_node,l.electric_edge,l.pressure_kinetic,l.pressure_internal,l.magnetic,l.wall,l.convection,l.perp_rate,l.parallel_rate,l.spectral_kinetic,l.spectral_perp,l.spectral_parallel});
    }
    auto result=advance(committed,cumulative,c);gate(result.checks);state("endpoint",committed.state);
    io::Json config={{"schema",1},{"points",points},{"K",p.K},{"Nphi",p.Nphi},{"velocity",velocity},{"dt",p.dt},{"stress",stress},{"reference_commit","48beca7"},{"hard_checks",stage.checks.size()}};
    out["metadata_json"]=io::from_utf8(config.dump());
    io::write_npz(path,out);std::cout<<config.dump()<<"\nexported_arrays="<<out.size()<<'\n';
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
