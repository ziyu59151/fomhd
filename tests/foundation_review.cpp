#include "fomhd/operators.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
using namespace fomhd;
void require(bool value,const char* text) {if(!value)throw std::runtime_error(text);}
template<class F> void rejects(F f) {bool failed=false;try{f();}catch(const NumericalError&){failed=true;}require(failed,"invalid input was accepted");}
Geometry irregular(bool notch) {
    RealVec R,Z;for(int i=0;i<8;++i)R.push_back(1.+i*.125);
    for(int i=0;i<8;++i)Z.push_back(i*.125);
    if(notch)return build_geometry(R,Z,{{1,0},{1.875,0},{1.875,.875},{1.625,.875},
        {1.625,.375},{1.5,.375},{1.5,.875},{1,.875}});
    return build_geometry(R,Z,{{1,0},{1.875,0},{1.875,.375},{1.375,.375},{1.375,.875},{1,.875}});
}
void component_probes(const Geometry& g) {
    auto o=build_operators(g,0);auto nc=o.Nc,nz=g.z_faces.keys.size(),nv=g.vertices.keys.size();
    const std::size_t first[3]={0,nz,nz+nv},last[3]={nz,nz+nv,o.MB.size()};
    for(int a=0;a<3;++a) {
        Vec b(o.MB.size());for(auto i=first[a];i<last[a];++i)b[i]=1;
        auto n=matvec(o.QB,b);
        for(int d=0;d<3;++d)for(std::size_t j=0;j<nc;++j)
            require(std::abs(n[d*nc+j]-(a==d?1.:0.))<2e-14,"QB component or fallback constant reproduction");
    }
    for(std::size_t cell=0;cell<nc;++cell) {
        auto key=g.cells.keys[cell];bool connected=true;
        for(int di=-1;di<=1;++di)for(int dj=-1;dj<=1;++dj)
            if(int(key.first)+di<0||int(key.second)+dj<0||
               !g.is_active(int(key.first)+di,int(key.second)+dj))connected=false;
        if(!connected) {
            for(std::size_t v=0;v<nv;++v) {
                auto vk=g.vertices.keys[v];
                bool corner=(vk.first==key.first||vk.first==key.first+1)&&
                    (vk.second==key.second||vk.second==key.second+1);
                require(o.QB(nc+cell,nz+v)==Complex(corner?.25:0),"QB stencil crossed exterior support");
            }
        }
    }
}
int main(int argc,char**argv) {
 try {
    std::string test=argc>1?argv[1]:"";
    if(test=="components")component_probes(build_rectangle(5,5,1,0,.125,.125));
    else if(test=="fallback")component_probes(irregular(false));
    else if(test=="notch")component_probes(irregular(true));
    else if(test=="magnetic_metric") {
        auto g=build_rectangle(5,5,1,0,.125,.125);
        auto offset=g.z_faces.keys.size()+g.vertices.keys.size();
        for(std::size_t j=0;j<g.r_faces.keys.size();++j) {
            double R=g.r_faces.R[j];auto k=g.r_faces.keys[j];
            double length=.125;
            if(k.first==0){R+=.03125;length*=.5;}
            if(k.first==4){R-=.03125;length*=.5;}
            require(std::abs(g.MB[offset+j]-R*.125*length)<1e-16,"magnetic half-dual centroid weight");
        }
    } else if(test=="norms") {
        require(std::abs(weighted_norm({{.1,.3}},{.1})-.1)<2e-17,"complex weighted norm");
        require(std::abs(norm({1e-200})/1e-200-1)<1e-15,"small stable norm");
        require(std::abs(norm({1e200})/1e200-1)<1e-15,"large stable norm");
        rejects([]{(void)max_abs(Vec{std::numeric_limits<double>::quiet_NaN()});});
    } else if(test=="rank_units") {
        Matrix a(2,2);a(0,0)=1e-200;a(1,1)=1e200;
        require(independent_rows(a,1e-12).size()==2,"rank changed under row units");
    } else if(test=="grid") {
        auto g=build_rectangle(3,3,1,0,.1,.1);
        auto r=g.R;r[2]=r[1];rejects([&]{(void)build_geometry(r,g.Z,g.polygon_input);});
        GeometryOptions o;o.grid_rtol=std::numeric_limits<double>::quiet_NaN();
        rejects([&]{(void)build_geometry(g.R,g.Z,g.polygon_input,o);});
    } else if(test=="ambiguous") {
        rejects([]{(void)build_geometry({1.0,1.1,1.2},{0.,.1,.2},
            {{.9,-.1},{1.3,-.1},{1.3,.3},{1.0,0.0}});});
    } else throw std::runtime_error("unknown test");
    std::cout<<"PASS "<<test<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
