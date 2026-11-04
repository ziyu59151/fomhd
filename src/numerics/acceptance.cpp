#include "fomhd/operators.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace fomhd {
std::vector<OperatorCheck> validate_operators(const Geometry& g,const Operators& o) {
    std::vector<OperatorCheck> checks;
    const auto check=[&](std::string name,int mode,const Vec& residual,double scale,std::size_t terms,std::string unit){
        OperatorCheck x{std::move(name),std::move(unit),mode,max_abs(residual),scale,
            128*std::numeric_limits<double>::epsilon()*std::max<std::size_t>(1,terms)*scale};
        if(!std::isfinite(x.scale)||x.scale<0||x.residual>x.roundoff_bound)
            throw NumericalError("generated operator acceptance failed: "+x.name+" mode="+std::to_string(mode));
        checks.push_back(std::move(x));
    };
    for(int k=-o.K;k<=o.K;++k) {
        const auto i=static_cast<std::size_t>(k+o.K);
        check("CG",k,matmul(o.C[i],o.G[i]).data,max_abs(o.C[i])*max_abs(o.G[i]),o.Ne,"1/m^2");
        check("DBC",k,matmul(o.DB[i],o.C[i]).data,max_abs(o.DB[i])*max_abs(o.C[i]),o.Nb,"1/m^2");
        Matrix adj=adjoint(o.G[i]);
        Vec residual(adj.data.size());
        for(std::size_t c=0;c<o.Ne;++c)for(std::size_t r=0;r<o.Nc;++r)
            residual[r+c*o.Nc]=o.M0[r]*o.DJ[i](r,c)+adj(r,c)*o.M1[c];
        check("gradient_divergence_adjoint",k,residual,max_abs(o.G[i])*(*std::max_element(o.M1.begin(),o.M1.end())),2,"m^2");
    }
    Vec div(o.Ne);
    for(std::size_t j=0;j<o.Ne;++j)for(std::size_t i=0;i<o.Nc;++i)div[j]+=o.M0[i]*o.DJ[o.K](i,j);
    double surface=0;
    for(auto w:g.walls) {
        const auto edge=w.component==WallComponent::r_face?w.face:g.r_faces.keys.size()+o.Nc+w.face;
        div[edge]-=w.sign*w.area_per_radian;surface=std::max(surface,w.area_per_radian);
    }
    check("divergence_theorem",0,div,surface,o.Nc,"m^2");
    Vec transfer(o.H.data.size());
    for(std::size_t e=0;e<o.Ne;++e)for(std::size_t j=0;j<3*o.Nc;++j)
        transfer[j+e*3*o.Nc]=o.M0[j%o.Nc]*o.H(j,e)-std::conj(o.I(e,j))*o.M1[e];
    check("transfer_adjoint",0,transfer,max_abs(o.I)*(*std::max_element(o.M1.begin(),o.M1.end())),2,"m^3");
    check("topology_curl",0,matmul(o.topology_flux_row,o.C[o.K]).data,
          max_abs(o.topology_flux_row)*max_abs(o.C[o.K]),o.Nb,"m");
    return checks;
}
} // namespace fomhd
