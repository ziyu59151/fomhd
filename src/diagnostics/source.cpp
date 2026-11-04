#include "fomhd/verification.hpp"
#include "fomhd/backend.hpp"
#include <cmath>
#include <numbers>

namespace fomhd::verification {
void check_source(const State& source, const Context& c) {
  auto scales=c.p;
  scales.ref.ni/=c.p.ref.time;
  scales.ref.ne/=c.p.ref.time;
  scales.ref.pressure/=c.p.ref.time;
  scales.ref.magnetic/=c.p.ref.time;
  Context rate_context{scales,c.geometry,c.op,c.compute,c.N};
  check_retained(source,rate_context);
  const double bound=1e-9*c.p.ref.magnetic/(c.p.ref.length*c.p.ref.time);
  if(max_abs(mode_apply(c.op.DB,source.B,c.compute))>bound)
    throw Failure("ManufacturedSource","magnetic source must preserve divergence");
}

SourceRates source_rates(const State& source, const State& midpoint,
                         const StageResult& frame, const Context& c) {
  check_source(source,c);
  const auto nc=c.op.Nc;
  const double tau=2*std::numbers::pi;
  SourceRates out;
  Vec rho(source.fluid[0].n.size());
  std::array<Vec,2> number,momentum;
  for(int a=0;a<2;++a) {
    rho=add(rho,scaled(source.fluid[a].n,c.p.species[a].charge));
    number[a]=inverse(source.fluid[a].n,nc,c);
    momentum[a]=inverse(source.fluid[a].mom,3*nc,c);
    auto pressure=inverse(source.fluid[a].perp,nc,c);
    auto parallel=a?inverse(source.fluid[a].parallel,nc,c):Vec{};
    const auto& f=frame.fluid[a];
    for(int l=0;l<c.N;++l)for(std::size_t j=0;j<nc;++j) {
      const auto q=l*nc+j;
      const double w=tau*c.op.M0[j]/c.N;
      out.totals.number[a]+=w*number[a][q].real();
      double work=0,speed2=0;
      for(int d=0;d<3;++d) {
        const auto i=(l*3+d)*nc+j;
        work+=f.u[i].real()*momentum[a][i].real();
        speed2+=std::norm(f.u[i]);
      }
      out.totals.kinetic[a]+=w*(work-.5*c.p.species[a].mass*speed2*number[a][q].real());
      if(a) {
        out.totals.electron_perp+=w*pressure[q].real();
        out.totals.electron_parallel+=w*parallel[q].real()/2;
      } else out.totals.ion_internal+=w*pressure[q].real()/(c.p.species[a].gamma-1);
    }
  }
  out.longitudinal_rate=electrostatic(rho,frame.tag,c).EL;
  for(std::size_t j=0;j<source.B.size();++j)
    out.totals.magnetic+=tau*c.op.MB[j%c.op.Nb]*std::real(std::conj(midpoint.B[j])*source.B[j])/c.p.mu0;
  for(std::size_t j=0;j<frame.cache.EL.size();++j)
    out.totals.longitudinal+=tau*c.p.epsilon0*c.op.M1[j%c.op.Ne]*
      std::real(std::conj(frame.cache.EL[j])*out.longitudinal_rate[j]);
  out.totals.topology=c.compute.mv(c.op.topology_flux_row,slice(source.B,c.p.K*c.op.Nb,c.op.Nb))[0].real();
  for (std::size_t v = 0; v < c.geometry.av.size(); ++v)
    out.totals.physical_flux += c.geometry.av[v] *
        source.B[c.p.K*c.op.Nb+c.geometry.z_faces.keys.size()+v].real();
  out.totals.energy=out.totals.kinetic[0]+out.totals.kinetic[1]+out.totals.ion_internal+
    out.totals.electron_perp+out.totals.electron_parallel+out.totals.magnetic+out.totals.longitudinal;
  const auto E=sample_apply(c.op.H,inverse(frame.cache.EL,c.op.Ne,c,true,c.p.ref.electric),c.N,c.compute);
  const auto dE=sample_apply(c.op.H,inverse(out.longitudinal_rate,c.op.Ne,c,true,c.p.ref.electric/c.p.ref.time),c.N,c.compute);
  const auto dB=inverse(sample_apply(c.op.QB,source.B,2*c.p.K+1,c.compute),3*nc,c,true,c.p.ref.magnetic/c.p.ref.time);
  const int samples=std::max(c.N,3);
  out.momentum.angular_samples=samples;
  for(int q=0;q<samples;++q) {
    const int l=c.N==1?0:q;const double phi=tau*q/samples;
    for(std::size_t j=0;j<nc;++j) {
      std::array<double,3> rate{};
      for(int d=0;d<3;++d) {
        const auto i=(l*3+d)*nc+j,ia=(l*3+(d+1)%3)*nc+j,ib=(l*3+(d+2)%3)*nc+j;
        rate[d]=momentum[0][i].real()+momentum[1][i].real()+c.p.epsilon0*
          (dE[ia].real()*frame.BN[ib].real()-dE[ib].real()*frame.BN[ia].real()+
           E[ia].real()*dB[ib].real()-E[ib].real()*dB[ia].real());
      }
      const double w=tau*c.op.M0[j]/samples;
      out.momentum.value[0]+=w*(rate[0]*std::cos(phi)-rate[1]*std::sin(phi));
      out.momentum.value[1]+=w*(rate[0]*std::sin(phi)+rate[1]*std::cos(phi));
      out.momentum.value[2]+=w*rate[2];
      out.momentum.angular+=w*c.geometry.cells.R[j]*rate[1];
    }
  }
  return out;
}
} // namespace fomhd::verification
