# Prescribed-source diagnostic rates

## A0-VERIFY-SOURCE-02 — Admissibility

Check finite retained shapes and Hermitian symmetry using the production
validator. Momentum rates obey the full homogeneous wall trace; use rate scales
`n_ref/t_ref`, `p_ref/t_ref`, `B_ref/t_ref`, preserving velocity units. Verify
`DB*S_B=0` against `1e-9*B_ref/(L_ref*t_ref)`. Do not enforce positivity on rates
or project an invalid supplied source to make it pass.

## A0-VERIFY-SOURCE-03 — Chain-rule accounting

At the actual numerical midpoint, reconstruct physical samples with the same
Fourier normalization and spatial weights as the normal diagnostics:

```text
for species a:
    number_rate[a] = integral(S_n[a])
    kinetic_rate[a] = integral(u[a] dot S_mom[a]
                              - m[a]*|u[a]|^2*S_n[a]/2)
ion_internal_rate = integral(S_p_i)/(gamma_i-1)
electron_perp_rate = integral(S_p_perp_e)
electron_parallel_rate = integral(S_p_parallel_e)/2
S_rho = sum_a q[a]*S_n[a]
S_EL = electrostatic(S_rho).EL
magnetic_rate = 2*pi*Re(sum(MB*conj(B)*S_B))/mu0
longitudinal_rate = 2*pi*epsilon0*Re(sum(M1*conj(EL)*S_EL))
energy_rate = sum(all material and field energy rates)
topology_rate = topology_flux_row*S_B[k=0]
physical_flux_rate = sum_v av[v]*S_B_phi[k=0,v]
momentum_density_rate = sum_a S_mom[a]
    + epsilon0*(H*S_EL cross QB*B + H*EL cross QB*S_B)
Cartesian momentum_rate = integrate rotated cylindrical components
angular momentum_rate = integral(R*momentum_density_rate_phi)
```

The momentum angular quadrature uses `max(Nphi,3)` and broadcasts the axisymmetric
sample when Nphi=1. Report that sample count. The physical current/flux producer
and stage field equations remain unchanged; these are external manufactured
rates used only by interval verification. A centered finite difference of
production totals independently verifies every component in `source_controls`.
