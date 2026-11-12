# Wall-current resolution of the weak cold Hall reference

The same-box independent Ritz solve is cold, with a vacuum toroidal guide field $B_0=B_\phi(R)\mathbf e_\phi$, uniform mass density $\rho$, and $n_\phi=1$. The mode used here has nonzero frequency and the convention $e^{+i\omega t+i n_\phi\phi}$. The evolved A0 state uses finite positive pressure and its own constrained wall reaction.

## Strong momentum and the Ritz constraints

In the cold volume equations,

$$
i\omega\rho\,(u_R,u_Z)=B_\phi(-J_Z,J_R),
\qquad
i\omega\,(b_R,b_Z)=
\frac{i n_\phi B_\phi}{R}\left((u_R,u_Z)-\frac{(J_R,J_Z)}{e n_e}\right).
$$

The second expression follows from the poloidal components of $\nabla\times[(\mathbf u-\mathbf J/(en_e))\times B_0]$. At a radial face, $u_R=J_R=0$ and the first equation implies $J_Z=0$ if the strong volume equation extends continuously to that face. At an axial face, $u_Z=J_Z=0$ gives $J_R=0$ under the same assumption. The remaining poloidal velocity also vanishes under the cold force law, and the poloidal Faraday relation then gives $b_R=b_Z=0$ on each face. Ampère's law consequently requires $\partial_Z b_\phi=0$ and $\partial_R(Rb_\phi)=0$ there. Thus a smooth strong solution in this cold setting would have a constant boundary value and zero normal derivative of $Rb_\phi$. The derivation assumes the cold interior force balance reaches the wall and does not account for a separate reaction or pressure force in that limit.

The Ritz momentum equation pairs the force with poloidal velocity test functions whose normal component vanishes at the wall. Its conserved quadratic norm controls $\mathbf u$ and $\mathbf b$ in volume $L^2$ and provides no independent control of the velocity trace or of $\nabla\times\mathbf b$. The magnetic trial space imposes the homogeneous relative potential and $J_n=0$. It leaves the tangential poloidal current free, so the weak solution can retain a normal Lorentz force at a face. For a toroidal guide field, $J_\phi$ drops out of the Lorentz force and both current arguments in the skew Hall bilinear form. A shrinking layer in the poloidal magnetic components can therefore change their curl much more than their volume magnetic norm.

Related Hall spectral work derives additional smooth wall trace relations in a warm one-dimensional slab. The slab uses its own equilibrium and wall conditions. Structure-preserving incompressible Hall finite elements likewise distinguish ideal energy conservation from the dissipative control of current. [De Jonghe et al., Legolas, §3.3.1](https://doi.org/10.1017/S0022377822000617) · [Laakmann, Hu and Farrell, §§1 and 2](https://arxiv.org/abs/2202.11586)

## Direct evaluation of the reference branch

The integration splits the 1.20 m square cross-section into a central rectangle, four 5 cm face strips, and four 5 cm corner squares. Tensor Gauss quadrature integrates each part with the cylindrical $R$ weight. The division is at the exact physical collar edge. The order-18 integral metrics agreed between 22 and 24 nodes per patch direction within relative tolerance $10^{-6}$ and absolute tolerance $10^{-8}$. The magnetic and current norms are volume $L^2$ norms. The Lorentz-force residual is $i\omega\rho\,\mathbf u-\mathbf J\times B_0$ evaluated from the fields, without applying the Ritz momentum projection.

| Ritz order | $\mu_0\|J\|_2/\|b\|_2$ (m⁻¹) | Total squared current on face strips | In corner squares | Face-strip strong momentum residual / Lorentz force | Core residual / Lorentz force |
|---:|---:|---:|---:|---:|---:|
| 12 | 6.694 | 31.55% | 0.68% | 53.90% | 5.00% |
| 14 | 7.494 | 43.74% | 1.08% | 55.73% | 4.16% |
| 16 | 8.597 | 56.03% | 1.44% | 57.66% | 3.63% |
| 18 | 9.991 | 66.64% | 1.76% | 58.83% | 3.38% |

At order 18, the toroidal component accounts for 97.52% of the face-strip squared current. The normalization $\mu_0\|J_\perp\|_2/\|b\|_2$ changes from about 5.73 to 5.71 m⁻¹ across orders 12 to 18. The corresponding toroidal value changes from 3.47 to 8.20 m⁻¹. At order 18, the face strips contain 5.06% of the squared poloidal current and the corners contain 0.058%. The walls have normal current at most $7.94\times 10^{-11}$ times the peak tangential poloidal wall current. That tangential current gives a wall normal Lorentz force whose RMS on the perimeter is 1.27 times the force RMS in the volume. The trace check and the strong momentum residual measure different properties.

The growth in the derivative norm, its dominance by toroidal current, and its location along the faces persist while the order-16 to order-18 frequency changes by 0.079%. This supports using the independent solve to identify and compare the compressional branch and its broad magnetic structure. It leaves the limiting current profile and current-layer width open. It also leaves open how a matched finite-pressure wall reference would change the A0 frequency and near-wall comparison. The observed current concentration alone does not attribute the full discrepancy to the A0 wall closure or establish a physical current sheet.

[Raw metrics](wall-current-weak-trace-audit.json) · [Quadrature refinement](wall-current-quadrature-check.json) · [Measurement script](wall_current_audit.py)
