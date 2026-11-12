# Cold Hall standing wave in the rectangular RZ domain

This is an independent linear reference with an axial field, uniform singly
charged deuterium density, zero equilibrium flow, and zero pressure. The
analytic model includes the Hall term and massless quasineutral electrons.
It retains the background normal magnetic flux through the two endcaps.

## Reference equations

Use cylindrical `(R, phi, Z)` components and complex time factor
`exp(-i omega t)`. Write the perturbed field and Ampère current as

$$
\delta\mathbf B=\frac{(-\psi_Z,b,\psi_R)}{R},
\qquad
\delta\mathbf J=\frac{(-b_Z,-\Delta^*\psi,b_R)}{\mu_0R},
\qquad
\Delta^*=\partial_R^2-R^{-1}\partial_R+\partial_Z^2 .
$$

For equilibrium field $B_0\mathbf e_Z$ and mass density $\rho_0=m_i n_0$,

$$
\delta u_{i,R}=\frac{-i B_0\Delta^*\psi}{\omega\mu_0\rho_0R},
\quad
\delta u_{i,\phi}=\frac{i B_0 b_Z}{\omega\mu_0\rho_0R},
\quad
\delta u_{i,Z}=0,
\quad
\delta n=-\frac{i n_0}{\omega R}\partial_R(R\delta u_{i,R}).
$$

The electron velocity and field follow from
$\delta\mathbf u_e=\delta\mathbf u_i-\delta\mathbf J/(e n_0)$ and
$\delta\mathbf E=-\delta\mathbf u_e\times B_0\mathbf e_Z$.
With $D=B_0/(\mu_0 e n_0)=v_A d_i$, Faraday gives

$$
\dot\psi=-B_0R\delta u_{i,R}-D b_Z,
\qquad
\dot b=B_0R\partial_Z\delta u_{i,\phi}
       + D\partial_Z\Delta^*\psi .
$$

Take $\psi=f(R)P(Z)$ and $b=f(R)Q(Z)$, with $f$ the first radial Dirichlet
eigenfunction of $\partial_R^2-R^{-1}\partial_R$ and eigenvalue
$-\kappa^2$. A single axial component uses $P_j=A_j\sin(k_j (Z-Z_{lo}))$ and
$Q_j=A_jq_j\cos(k_j (Z-Z_{lo}))$, where

$$
q_j=\frac{\omega^2-v_A^2(k_j^2+\kappa^2)}{i\omega D k_j},
\qquad
(\xi-k_j^2-\kappa^2)(\xi-k_j^2)=\xi d_i^2 k_j^2(k_j^2+\kappa^2),
\quad \xi=\omega^2/v_A^2.
$$

For two same-parity axial indices $k_j=m_j\pi/(Z_{hi}-Z_{lo})$, set
$X_j=k_j^2$, $S=X_1+X_2+\kappa^2$,
$\xi=\kappa^2+2X_1X_2/S$, and $d_i^2=(1-2\xi/S)/\xi$. This tunes the density
so the two components have one frequency. Choose $A_1q_1+A_2q_2=0$. Each
radial side has $f=0$. Both caps have $P=P''=Q=Q'=0$. These traces give
$\delta B_n=0$, $\delta\mathbf E_t=0$, and zero species normal velocity and
normal current on each side. At a corner each of the two incident side
normals is checked separately. The axial background field has
$B_n=\pm B_0$ on the two endcaps.

## Sampling and independent check

```python
from axial_hall_box import make_mode, wave

mode = make_mode()
initial = wave(R, Z, phase=0.5 * np.pi, amplitude=1e-4, mode=mode)
# Vectors use the last axis for (R, phi, Z).
# initial["psi"] is the total poloidal streamfunction in Wb/rad.
```

The amplitude is the peak envelope of $|\delta B_\phi|/B_0$. With default
$R=[2.4,3.6]$ m, $Z=[-0.6,0.6]$ m, $B_0=2$ T and $(m_1,m_2)=(1,3)$, the
density is $3.62113\times10^{18}\,\mathrm{m}^{-3}$. The angular frequency is
$6.90610\times10^7\,\mathrm{s}^{-1}$, the period is $90.9803$ ns, and
$d_i=0.169188$ m. The density follows from the box and mode numbers. Changing
the box without retuning requires either a different matching pair or a new
mode calculation.

The cold pressure perturbations are zero. At phase $\pi/2$ the cold density
perturbation also vanishes, as do the passive linear pressure perturbations
that would arise from constant positive background pressures evaluated at that
same phase. A finite-pressure FOMHD comparison can initialize each pressure
to a constant above its configured floor, for example 10 Pa per species if
the floors permit it. The cold prediction then has an independent finite-
pressure error that should be measured. For 10 Pa per species the background
total pressure ratio $2\mu_0(p_i+p_e)/B_0^2$ is about $1.26\times10^{-5}$.

`verify_axial_hall_box.py` independently differences sampled physical
fields in cylindrical coordinates. It checks divergence, Ampère, Faraday,
ion momentum, both continuity equations, phase, normalization, all wall traces,
and cold linear wave energy across five phases. The retained `verification.json` shows about second-order residual
decrease across three step sizes. The velocity reconstruction and Ohm checks
record consistency with the defining algebra. The independent differential
checks use the public samples and do not call the analytic profile
derivatives. This directory contains the continuum reference and its checks.
Solver evolution, finite-pressure effects, and agreement across grids remain
separate measurements.
