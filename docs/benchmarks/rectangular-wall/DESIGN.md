# Rectangular conductor and paired curl

## Physical boundary and stored fields

The outer four faces of the full uniform RZ cell rectangle are the
perfect conductor. The physical limits are the extreme cell centers
plus or minus half a cell width. The LCFS stays strictly inside the
box. It marks the profile and diagnostics. The finite halo has positive
density and pressure, and the supplied replay has zero resistivity.
The formulation uses electric and magnetic vector components. Faraday
advances the magnetic field, and the massless model obtains the electric
field from its Ohm and wall solves. The HYM simulations report a conducting
bounding box for their cylindrical-grid calculation. The link moments and coupled Hall/CGL-B wall
work here are the FOMHD construction. [Lestz, Belova and Gorelenkov, §II](https://arxiv.org/html/2101.05976#S2)

The relative Faraday complex omits the tangential electric wall
cochains, setting their trace to zero. The input stores two signed normal magnetic flux integrals per wall
segment, in Wb/rad, with outward orientation.
Each half-segment contributes to the adjacent dual vertex. If their sum
at vertex $v$ is $c_v$, the physical divergence is
$D_{B,h} B + T_3(c_v/w_v)$. Faraday evolves the interior cochains with
homogeneous cap rate. The normal-current constraint is enforced by the
normal-E reaction from the two impermeable species. The pressure
traction and wall power use the fixed normal magnetic area mean.

## Poloidal magnetic link norm

For an axial link use the weight $dZ$. For a radial link use $R\,dR$.
Let $x_i$ be the face location, $\bar x_i$ the centroid under that
weight and $w_i$ the integrated weight over the link. A link terminates
at the wall if its full cell-center interval would cross it. The
centroid of the radial link is
$\bar x_i=\overline R_i+\ell_i^2/(12\overline R_i)$.
On an axial first layer, the affine samples on the adjacent half and
full links are $3h/4$ apart. Applying the old graph factors and dividing
by $h$ yields a limiting slope of $383/576$.

FOMHD fixes the electric norm $H_E=M_1$ and retains the Faraday
topology. With $W=\operatorname{diag}(w_i)$ and $D$ the first difference
between adjacent link means, set

$$
\delta_i=\bar x_i-x_i,\qquad
s=\frac{\sum_i w_i\delta_i}{\sum_i w_i},\qquad
k_i=\frac{\sum_{j\le i}w_j(\delta_j-s)}{\bar x_{i+1}-\bar x_i},\qquad
H_{\mathrm{link}}=W+D^*\operatorname{diag}(k_i)D .
$$

The adjacent differences are ordered from lower to upper coordinate.
The coefficients $k_i$ are positive for the supplied full rectangular
box. For the axial line they are $h_Z/6$ at the first and last pair
and $h_Z/8$ between them. For a radial interval $[a,b]$, the weighted
centroid shift is $s=-h_R^2/[12(a+b)]$ and every numerator before the
last endpoint is positive. Thus the raw line norm is symmetric
positive definite. Its actions on constants and affine samples are

$$
H_{\mathrm{link}}\mathbf 1 = W\mathbf 1,\qquad
H_{\mathrm{link}}\bar x = W(x+s\mathbf 1).
$$

Differentiation removes the common shift. Holding the original
electric norm and adjusting a single finite wall collar would impose
incompatible moments. On the axial line a constant-preserving
symmetric correction has zero sum when applied to an affine sample.
The half-link displacement requires a nonzero sum in that collar.
The repaired raw norm uses compact nearest-neighbor coefficients
along the full line.

For $B_R$ the raw axial norm is multiplied by the transverse
$R h_R$ measure. For $B_Z$ the raw radial norm is multiplied by $h_Z$.
Apply these blocks in the stored chart by congruence with the inverse
graph factors, $H_B=T_2^{-*} H_{\mathrm{raw}} T_2^{-1}$. The toroidal
block uses its separate harmonic norm. The paired curl is

$$
A_h=H_E^{-1}C_h^* H_B,\qquad C_h=T_2 C_2 T_1^{-1}.
$$

The same $H_B$ defines magnetic energy and induction work. This gives
the electric-magnetic power pairing and preserves $D_{B,h}C_h=0$,
$D_J A_h=0$ and the fixed wall-normal flux. Restart fingerprints
include the repaired magnetic norm. These ideas draw on paired
mimetic curls and polynomial norm moments. The coefficients above are
for FOMHD's weighted half-link and full-link data. [Hyman and Shashkov, equations (3.9) and (5.1)–(5.12)](https://doi.org/10.1006/jcph.1999.6225) · [Lipnikov et al., equations (46)–(58)](https://doi.org/10.1016/j.jcp.2010.09.007) · [O’Reilly et al., equations (2)–(6)](https://pangea.stanford.edu/~edunham/publications/OReilly_etal_SG-acoustic_JCP17.pdf)

The two poloidal derivatives give affine-exact toroidal current
independently, including corners. The target orders for these terms are
one in the first two layers and two farther inside the domain. The old
Cartesian fourth-order interior limit is relinquished for these terms.
The nonlinear FreeGS verification is axisymmetric. Tests with $m=0,\pm1$
verify the discrete constraints and work, and general toroidal-field
near-wall accuracy remains a separate question.

[Benchmark report and verification scope](REPORT.md) ·
[Run the benchmark](README.md)
