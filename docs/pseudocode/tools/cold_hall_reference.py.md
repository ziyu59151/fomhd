# Independent cold Hall reference

## A0-CAE-REF-01 Trial fields and wall constraints

Construct polynomial velocity fields on a circular poloidal disk with zero
normal velocity. Construct relative vector potentials with zero tangential
trace, including bubble*(-z,0,x)/R to retain the admissible constant R*b_phi
wall trace. Apply the analytic cylindrical Fourier curl twice to obtain b and
J. Integrate with independent Gauss-radial/uniform-angle quadrature and physical
R*r volume measure. Eliminate J-normal boundary rows by nullspace, then remove
mass/gauge redundancy by weighted SVD. Check a shifted independent wall grid.

## A0-CAE-REF-02 Weak generator and mode selection

Whiten velocity mass rho and magnetic mass 1/mu0. Assemble L=integral(v*.Jxb0)
and H=-integral(J*.Jxb0)/(e*n). Use generator [0,L;-L.H,H] and solve the
Hermitian problem -i*A*z=omega*z, with exp(+i*omega*t) time convention.
Retain physical zero modes in metadata and select the lowest positive finite
frequency with magnetic parallel fraction above one half. Return all candidate
frequencies/fractions, rank and skew residual, plus coefficients sufficient for
independent evaluation at arbitrary production entities. No production solver
operator or RHS is imported. This is the constrained weak wall model, not an
assertion of smooth Hall boundary equivalence or a reproduction of published
CAE frequencies. Positive-pressure CGL comparisons are cold-limit comparisons.
