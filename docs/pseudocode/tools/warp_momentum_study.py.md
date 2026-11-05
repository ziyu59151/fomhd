# Fixed-domain finite-inertia momentum evidence

This harness uses the production finite-electron-mass stage and RK4 transaction.
It changes neither tolerances nor wall conditions. CPU setup and plotting are
allowed; CUDA execution requires an owned Slurm allocation.

1. Build explicit cell centers inside a fixed polygon. The static pressure
   control uses [1,2] x [0,1], either rectangle or a fixed 128-sided disk.
   The evolved control uses the original numerical rectangle [0.9,1.5] x
   [-0.1,0.5]. Changing the number of cells never changes those bounds.
2. Construct constant neutral densities and positive pressures. Normalize the
   discrete harmonic magnetic field to one fixed toroidal flux, not a
   mesh-dependent maximum reconstructed field.
3. Static control: prescribe p_e=p0+A cos(phi), u_i=u_e=0. Evaluate the complete
   finite-mass stage. Integrate cylindrical vector components in Cartesian
   coordinates with M0 and uniform phi quadrature. Include the derivative of
   epsilon0 E_L cross B. Compare dP_x/dt and outward physical traction against
   -pi A area_h and +pi A area_h. Report smooth-domain area separately.
4. Evolved control: add the fixed physical electron-density envelope
   1e-6 cos^2(pi(R-1.2)) cos^2(pi(Z-0.2)) cos(phi), keep ion density unchanged,
   and solve the normal initial constraints. Advance RK4 at fixed final time
   with dt_guard=error. Retain every momentum budget scalar including passes,
   rejected attempts, accepted step count, and exact initial/final arrays.
5. Quiet control: omit the seed and retain initial RHS and state drift. This
   tests a simple discrete reference state, not a GS/two-fluid Tokamak solver.
6. Report the endpoint semidiscrete total momentum rate, physical traction,
   their signed defect, and numerical constraint reaction separately. Never
   add the latter to the physical traction to manufacture a passing result.
7. Snapshot actual imported Python source and harness inputs. A failed case
   remains failed and does not erase evidence from other cases. Refinement
   establishes a trend only for this fixed family; the massless Ohm MMS is a
   separate experiment.
