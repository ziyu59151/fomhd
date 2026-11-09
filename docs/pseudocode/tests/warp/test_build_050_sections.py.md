# test_build_050_sections.py pseudocode

Load `build_050_sections.py` from the sibling test layout or repository
`tools/plots` layout.

Exercise the bounded behavioral contract:

1. Nonrectangular active cells produce one quadrilateral per supplied center
   and no cell for an unselected bounding-grid location.
2. Vector samples use `(Nphi, component R/phi/Z, Nc)` layout and select phi
   index zero.
3. `perturbed - control` has the expected sign.
4. Compared delta arrays receive one centered shared limit.
5. Step/time mismatches raise an explicit error.
6. Different active grids raise an explicit error.
7. Existing outputs and outputs nested in either input root are refused.
8. Large imaginary components at phi zero are refused as non-physical-real
   samples.

No test runs the solver or writes outside pytest temporary directories.
