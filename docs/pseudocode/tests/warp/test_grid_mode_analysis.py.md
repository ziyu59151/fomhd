# Grid-mode analysis regressions

Test the nodal energy overlap for complex phase and magnitude invariance.
Test partial and disjoint `LinearNDInterpolator` support, exact `M0` coverage,
and absence of extrapolated values. Reject comparisons differing in family,
mode, or physical reference units. Reject candidate arrays whose dimensions
disagree with the cell count. Check frequency/index selection and that source
file hashes remain unchanged. Verify magnetic-transfer-ratio and true
staggered-MB/ion-kinetic norm metadata survive selection; label overlaps as
structure comparisons rather than physical total wave energy.

Run ordinary scoring with no previous/selection and require a completed result.
For a regular source grid with one missing active center, require the target at
that absent source cell to be excluded even when the convex hull interpolator
returns a finite value. Check cold and continuation rankings independently with
one ambiguous score pair and one distinct pair; ambiguity must remain attached
to the correct metric.
