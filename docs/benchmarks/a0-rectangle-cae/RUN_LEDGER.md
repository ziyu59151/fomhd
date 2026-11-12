# A0 mode campaign provenance and trial ledger

All GPU work used owned single-card RTX 5090 Slurm allocations.
The long A0 runs use the frozen source at commit `803b332ce45750b8c6e759f7405ba76ed87ee213`.
Commit `95e5b56f23be59f46bbf6f47206d23b23c34645c` restores two extreme-value
linear-solve gates. Its separate GPU replay on a representative normal
nonzero-mode input agrees with the measured source's primary fields
and hard-check records at the six-step endpoint. The original and
provisional optimized pilots preserve separate source captures. The
eigensolver and sampling source are archived under this campaign.
[Scaling source and guarded replay](https://github.com/ziyu59151/fomhd/pull/4)

| Allocation | Scope |
|---|---|
| 11542 | Nominal eight-cell pilot, non-Hermitian input rejected. The missing conjugate of the global seed phase was corrected in the case builder. |
| 11544 | Corrected nominal eight-cell baseline, four accepted RK4 steps. |
| 11545, 11547 | Axial cold Hall pilot at 1 ns and 0.25 ns rejected by the existing time accuracy guard. |
| 11548 | Axial cold Hall control at 0.1 ns, 120 accepted steps. |
| 11550 | External profiling script attempted the dense array byte API on CSR. Updated instrumentation to count its data and index arrays. |
| 11551 | Nominal eight, ten and twelve-cell scaling pilots, eight accepted steps each. |
| 11555, 11556 | Large major radius input precision triggered the strict Z grid config check. Case metadata now uses the nominal spacing from the continuous box. |
| 11558, 11559 | Provisional scaling pilots and original same-input R15 16×16 reference. All compared steps accepted. |
| 11560–11563 | Reference guide-field metadata entered as a TOML integer instead of a required float. Corrected the case builder before the next four submissions. |
| 11565 | Fixed-box 12×12 perturbation, 195 ps. |
| 11566 | Fixed-box 16×16 perturbation, 130 ps. |
| 11567 | Fixed-box 16×16 quiet control, 130 ps. |
| 11568 | Fixed-box 16×16 half step, 65 ps. |
| 11569 | Original wider-box job canceled while pending to use the exact narrow-box physical window. |
| 11572 | Wider-box 20×20 perturbation, unchanged 0.075 m cells and 130 ps. |

The earlier nominal R0=3 m branch screening, higher-order reference
solves, alternative profile screens and grids remain in the
campaign directory. The nominal case showed a lower full-domain
mode overlap on refinement. The selected R0=15 m case maintains a
clear interior mode and a measured near-wall difference. No physics
or stability gates were relaxed in this selection.
