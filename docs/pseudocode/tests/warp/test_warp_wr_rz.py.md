# test_warp_wr_rz.py production pseudocode

Load the runner from `Path(__file__).parents[2]`. Derive the default accepted
source root from the same repository parents; permit `FOMHD_SOURCE_ROOT` and a
tool-root override only for scratch verification. Require the real accepted
`multirate_wr_physics.py` and `time_multirate.py`.

Use genuine pinned-EAST node-4 CPU fixtures with `h=h_default/2`. The default-h
node-4 case correctly fails the unchanged continuity hard gate; tests do not
relax it.

Exercise:

1. Two complete accepted WR macros with cumulative ledger, accepted checks,
   source/model/scheme metadata, and no fine full-ion stages.
2. All 12 mixed ledger fields per species and six boundary integrals.
3. Uninterrupted two-macro state versus one macro plus `--resume` in another
   output, using `--resume-checkpoint` and matching state/clock/counters.
4. Resume rejection when global macros change, before output creation.
   Also reject changed perturbation amplitude through the exact scheme hash.
5. WR, waveform, and synchronous same outer `M*h` horizon; sync has M microsteps
   and M complete advances per ensemble record.
6. Reference work counts from scoped real producers rather than zero/constants.
7. Injected resumed-transaction failure leaves the parent checkpoint unchanged
   and preserves left JSON/NPZ evidence.
8. Existing output and unowned CUDA guards.
9. Nonzero K=1 perturbation, physical sample layouts, pressure fields, and
   retained m=1 response.
10. An eight-macro `M=4` segmented/resumed run crosses the known macro-6
    associativity boundary and requires `core.time_at_step` exactness. The
    case exposes `--waveform-rtol=1e-14` and verifies config, context,
    scheme/resume identity, checkpoint, and snapshot metadata.
