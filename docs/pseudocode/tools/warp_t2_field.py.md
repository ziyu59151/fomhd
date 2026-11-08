# T2-E bounded evidence runner

Status: implemented bounded field evidence runner under D070, 2026-11-08.
The command selects E1 linear, E2 periodic nonlinear, or E3 actual closed RZ;
CPU/GPU and serial/prefix are explicit. GPU requires an owned Slurm allocation.
Each new output directory stores input/config, full waveform/stages/rates,
failure/residual history, source hashes and resource/timing records. No accepted
solver transaction or complete ion budget is produced by this field subproblem.

E1 compares independent sequential four-source RK4 to affine tree and recovers
all stages. An independent circular Hall solution checks phase/amplitude in
the unforced control. E2/E3 use FieldBatch validated against production stages,
and an independent sequential RK4 time oracle. E2 uniform frozen Hall FFT
reference retains real nonlinear variable-density RHS; E3 frozen actual spatial
JVP/polynomial reference counts every derivative and its sequential degree.

Input/geometry construction and initial transfers are recorded separately as
input_setup_s. Timed windows include reference propagation construction,
tangent guess, all trial residuals, tree levels, final stage recovery, checks
and synchronization; the independent serial oracle is separately charged.
Two warmups, five synchronized repetitions for qualified timing; raw samples,
cold/setup, CPU hardware and same-GPU comparisons remain separate. Failed
cases retain the original failure code/message/traceback/nodes, attempted
wrapper counters and a copy of the available producer work. Before the
independent oracle the attempted-work phase is `oracle`; after its counter reset
and before timed execution it is `timed`, so an oracle failure is not attributed
to the measured algorithm. `host_peak_rss_bytes` is recorded on both successful
and failed exits. Failed cases do not rerun work, infer missing historical work,
save additional large arrays or substitute old results. Refinement at fixed H
and window growth at fixed h are explicit configurations; sample
sizes/observation times are never silently changed.

Both serial and prefix return E/current/velocity for every selected RK stage;
the already evaluated selected residual provides these arrays without replay.
Both charge an additional endpoint validity RHS against the last supplied ion
stage. Reset actual producer and wrapper counters after the independent serial
oracle. Work totals include cold, warm and measured windows only. Label last
endpoint checks explicitly; retain min/max checks over all timed RHS, including
JVPs and rejected trials. These are not a complete coupled energy ledger.
