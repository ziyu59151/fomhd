# FOMHD source collaboration rules

- The implementation language is C++20 with a CUDA backend. All source code,
  comments, test descriptions, runtime logs, diagnostics, and implementation
  records must be English. The pseudocode directory is excluded because its
  bilingual edition is maintained separately.
- The design authority is the sibling repository
  `/home/lxzeng/fomhd-design`: DECISIONS, A0_IMPLEMENTATION_SPEC, and
  A0_GLOBAL_DISCRETIZATION. It is navigation and review context, not a build
  or runtime dependency. If it is unavailable, report that fact instead of
  reconstructing the model.
- Implementation-level pseudocode under `docs/pseudocode/` precedes and stays
  synchronized with its source and tests. Do not create placeholder APIs or
  empty tests to claim implementation.
- Interfaces must state units, layout, stage identity, ownership, host/device
  placement, synchronization, and failure behavior. Time integration owns the
  accepted/rejected transaction; diagnostics are read-only; shared physical
  fluxes and currents have one producer.
- CMake owns C++/CUDA targets and native dependencies. `pyproject.toml` and
  `uv.lock` manage auxiliary Python/PyTorch tools only. Do not assume an
  unrecorded library is installed or silently switch the solver to Python.
- Use ordinary user permissions. Do not use sudo or modify public drivers,
  CUDA Toolkit installation, device ACLs, Slurm configuration, or other users'
  files. Run GPU work only inside the user's own Slurm allocation. Never read,
  print, or commit credentials.
- Use `codex/<task>` branches during development. Merge to `master` only after
  review and verification. Create release branches and immutable tags only for
  a stated, passed benchmark scope. Do not force-push or rewrite tags.
- Verify the repository root before Git operations. Keep build directories,
  virtual environments, run products, literature, and design evidence out of
  the source repository unless explicitly requested.
- Use `/home/lxzeng/runs/fomhd/` for personal FOMHD runs. Each run has a new
  recognizable output directory and retains inputs, provenance, diagnostics,
  checkpoints, and failure evidence.
- Select the relevant C++/CUDA, testing, systematic-debugging, and
  verification skills for changes. Performance optimization is a separate
  opt-in phase and is not implied by ordinary CUDA correctness work.
- Before any completion claim, run the required build/tests and inspect their
  actual output. Report unverified scientific or performance scope explicitly.
