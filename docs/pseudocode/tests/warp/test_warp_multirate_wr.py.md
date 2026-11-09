# Runner test pseudocode

```text
required JSON:
  reject missing/unknown controls before output creation
  default omitted h to 1.9794733885623e-10 s

relocation:
  every source/test/pseudocode path from _source_paths resolves under SOURCE_ROOT

periodic smoke:
  run a very short one-macro input at a test h
  require completed JSON, all required provenance fields, lossless arrays,
  accepted mode amplitudes, independent h/h2 references, and no production
  acceptance claim
  require exact candidate s coarse and 4M final accepted fine identities,
  distinct RK midpoint generations, actual A/projection/G shapes, and explicit
  trial-endpoint/interval-transaction flags

overwrite/failure:
  require an existing output is unchanged
  inject or select an unsupported/failing case and require preserved failure
  evidence plus nonzero exit

CUDA guard:
  require device=cuda rejected outside an owned Slurm allocation
```
