# FOMHD branch history

This graph shows actual commit ancestry and every fetched branch head.
It includes up to 120 recent commits plus older branch heads; omitted ancestry is marked.
The graph is a generated snapshot, preceding its own rendering commit.
Main-branch pushes refresh it automatically; other branch updates are also polled hourly.
For the live GitHub interface, use [Network](https://github.com/ziyu59151/fomhd/network)
(private-repository plan restrictions may apply).

<!-- branch-graph-refs: {"codex/a0-benchmark-gallery": "0f094115c385781c0356b333dc545539c2ed43d8", "codex/a0-runtime-guidance": "f514d38fd4f51a92ca93f68035896675b11dffe1", "codex/a0-warp": "b2dc5797fbe11ef32bb0e42d1b52a411591370e8", "main": "c55f1b6485873868527874e36e8d84e166756275", "reference/a0-native": "1840cfa60e3b29401c9ea55a4ee8876351f582c1"} -->

```mermaid
flowchart LR
  cf25329978364763a695bd185e986a6c0255f5df6["f2532997 · Establish C++/CUDA project conventions and auxiliary environment"]
  click cf25329978364763a695bd185e986a6c0255f5df6 "https://github.com/ziyu59151/fomhd/commit/f25329978364763a695bd185e986a6c0255f5df6"
  c48beca72d3b7260673d6a6823cbbd2b7fcb90581["48beca72 · Preserve verified A0 native reference and implementation pseudocode"]
  cf25329978364763a695bd185e986a6c0255f5df6 --> c48beca72d3b7260673d6a6823cbbd2b7fcb90581
  click c48beca72d3b7260673d6a6823cbbd2b7fcb90581 "https://github.com/ziyu59151/fomhd/commit/48beca72d3b7260673d6a6823cbbd2b7fcb90581"
  c1840cfa60e3b29401c9ea55a4ee8876351f582c1["1840cfa6 · Preserve independent native oracle exporter on reference branch"]
  c48beca72d3b7260673d6a6823cbbd2b7fcb90581 --> c1840cfa60e3b29401c9ea55a4ee8876351f582c1
  click c1840cfa60e3b29401c9ea55a4ee8876351f582c1 "https://github.com/ziyu59151/fomhd/commit/1840cfa60e3b29401c9ea55a4ee8876351f582c1"
  cd9390ac65657974a1cbf7819b9c79b35e2cecd03["d9390ac6 · Implement Warp A0 and switchable quasineutral electrons"]
  c48beca72d3b7260673d6a6823cbbd2b7fcb90581 --> cd9390ac65657974a1cbf7819b9c79b35e2cecd03
  click cd9390ac65657974a1cbf7819b9c79b35e2cecd03 "https://github.com/ziyu59151/fomhd/commit/d9390ac65657974a1cbf7819b9c79b35e2cecd03"
  c0f094115c385781c0356b333dc545539c2ed43d8["0f094115 · docs: record completed A0 scans and clarify frequency scope"]
  cd9390ac65657974a1cbf7819b9c79b35e2cecd03 --> c0f094115c385781c0356b333dc545539c2ed43d8
  click c0f094115c385781c0356b333dc545539c2ed43d8 "https://github.com/ziyu59151/fomhd/commit/0f094115c385781c0356b333dc545539c2ed43d8"
  cf514d38fd4f51a92ca93f68035896675b11dffe1["f514d38f · docs: clarify A0 diagnostics and runtime responsibilities"]
  c0f094115c385781c0356b333dc545539c2ed43d8 --> cf514d38fd4f51a92ca93f68035896675b11dffe1
  click cf514d38fd4f51a92ca93f68035896675b11dffe1 "https://github.com/ziyu59151/fomhd/commit/f514d38fd4f51a92ca93f68035896675b11dffe1"
  cb2dc5797fbe11ef32bb0e42d1b52a411591370e8["b2dc5797 · Make Python Warp the RK4 mainline and add fixed-domain verification"]
  cf514d38fd4f51a92ca93f68035896675b11dffe1 --> cb2dc5797fbe11ef32bb0e42d1b52a411591370e8
  click cb2dc5797fbe11ef32bb0e42d1b52a411591370e8 "https://github.com/ziyu59151/fomhd/commit/b2dc5797fbe11ef32bb0e42d1b52a411591370e8"
  c92c1d3e461424018ac85b1bc71deb5904caf838e["92c1d3e4 · docs: update branch graph [skip ci]"]
  cb2dc5797fbe11ef32bb0e42d1b52a411591370e8 --> c92c1d3e461424018ac85b1bc71deb5904caf838e
  click c92c1d3e461424018ac85b1bc71deb5904caf838e "https://github.com/ziyu59151/fomhd/commit/92c1d3e461424018ac85b1bc71deb5904caf838e"
  caa06cf91b5a0255c31345989f660299822eec8ef["aa06cf91 · docs: update branch graph [skip ci]"]
  c92c1d3e461424018ac85b1bc71deb5904caf838e --> caa06cf91b5a0255c31345989f660299822eec8ef
  click caa06cf91b5a0255c31345989f660299822eec8ef "https://github.com/ziyu59151/fomhd/commit/aa06cf91b5a0255c31345989f660299822eec8ef"
  c267d8e11d0f359bb7032c0500613c7f608bceda8["267d8e11 · docs: update branch graph [skip ci]"]
  caa06cf91b5a0255c31345989f660299822eec8ef --> c267d8e11d0f359bb7032c0500613c7f608bceda8
  click c267d8e11d0f359bb7032c0500613c7f608bceda8 "https://github.com/ziyu59151/fomhd/commit/267d8e11d0f359bb7032c0500613c7f608bceda8"
  cc55f1b6485873868527874e36e8d84e166756275["c55f1b64 · Refine the forward wall trace without relaxing physical constraints"]
  c267d8e11d0f359bb7032c0500613c7f608bceda8 --> cc55f1b6485873868527874e36e8d84e166756275
  click cc55f1b6485873868527874e36e8d84e166756275 "https://github.com/ziyu59151/fomhd/commit/c55f1b6485873868527874e36e8d84e166756275"
  b0(["codex/a0-benchmark-gallery"]) -.-> c0f094115c385781c0356b333dc545539c2ed43d8
  b1(["codex/a0-runtime-guidance"]) -.-> cf514d38fd4f51a92ca93f68035896675b11dffe1
  b2(["codex/a0-warp"]) -.-> cb2dc5797fbe11ef32bb0e42d1b52a411591370e8
  b3(["main"]) -.-> cc55f1b6485873868527874e36e8d84e166756275
  b4(["reference/a0-native"]) -.-> c1840cfa60e3b29401c9ea55a4ee8876351f582c1
```

Regenerate after fetching:

```bash
git fetch origin --prune
python3 tools/branch_graph.py --output docs/BRANCH_GRAPH.md
```

Local terminal graph: `git graph` (see [development policy](development.md)).
