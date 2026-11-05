# FOMHD branch history

This graph shows actual commit ancestry and every fetched branch head.
It includes up to 120 recent commits plus older branch heads; omitted ancestry is marked.
The graph is a generated snapshot, preceding its own rendering commit.
Main-branch pushes refresh it automatically; other branch updates are also polled hourly.
For the live GitHub interface, use [Network](https://github.com/ziyu59151/fomhd/network)
(private-repository plan restrictions may apply).

<!-- branch-graph-refs: {"codex/a0-benchmark-gallery": "0f094115c385781c0356b333dc545539c2ed43d8", "codex/a0-runtime-guidance": "f514d38fd4f51a92ca93f68035896675b11dffe1", "codex/a0-warp": "d9390ac65657974a1cbf7819b9c79b35e2cecd03", "main": "b2dc5797fbe11ef32bb0e42d1b52a411591370e8", "reference/a0-native": "48beca72d3b7260673d6a6823cbbd2b7fcb90581"} -->

```mermaid
flowchart LR
  cf25329978364763a695bd185e986a6c0255f5df6["f2532997 · Establish C++/CUDA project conventions and auxiliary environment"]
  click cf25329978364763a695bd185e986a6c0255f5df6 "https://github.com/ziyu59151/fomhd/commit/f25329978364763a695bd185e986a6c0255f5df6"
  c48beca72d3b7260673d6a6823cbbd2b7fcb90581["48beca72 · Preserve verified A0 native reference and implementation pseudocode"]
  cf25329978364763a695bd185e986a6c0255f5df6 --> c48beca72d3b7260673d6a6823cbbd2b7fcb90581
  click c48beca72d3b7260673d6a6823cbbd2b7fcb90581 "https://github.com/ziyu59151/fomhd/commit/48beca72d3b7260673d6a6823cbbd2b7fcb90581"
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
  b0(["codex/a0-benchmark-gallery"]) -.-> c0f094115c385781c0356b333dc545539c2ed43d8
  b1(["codex/a0-runtime-guidance"]) -.-> cf514d38fd4f51a92ca93f68035896675b11dffe1
  b2(["codex/a0-warp"]) -.-> cd9390ac65657974a1cbf7819b9c79b35e2cecd03
  b3(["main"]) -.-> cb2dc5797fbe11ef32bb0e42d1b52a411591370e8
  b4(["reference/a0-native"]) -.-> c48beca72d3b7260673d6a6823cbbd2b7fcb90581
```

Regenerate after fetching:

```bash
git fetch origin --prune
python3 tools/branch_graph.py --output docs/BRANCH_GRAPH.md
```

Local terminal graph: `git graph` (see [development policy](development.md)).
