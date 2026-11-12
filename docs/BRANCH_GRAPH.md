# FOMHD branch history

This graph shows actual commit ancestry and every fetched branch head.
It includes up to 120 recent commits plus older branch heads; omitted ancestry is marked.
The graph is a generated snapshot, preceding its own rendering commit.
Main-branch pushes refresh it automatically; other branch updates are also polled hourly.
For the live GitHub interface, use [Network](https://github.com/ziyu59151/fomhd/network)
(private-repository plan restrictions may apply).

<!-- branch-graph-refs: {"codex/a0-benchmark-gallery": "0f094115c385781c0356b333dc545539c2ed43d8", "codex/a0-grid-convergence": "81e89dbb9f58e36ff73be89e2e1912e7354920c4", "codex/a0-nl-cellplots": "0ff316a7ac35d7508b426fac3b268d99713dd1cb", "codex/a0-rect-wall-curl": "ac8b48003416357fbc8312d19bdfaa1deebcaa0d", "codex/a0-rectangle-cae-evidence": "a3ac3c1766a0a34d1d14111fc9b622f375cf299f", "codex/a0-report-closeout": "b74e85782de736522730d95b2f329be40134fe7a", "codex/a0-rk4-scaling": "c2af09e257b8b5159070482dba0627fe2c8c9f08", "codex/a0-runtime-guidance": "f514d38fd4f51a92ca93f68035896675b11dffe1", "codex/a0-tokamak-cae": "f487d8c5fecb54fc26dfe5600a72cf0b1933b844", "codex/a0-warp": "b2dc5797fbe11ef32bb0e42d1b52a411591370e8", "codex/a050-clt-intercept-wall": "f6175c2876802160cec4f901ac080fa657654198", "codex/a050-polygon-wall-research": "d323f875c76b7e6934af423dcf6b21653c3b4f47", "codex/a050-two-fluid-experiments": "0ec6b9320a937d7a7e4453c17a0a560b09ad1d53", "codex/a1-benchmark-report": "ece24d48046895abc48985b2a9515be6f5311e20", "codex/a1-multirate-ion": "b54c31e1ff47d318ce0b86c862c76e1aa0d156a1", "codex/a1-rect-wall-curl": "01f1c9857d132158f8cfcbd15886768540a6991c", "codex/a1-t0": "b9af715ab2cb75f661b2254a2ea05d2179334599", "codex/a1-t1-waveform": "4c75394ec2f540cf224b91b0cb08766c34d0a506", "codex/a1-t2-electron-field": "be5f1c07913409bfd377f22e9ea9d69ac614b158", "main": "6be975994ca2a284305d4fee1f8853fd82b17112", "reference/a0-native": "1840cfa60e3b29401c9ea55a4ee8876351f582c1", "release/a0": "1d5f684d56416b054135b894aae5485f3bac532c", "release/a1-t0": "fc9a19e26fc2fb6d601c39e18a11a095386e3038"} -->

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
  cbf9c06576d0e36b6b6e3cf64d2d5571aeb3b88fd["bf9c0657 · docs: update branch graph [skip ci]"]
  cc55f1b6485873868527874e36e8d84e166756275 --> cbf9c06576d0e36b6b6e3cf64d2d5571aeb3b88fd
  click cbf9c06576d0e36b6b6e3cf64d2d5571aeb3b88fd "https://github.com/ziyu59151/fomhd/commit/bf9c06576d0e36b6b6e3cf64d2d5571aeb3b88fd"
  c69efcac0dbf83fc8fe8092ec878e88656ef13ff7["69efcac0 · Record fixed-domain spatial controls and eight-cycle mode evidence"]
  cbf9c06576d0e36b6b6e3cf64d2d5571aeb3b88fd --> c69efcac0dbf83fc8fe8092ec878e88656ef13ff7
  click c69efcac0dbf83fc8fe8092ec878e88656ef13ff7 "https://github.com/ziyu59151/fomhd/commit/69efcac0dbf83fc8fe8092ec878e88656ef13ff7"
  cad7173810ff59367d1bab807269b6866835b3e69["ad717381 · docs: update branch graph [skip ci]"]
  c69efcac0dbf83fc8fe8092ec878e88656ef13ff7 --> cad7173810ff59367d1bab807269b6866835b3e69
  click cad7173810ff59367d1bab807269b6866835b3e69 "https://github.com/ziyu59151/fomhd/commit/ad7173810ff59367d1bab807269b6866835b3e69"
  c242f721c3d08cad3ebe28339ceced1ddeae177ca["242f721c · docs: update branch graph [skip ci]"]
  cad7173810ff59367d1bab807269b6866835b3e69 --> c242f721c3d08cad3ebe28339ceced1ddeae177ca
  click c242f721c3d08cad3ebe28339ceced1ddeae177ca "https://github.com/ziyu59151/fomhd/commit/242f721c3d08cad3ebe28339ceced1ddeae177ca"
  ccba996c68f8c92f607d4a1da6acd3e2ee383785c["cba996c6 · docs: update branch graph [skip ci]"]
  c242f721c3d08cad3ebe28339ceced1ddeae177ca --> ccba996c68f8c92f607d4a1da6acd3e2ee383785c
  click ccba996c68f8c92f607d4a1da6acd3e2ee383785c "https://github.com/ziyu59151/fomhd/commit/cba996c68f8c92f607d4a1da6acd3e2ee383785c"
  cf487d8c5fecb54fc26dfe5600a72cf0b1933b844["f487d8c5 · feat: establish massless A0 global fast-mode prototype"]
  cad7173810ff59367d1bab807269b6866835b3e69 --> cf487d8c5fecb54fc26dfe5600a72cf0b1933b844
  click cf487d8c5fecb54fc26dfe5600a72cf0b1933b844 "https://github.com/ziyu59151/fomhd/commit/f487d8c5fecb54fc26dfe5600a72cf0b1933b844"
  cb227f253d80227fa25e137486391f9afdea90d68["b227f253 · Merge reviewed massless A0 global fast-branch prototype"]
  ccba996c68f8c92f607d4a1da6acd3e2ee383785c --> cb227f253d80227fa25e137486391f9afdea90d68
  cf487d8c5fecb54fc26dfe5600a72cf0b1933b844 --> cb227f253d80227fa25e137486391f9afdea90d68
  click cb227f253d80227fa25e137486391f9afdea90d68 "https://github.com/ziyu59151/fomhd/commit/b227f253d80227fa25e137486391f9afdea90d68"
  ce18b9517faa51a63f4a0e34dcf79019c29c806b7["e18b9517 · docs: update branch graph [skip ci]"]
  cb227f253d80227fa25e137486391f9afdea90d68 --> ce18b9517faa51a63f4a0e34dcf79019c29c806b7
  click ce18b9517faa51a63f4a0e34dcf79019c29c806b7 "https://github.com/ziyu59151/fomhd/commit/e18b9517faa51a63f4a0e34dcf79019c29c806b7"
  ca3559899f2addd43aa889d1e5b4f6faef16eff67["a3559899 · docs: update branch graph [skip ci]"]
  ce18b9517faa51a63f4a0e34dcf79019c29c806b7 --> ca3559899f2addd43aa889d1e5b4f6faef16eff67
  click ca3559899f2addd43aa889d1e5b4f6faef16eff67 "https://github.com/ziyu59151/fomhd/commit/a3559899f2addd43aa889d1e5b4f6faef16eff67"
  ca57daedf5ce36aafd44e9639826c519d6cd024d8["a57daedf · docs: update branch graph [skip ci]"]
  ca3559899f2addd43aa889d1e5b4f6faef16eff67 --> ca57daedf5ce36aafd44e9639826c519d6cd024d8
  click ca57daedf5ce36aafd44e9639826c519d6cd024d8 "https://github.com/ziyu59151/fomhd/commit/a57daedf5ce36aafd44e9639826c519d6cd024d8"
  c0bd141fa5945806497f6a17e908fe921ed692672["0bd141fa · docs: update branch graph [skip ci]"]
  ca57daedf5ce36aafd44e9639826c519d6cd024d8 --> c0bd141fa5945806497f6a17e908fe921ed692672
  click c0bd141fa5945806497f6a17e908fe921ed692672 "https://github.com/ziyu59151/fomhd/commit/0bd141fa5945806497f6a17e908fe921ed692672"
  cbf037af95705b42bfea4d31639f1bd847b9a10c4["bf037af9 · docs: update branch graph [skip ci]"]
  c0bd141fa5945806497f6a17e908fe921ed692672 --> cbf037af95705b42bfea4d31639f1bd847b9a10c4
  click cbf037af95705b42bfea4d31639f1bd847b9a10c4 "https://github.com/ziyu59151/fomhd/commit/bf037af95705b42bfea4d31639f1bd847b9a10c4"
  cbb347d39df387a1101f2f21a23a3ff58f4d983a9["bb347d39 · Add bounded spatial refinement and explicit mode structure analysis"]
  ca3559899f2addd43aa889d1e5b4f6faef16eff67 --> cbb347d39df387a1101f2f21a23a3ff58f4d983a9
  click cbb347d39df387a1101f2f21a23a3ff58f4d983a9 "https://github.com/ziyu59151/fomhd/commit/bb347d39df387a1101f2f21a23a3ff58f4d983a9"
  c9fb4dc41bddc42ff9d885fa5ea2fd0d3ad24df2c["9fb4dc41 · Make mode structure comparisons independent of eigenvector scale and phase"]
  cbb347d39df387a1101f2f21a23a3ff58f4d983a9 --> c9fb4dc41bddc42ff9d885fa5ea2fd0d3ad24df2c
  click c9fb4dc41bddc42ff9d885fa5ea2fd0d3ad24df2c "https://github.com/ziyu59151/fomhd/commit/9fb4dc41bddc42ff9d885fa5ea2fd0d3ad24df2c"
  c81e89dbb9f58e36ff73be89e2e1912e7354920c4["81e89dbb · Record interim grid spectra and unresolved EAST branch continuity"]
  c9fb4dc41bddc42ff9d885fa5ea2fd0d3ad24df2c --> c81e89dbb9f58e36ff73be89e2e1912e7354920c4
  click c81e89dbb9f58e36ff73be89e2e1912e7354920c4 "https://github.com/ziyu59151/fomhd/commit/81e89dbb9f58e36ff73be89e2e1912e7354920c4"
  c25d316c43a73844fddd768764cf6bf09ce416232["25d316c4 · Merge reviewed grid refinement tools and interim evidence"]
  cbf037af95705b42bfea4d31639f1bd847b9a10c4 --> c25d316c43a73844fddd768764cf6bf09ce416232
  c81e89dbb9f58e36ff73be89e2e1912e7354920c4 --> c25d316c43a73844fddd768764cf6bf09ce416232
  click c25d316c43a73844fddd768764cf6bf09ce416232 "https://github.com/ziyu59151/fomhd/commit/25d316c43a73844fddd768764cf6bf09ce416232"
  c52fedecc471f0a681020f3894623924da6029fa3["52fedecc · docs: update branch graph [skip ci]"]
  cbf037af95705b42bfea4d31639f1bd847b9a10c4 --> c52fedecc471f0a681020f3894623924da6029fa3
  click c52fedecc471f0a681020f3894623924da6029fa3 "https://github.com/ziyu59151/fomhd/commit/52fedecc471f0a681020f3894623924da6029fa3"
  c7a390027dc80b52ed2f268d5ccf76d06f4f82f3b["7a390027 · Preserve branch graph updates alongside reviewed grid refinement"]
  c25d316c43a73844fddd768764cf6bf09ce416232 --> c7a390027dc80b52ed2f268d5ccf76d06f4f82f3b
  c52fedecc471f0a681020f3894623924da6029fa3 --> c7a390027dc80b52ed2f268d5ccf76d06f4f82f3b
  click c7a390027dc80b52ed2f268d5ccf76d06f4f82f3b "https://github.com/ziyu59151/fomhd/commit/7a390027dc80b52ed2f268d5ccf76d06f4f82f3b"
  c5494eb299e2f2b1c66c88eaf38932dbcaff8bc7e["5494eb29 · docs: update branch graph [skip ci]"]
  c7a390027dc80b52ed2f268d5ccf76d06f4f82f3b --> c5494eb299e2f2b1c66c88eaf38932dbcaff8bc7e
  click c5494eb299e2f2b1c66c88eaf38932dbcaff8bc7e "https://github.com/ziyu59151/fomhd/commit/5494eb299e2f2b1c66c88eaf38932dbcaff8bc7e"
  cb49157977ffd8a3b6c69ad665d826ec07f34017f["b4915797 · docs: update branch graph [skip ci]"]
  c5494eb299e2f2b1c66c88eaf38932dbcaff8bc7e --> cb49157977ffd8a3b6c69ad665d826ec07f34017f
  click cb49157977ffd8a3b6c69ad665d826ec07f34017f "https://github.com/ziyu59151/fomhd/commit/b49157977ffd8a3b6c69ad665d826ec07f34017f"
  cb74e85782de736522730d95b2f329be40134fe7a["b74e8578 · Clarify A0 benchmark evidence and record prototype development priorities"]
  c7a390027dc80b52ed2f268d5ccf76d06f4f82f3b --> cb74e85782de736522730d95b2f329be40134fe7a
  click cb74e85782de736522730d95b2f329be40134fe7a "https://github.com/ziyu59151/fomhd/commit/b74e85782de736522730d95b2f329be40134fe7a"
  ca918284c702800c14ee25b73b95fd5b028de6720["a918284c · Merge A0 benchmark presentation and prototype workflow"]
  cb49157977ffd8a3b6c69ad665d826ec07f34017f --> ca918284c702800c14ee25b73b95fd5b028de6720
  cb74e85782de736522730d95b2f329be40134fe7a --> ca918284c702800c14ee25b73b95fd5b028de6720
  click ca918284c702800c14ee25b73b95fd5b028de6720 "https://github.com/ziyu59151/fomhd/commit/a918284c702800c14ee25b73b95fd5b028de6720"
  c0faa08f41f17ce3f09beb5a47a22c689ec1a2b52["0faa08f4 · docs: update branch graph [skip ci]"]
  cb49157977ffd8a3b6c69ad665d826ec07f34017f --> c0faa08f41f17ce3f09beb5a47a22c689ec1a2b52
  click c0faa08f41f17ce3f09beb5a47a22c689ec1a2b52 "https://github.com/ziyu59151/fomhd/commit/0faa08f41f17ce3f09beb5a47a22c689ec1a2b52"
  cb62683e21b8f33e47dc0ab40c938cfab29928228["b62683e2 · Merge latest generated branch graph"]
  ca918284c702800c14ee25b73b95fd5b028de6720 --> cb62683e21b8f33e47dc0ab40c938cfab29928228
  c0faa08f41f17ce3f09beb5a47a22c689ec1a2b52 --> cb62683e21b8f33e47dc0ab40c938cfab29928228
  click cb62683e21b8f33e47dc0ab40c938cfab29928228 "https://github.com/ziyu59151/fomhd/commit/b62683e21b8f33e47dc0ab40c938cfab29928228"
  c3d2fe01cc315950ff85ac56623fa81905604a894["3d2fe01c · docs: update branch graph [skip ci]"]
  cb62683e21b8f33e47dc0ab40c938cfab29928228 --> c3d2fe01cc315950ff85ac56623fa81905604a894
  click c3d2fe01cc315950ff85ac56623fa81905604a894 "https://github.com/ziyu59151/fomhd/commit/3d2fe01cc315950ff85ac56623fa81905604a894"
  c26829849839c15ff72bc75e881dcb3dd46e18a9e["26829849 · docs: update branch graph [skip ci]"]
  c3d2fe01cc315950ff85ac56623fa81905604a894 --> c26829849839c15ff72bc75e881dcb3dd46e18a9e
  click c26829849839c15ff72bc75e881dcb3dd46e18a9e "https://github.com/ziyu59151/fomhd/commit/26829849839c15ff72bc75e881dcb3dd46e18a9e"
  c0ff316a7ac35d7508b426fac3b268d99713dd1cb["0ff316a7 · Render nonlinear mode maps with contiguous cells and compact vectors"]
  c3d2fe01cc315950ff85ac56623fa81905604a894 --> c0ff316a7ac35d7508b426fac3b268d99713dd1cb
  click c0ff316a7ac35d7508b426fac3b268d99713dd1cb "https://github.com/ziyu59151/fomhd/commit/0ff316a7ac35d7508b426fac3b268d99713dd1cb"
  cdc651a514299f1e0fd5e858364d605f3f603602c["dc651a51 · Merge compact nonlinear benchmark figures"]
  c26829849839c15ff72bc75e881dcb3dd46e18a9e --> cdc651a514299f1e0fd5e858364d605f3f603602c
  c0ff316a7ac35d7508b426fac3b268d99713dd1cb --> cdc651a514299f1e0fd5e858364d605f3f603602c
  click cdc651a514299f1e0fd5e858364d605f3f603602c "https://github.com/ziyu59151/fomhd/commit/dc651a514299f1e0fd5e858364d605f3f603602c"
  c9a38bf41e7bc1fc50186c81967312a7ec5f028e8["9a38bf41 · docs: update branch graph [skip ci]"]
  cdc651a514299f1e0fd5e858364d605f3f603602c --> c9a38bf41e7bc1fc50186c81967312a7ec5f028e8
  click c9a38bf41e7bc1fc50186c81967312a7ec5f028e8 "https://github.com/ziyu59151/fomhd/commit/9a38bf41e7bc1fc50186c81967312a7ec5f028e8"
  c34ebff8678dd63feece2a5fde30c595872eb3df1["34ebff86 · docs: update branch graph [skip ci]"]
  c9a38bf41e7bc1fc50186c81967312a7ec5f028e8 --> c34ebff8678dd63feece2a5fde30c595872eb3df1
  click c34ebff8678dd63feece2a5fde30c595872eb3df1 "https://github.com/ziyu59151/fomhd/commit/34ebff8678dd63feece2a5fde30c595872eb3df1"
  c93219d2f85c5185cefaa97fd3e8125cef632f4cb["93219d2f · feat: implement bounded A1 T0 predictor and fixed wall reuse"]
  c9a38bf41e7bc1fc50186c81967312a7ec5f028e8 --> c93219d2f85c5185cefaa97fd3e8125cef632f4cb
  click c93219d2f85c5185cefaa97fd3e8125cef632f4cb "https://github.com/ziyu59151/fomhd/commit/93219d2f85c5185cefaa97fd3e8125cef632f4cb"
  c581f20d050df50718e028d7550e9e65d57cc2ea7["581f20d0 · fix: explicitly upload closed temporal fixture state"]
  c93219d2f85c5185cefaa97fd3e8125cef632f4cb --> c581f20d050df50718e028d7550e9e65d57cc2ea7
  click c581f20d050df50718e028d7550e9e65d57cc2ea7 "https://github.com/ziyu59151/fomhd/commit/581f20d050df50718e028d7550e9e65d57cc2ea7"
  cb9af715ab2cb75f661b2254a2ea05d2179334599["b9af715a · docs: record A1 T0 evidence and acceptance limits"]
  c581f20d050df50718e028d7550e9e65d57cc2ea7 --> cb9af715ab2cb75f661b2254a2ea05d2179334599
  click cb9af715ab2cb75f661b2254a2ea05d2179334599 "https://github.com/ziyu59151/fomhd/commit/b9af715ab2cb75f661b2254a2ea05d2179334599"
  c02b895b07785ece33c2c782f69a66882cd724447["02b895b0 · Merge bounded A1 T0 implementation and evidence"]
  c34ebff8678dd63feece2a5fde30c595872eb3df1 --> c02b895b07785ece33c2c782f69a66882cd724447
  cb9af715ab2cb75f661b2254a2ea05d2179334599 --> c02b895b07785ece33c2c782f69a66882cd724447
  click c02b895b07785ece33c2c782f69a66882cd724447 "https://github.com/ziyu59151/fomhd/commit/02b895b07785ece33c2c782f69a66882cd724447"
  ca82df6e48c35f4522ed8091b1ba97392e40a406a["a82df6e4 · docs: update branch graph [skip ci]"]
  c02b895b07785ece33c2c782f69a66882cd724447 --> ca82df6e48c35f4522ed8091b1ba97392e40a406a
  click ca82df6e48c35f4522ed8091b1ba97392e40a406a "https://github.com/ziyu59151/fomhd/commit/a82df6e48c35f4522ed8091b1ba97392e40a406a"
  cece24d48046895abc48985b2a9515be6f5311e20["ece24d48 · docs: clarify T0 issues and benchmark interpretation"]
  ca82df6e48c35f4522ed8091b1ba97392e40a406a --> cece24d48046895abc48985b2a9515be6f5311e20
  click cece24d48046895abc48985b2a9515be6f5311e20 "https://github.com/ziyu59151/fomhd/commit/ece24d48046895abc48985b2a9515be6f5311e20"
  c91602ee825bb14943840449f473184a0edcd6b78["91602ee8 · Merge A1 benchmark presentation and T0 issue clarification"]
  ca82df6e48c35f4522ed8091b1ba97392e40a406a --> c91602ee825bb14943840449f473184a0edcd6b78
  cece24d48046895abc48985b2a9515be6f5311e20 --> c91602ee825bb14943840449f473184a0edcd6b78
  click c91602ee825bb14943840449f473184a0edcd6b78 "https://github.com/ziyu59151/fomhd/commit/91602ee825bb14943840449f473184a0edcd6b78"
  cae6b81c75521652bb3f84e815009e101fc022eca["ae6b81c7 · Add controlled T0 temporal baseline and retained evidence"]
  c91602ee825bb14943840449f473184a0edcd6b78 --> cae6b81c75521652bb3f84e815009e101fc022eca
  click cae6b81c75521652bb3f84e815009e101fc022eca "https://github.com/ziyu59151/fomhd/commit/ae6b81c75521652bb3f84e815009e101fc022eca"
  c206e850c3220c021b03932e262e8e52c307f4a7e["206e850c · Bound failure evidence merging and correct frozen control workspace"]
  cae6b81c75521652bb3f84e815009e101fc022eca --> c206e850c3220c021b03932e262e8e52c307f4a7e
  click c206e850c3220c021b03932e262e8e52c307f4a7e "https://github.com/ziyu59151/fomhd/commit/206e850c3220c021b03932e262e8e52c307f4a7e"
  c107ce4fac34bc6d1fa2a920f5ba2ce8c87258c01["107ce4fa · Publish controlled T0 evidence and A0-style benchmark gallery"]
  c206e850c3220c021b03932e262e8e52c307f4a7e --> c107ce4fac34bc6d1fa2a920f5ba2ce8c87258c01
  click c107ce4fac34bc6d1fa2a920f5ba2ce8c87258c01 "https://github.com/ziyu59151/fomhd/commit/107ce4fac34bc6d1fa2a920f5ba2ce8c87258c01"
  cfc9a19e26fc2fb6d601c39e18a11a095386e3038["fc9a19e2 · Merge controlled T0 baseline and benchmark evidence"]
  c91602ee825bb14943840449f473184a0edcd6b78 --> cfc9a19e26fc2fb6d601c39e18a11a095386e3038
  c107ce4fac34bc6d1fa2a920f5ba2ce8c87258c01 --> cfc9a19e26fc2fb6d601c39e18a11a095386e3038
  click cfc9a19e26fc2fb6d601c39e18a11a095386e3038 "https://github.com/ziyu59151/fomhd/commit/fc9a19e26fc2fb6d601c39e18a11a095386e3038"
  c3bd295608440af87850cb4b814b4c85ce1f1abff["3bd29560 · docs: update branch graph [skip ci]"]
  c91602ee825bb14943840449f473184a0edcd6b78 --> c3bd295608440af87850cb4b814b4c85ce1f1abff
  click c3bd295608440af87850cb4b814b4c85ce1f1abff "https://github.com/ziyu59151/fomhd/commit/3bd295608440af87850cb4b814b4c85ce1f1abff"
  c439eff62dd7e7e42bc211b4019f5799ffd99e8f3["439eff62 · Merge remote-tracking branch #39;origin/main#39;"]
  cfc9a19e26fc2fb6d601c39e18a11a095386e3038 --> c439eff62dd7e7e42bc211b4019f5799ffd99e8f3
  c3bd295608440af87850cb4b814b4c85ce1f1abff --> c439eff62dd7e7e42bc211b4019f5799ffd99e8f3
  click c439eff62dd7e7e42bc211b4019f5799ffd99e8f3 "https://github.com/ziyu59151/fomhd/commit/439eff62dd7e7e42bc211b4019f5799ffd99e8f3"
  c4c75394ec2f540cf224b91b0cb08766c34d0a506["4c75394e · Implement complete ion stage-path waveform iteration and controlled T1 evidence"]
  c439eff62dd7e7e42bc211b4019f5799ffd99e8f3 --> c4c75394ec2f540cf224b91b0cb08766c34d0a506
  click c4c75394ec2f540cf224b91b0cb08766c34d0a506 "https://github.com/ziyu59151/fomhd/commit/4c75394ec2f540cf224b91b0cb08766c34d0a506"
  c4e9794a3f4f3d146e4a2badc124ea3dcbe04e3c6["4e9794a3 · Preserve raw accepted histories before T1 reference export and match archived initial layout"]
  c4c75394ec2f540cf224b91b0cb08766c34d0a506 --> c4e9794a3f4f3d146e4a2badc124ea3dcbe04e3c6
  click c4e9794a3f4f3d146e4a2badc124ea3dcbe04e3c6 "https://github.com/ziyu59151/fomhd/commit/4e9794a3f4f3d146e4a2badc124ea3dcbe04e3c6"
  c88c26a996930d0f81d797ecc849b69fa803fa614["88c26a99 · Complete controlled T1 evidence and preserve gain-budget failures"]
  c4e9794a3f4f3d146e4a2badc124ea3dcbe04e3c6 --> c88c26a996930d0f81d797ecc849b69fa803fa614
  click c88c26a996930d0f81d797ecc849b69fa803fa614 "https://github.com/ziyu59151/fomhd/commit/88c26a996930d0f81d797ecc849b69fa803fa614"
  c337ad85944f2da0bfebdda7c52846918fd4c5c7d["337ad859 · Merge reviewed A1 T1 waveform implementation and controlled evidence"]
  c439eff62dd7e7e42bc211b4019f5799ffd99e8f3 --> c337ad85944f2da0bfebdda7c52846918fd4c5c7d
  c88c26a996930d0f81d797ecc849b69fa803fa614 --> c337ad85944f2da0bfebdda7c52846918fd4c5c7d
  click c337ad85944f2da0bfebdda7c52846918fd4c5c7d "https://github.com/ziyu59151/fomhd/commit/337ad85944f2da0bfebdda7c52846918fd4c5c7d"
  cbe5f1c07913409bfd377f22e9ea9d69ac614b158["be5f1c07 · Add bounded CPU and GPU time-parallel field windows"]
  c337ad85944f2da0bfebdda7c52846918fd4c5c7d --> cbe5f1c07913409bfd377f22e9ea9d69ac614b158
  click cbe5f1c07913409bfd377f22e9ea9d69ac614b158 "https://github.com/ziyu59151/fomhd/commit/be5f1c07913409bfd377f22e9ea9d69ac614b158"
  c2a3f226f0f08e93cb313507a67a9a8a305c89787["2a3f226f · Avoid unused finite-mass storage in massless runs"]
  cbe5f1c07913409bfd377f22e9ea9d69ac614b158 --> c2a3f226f0f08e93cb313507a67a9a8a305c89787
  click c2a3f226f0f08e93cb313507a67a9a8a305c89787 "https://github.com/ziyu59151/fomhd/commit/2a3f226f0f08e93cb313507a67a9a8a305c89787"
  c1184e3f89d3f5e44fd639842999f7ef77deee882["1184e3f8 · Add two bounded reduced-ion multirate research schemes"]
  c2a3f226f0f08e93cb313507a67a9a8a305c89787 --> c1184e3f89d3f5e44fd639842999f7ef77deee882
  click c1184e3f89d3f5e44fd639842999f7ef77deee882 "https://github.com/ziyu59151/fomhd/commit/1184e3f89d3f5e44fd639842999f7ef77deee882"
  c679063fe10678dafdede9a75906ee09f0531e524["679063fe · Fix backend placement in multirate cost phase diagnostics"]
  c1184e3f89d3f5e44fd639842999f7ef77deee882 --> c679063fe10678dafdede9a75906ee09f0531e524
  click c679063fe10678dafdede9a75906ee09f0531e524 "https://github.com/ziyu59151/fomhd/commit/679063fe10678dafdede9a75906ee09f0531e524"
  cb54c31e1ff47d318ce0b86c862c76e1aa0d156a1["b54c31e1 · Record bounded multirate evidence and follow-up limits"]
  c679063fe10678dafdede9a75906ee09f0531e524 --> cb54c31e1ff47d318ce0b86c862c76e1aa0d156a1
  click cb54c31e1ff47d318ce0b86c862c76e1aa0d156a1 "https://github.com/ziyu59151/fomhd/commit/b54c31e1ff47d318ce0b86c862c76e1aa0d156a1"
  c0199021af2f4e7d5e93826432575b6e0a9681e6f["0199021a · Avoid unused finite-mass storage in massless runs"]
  cbe5f1c07913409bfd377f22e9ea9d69ac614b158 --> c0199021af2f4e7d5e93826432575b6e0a9681e6f
  click c0199021af2f4e7d5e93826432575b6e0a9681e6f "https://github.com/ziyu59151/fomhd/commit/0199021af2f4e7d5e93826432575b6e0a9681e6f"
  cca867c541d77712d3dde5c8215f61a254be462ff["ca867c54 · Keep waveform history ownership in alias memory plans"]
  c0199021af2f4e7d5e93826432575b6e0a9681e6f --> cca867c541d77712d3dde5c8215f61a254be462ff
  click cca867c541d77712d3dde5c8215f61a254be462ff "https://github.com/ziyu59151/fomhd/commit/ca867c541d77712d3dde5c8215f61a254be462ff"
  c4ba3e3bfc1abc9f65876bc09aa8e47880659a3ab["4ba3e3bf · Add the bounded 0.5.0 two-fluid experiment platform"]
  cca867c541d77712d3dde5c8215f61a254be462ff --> c4ba3e3bfc1abc9f65876bc09aa8e47880659a3ab
  click c4ba3e3bfc1abc9f65876bc09aa8e47880659a3ab "https://github.com/ziyu59151/fomhd/commit/4ba3e3bfc1abc9f65876bc09aa8e47880659a3ab"
  ce756ca6af3571f35de25ea20c94b08cbea62e6ea["e756ca6a · Allow the bounded 40-cell capacity probe through shared input validation"]
  c4ba3e3bfc1abc9f65876bc09aa8e47880659a3ab --> ce756ca6af3571f35de25ea20c94b08cbea62e6ea
  click ce756ca6af3571f35de25ea20c94b08cbea62e6ea "https://github.com/ziyu59151/fomhd/commit/e756ca6af3571f35de25ea20c94b08cbea62e6ea"
  cf7074744aa184bcfbffd98b3ee110f2885ce7448["f7074744 · Add reviewed A0-style gallery for bounded 0.5.0 and multirate experiments"]
  ce756ca6af3571f35de25ea20c94b08cbea62e6ea --> cf7074744aa184bcfbffd98b3ee110f2885ce7448
  click cf7074744aa184bcfbffd98b3ee110f2885ce7448 "https://github.com/ziyu59151/fomhd/commit/f7074744aa184bcfbffd98b3ee110f2885ce7448"
  c75e4083797716cfb5ac7fa03f105edd0a64a4f1a["75e40837 · Implement few-ion-node multirate waveform relaxation and complete interval verification"]
  cf7074744aa184bcfbffd98b3ee110f2885ce7448 --> c75e4083797716cfb5ac7fa03f105edd0a64a4f1a
  click c75e4083797716cfb5ac7fa03f105edd0a64a4f1a "https://github.com/ziyu59151/fomhd/commit/75e4083797716cfb5ac7fa03f105edd0a64a4f1a"
  cbf74c0f8caf178f1a9ef05f4fa20407f8d6ef8f5["bf74c0f8 · Link the bounded wall-response audit from the WR gallery"]
  c75e4083797716cfb5ac7fa03f105edd0a64a4f1a --> cbf74c0f8caf178f1a9ef05f4fa20407f8d6ef8f5
  click cbf74c0f8caf178f1a9ef05f4fa20407f8d6ef8f5 "https://github.com/ziyu59151/fomhd/commit/bf74c0f8caf178f1a9ef05f4fa20407f8d6ef8f5"
  c8c1995a0afdb5d0244e4e26677b5bff5a5abb4ba["8c1995a0 · Record completed WR wall controls and qualify boundary response evidence"]
  cbf74c0f8caf178f1a9ef05f4fa20407f8d6ef8f5 --> c8c1995a0afdb5d0244e4e26677b5bff5a5abb4ba
  click c8c1995a0afdb5d0244e4e26677b5bff5a5abb4ba "https://github.com/ziyu59151/fomhd/commit/8c1995a0afdb5d0244e4e26677b5bff5a5abb4ba"
  cefa1f05b666c171f73fea67303e41beb88a09a46["efa1f05b · docs: update branch graph [skip ci]"]
  c439eff62dd7e7e42bc211b4019f5799ffd99e8f3 --> cefa1f05b666c171f73fea67303e41beb88a09a46
  click cefa1f05b666c171f73fea67303e41beb88a09a46 "https://github.com/ziyu59151/fomhd/commit/efa1f05b666c171f73fea67303e41beb88a09a46"
  c9283abc3cea2e546bd57f9cc841c6ee19733e5cc["9283abc3 · docs: update branch graph [skip ci]"]
  cefa1f05b666c171f73fea67303e41beb88a09a46 --> c9283abc3cea2e546bd57f9cc841c6ee19733e5cc
  click c9283abc3cea2e546bd57f9cc841c6ee19733e5cc "https://github.com/ziyu59151/fomhd/commit/9283abc3cea2e546bd57f9cc841c6ee19733e5cc"
  c446a6d6d11bb1c71be0b7bcdf758cb86f183a2ea["446a6d6d · docs: update branch graph [skip ci]"]
  c9283abc3cea2e546bd57f9cc841c6ee19733e5cc --> c446a6d6d11bb1c71be0b7bcdf758cb86f183a2ea
  click c446a6d6d11bb1c71be0b7bcdf758cb86f183a2ea "https://github.com/ziyu59151/fomhd/commit/446a6d6d11bb1c71be0b7bcdf758cb86f183a2ea"
  c93cf86019ea6a1dcaaa79710b0ba8160d04c376a["93cf8601 · docs: update branch graph [skip ci]"]
  c446a6d6d11bb1c71be0b7bcdf758cb86f183a2ea --> c93cf86019ea6a1dcaaa79710b0ba8160d04c376a
  click c93cf86019ea6a1dcaaa79710b0ba8160d04c376a "https://github.com/ziyu59151/fomhd/commit/93cf86019ea6a1dcaaa79710b0ba8160d04c376a"
  c163c3bec100c1750c3f748f2273660e20e59db5b["163c3bec · docs: update branch graph [skip ci]"]
  c93cf86019ea6a1dcaaa79710b0ba8160d04c376a --> c163c3bec100c1750c3f748f2273660e20e59db5b
  click c163c3bec100c1750c3f748f2273660e20e59db5b "https://github.com/ziyu59151/fomhd/commit/163c3bec100c1750c3f748f2273660e20e59db5b"
  c1007f6e6d30802f875b97e5ec6d1f9161933ff42["1007f6e6 · docs: update branch graph [skip ci]"]
  c163c3bec100c1750c3f748f2273660e20e59db5b --> c1007f6e6d30802f875b97e5ec6d1f9161933ff42
  click c1007f6e6d30802f875b97e5ec6d1f9161933ff42 "https://github.com/ziyu59151/fomhd/commit/1007f6e6d30802f875b97e5ec6d1f9161933ff42"
  cd11901914f2710dd1afcdb49bb34d3f41e492761["d1190191 · docs: update branch graph [skip ci]"]
  c1007f6e6d30802f875b97e5ec6d1f9161933ff42 --> cd11901914f2710dd1afcdb49bb34d3f41e492761
  click cd11901914f2710dd1afcdb49bb34d3f41e492761 "https://github.com/ziyu59151/fomhd/commit/d11901914f2710dd1afcdb49bb34d3f41e492761"
  c7e9602ba92b2c341887b0a8753f5e8c29ed6b9a2["7e9602ba · docs: update branch graph [skip ci]"]
  cd11901914f2710dd1afcdb49bb34d3f41e492761 --> c7e9602ba92b2c341887b0a8753f5e8c29ed6b9a2
  click c7e9602ba92b2c341887b0a8753f5e8c29ed6b9a2 "https://github.com/ziyu59151/fomhd/commit/7e9602ba92b2c341887b0a8753f5e8c29ed6b9a2"
  c617b9e9d31bdadc6e0fa65db005ef8bcd5a3872a["617b9e9d · Merge remote-tracking branch #39;origin/main#39; into codex/a050-two-fluid-experiments"]
  c8c1995a0afdb5d0244e4e26677b5bff5a5abb4ba --> c617b9e9d31bdadc6e0fa65db005ef8bcd5a3872a
  c7e9602ba92b2c341887b0a8753f5e8c29ed6b9a2 --> c617b9e9d31bdadc6e0fa65db005ef8bcd5a3872a
  click c617b9e9d31bdadc6e0fa65db005ef8bcd5a3872a "https://github.com/ziyu59151/fomhd/commit/617b9e9d31bdadc6e0fa65db005ef8bcd5a3872a"
  c80337cd3b5c3e387b5d035e9c3c1d658713d1e65["80337cd3 · docs: update branch graph [skip ci]"]
  c7e9602ba92b2c341887b0a8753f5e8c29ed6b9a2 --> c80337cd3b5c3e387b5d035e9c3c1d658713d1e65
  click c80337cd3b5c3e387b5d035e9c3c1d658713d1e65 "https://github.com/ziyu59151/fomhd/commit/80337cd3b5c3e387b5d035e9c3c1d658713d1e65"
  c73fb2a048813d622b643986cf6ea58dfcb82e139["73fb2a04 · Merge remote-tracking branch #39;origin/main#39; into codex/a050-two-fluid-experiments"]
  c617b9e9d31bdadc6e0fa65db005ef8bcd5a3872a --> c73fb2a048813d622b643986cf6ea58dfcb82e139
  c80337cd3b5c3e387b5d035e9c3c1d658713d1e65 --> c73fb2a048813d622b643986cf6ea58dfcb82e139
  click c73fb2a048813d622b643986cf6ea58dfcb82e139 "https://github.com/ziyu59151/fomhd/commit/73fb2a048813d622b643986cf6ea58dfcb82e139"
  cd323f875c76b7e6934af423dcf6b21653c3b4f47["d323f875 · Preserve provisional polygon wall research and consistency failures"]
  c73fb2a048813d622b643986cf6ea58dfcb82e139 --> cd323f875c76b7e6934af423dcf6b21653c3b4f47
  click cd323f875c76b7e6934af423dcf6b21653c3b4f47 "https://github.com/ziyu59151/fomhd/commit/d323f875c76b7e6934af423dcf6b21653c3b4f47"
  cf6175c2876802160cec4f901ac080fa657654198["f6175c28 · Add structured wall probes and pinned VEQ equilibrium adapter"]
  c73fb2a048813d622b643986cf6ea58dfcb82e139 --> cf6175c2876802160cec4f901ac080fa657654198
  click cf6175c2876802160cec4f901ac080fa657654198 "https://github.com/ziyu59151/fomhd/commit/f6175c2876802160cec4f901ac080fa657654198"
  ce318e6a899a1fc1a56a2c903eba81c1b82e53a76["e318e6a8 · Use the verified rectangular wall in A1 time paths"]
  c73fb2a048813d622b643986cf6ea58dfcb82e139 --> ce318e6a899a1fc1a56a2c903eba81c1b82e53a76
  click ce318e6a899a1fc1a56a2c903eba81c1b82e53a76 "https://github.com/ziyu59151/fomhd/commit/e318e6a899a1fc1a56a2c903eba81c1b82e53a76"
  ce8fb707507a8f7a018193e848c5420b79f2171ad["e8fb7075 · Use the paired rectangular-wall curl in A1 time paths"]
  ce318e6a899a1fc1a56a2c903eba81c1b82e53a76 --> ce8fb707507a8f7a018193e848c5420b79f2171ad
  click ce8fb707507a8f7a018193e848c5420b79f2171ad "https://github.com/ziyu59151/fomhd/commit/e8fb707507a8f7a018193e848c5420b79f2171ad"
  c01f1c9857d132158f8cfcbd15886768540a6991c["01f1c985 · Publish the wall benchmark and record the A0-first development order"]
  ce8fb707507a8f7a018193e848c5420b79f2171ad --> c01f1c9857d132158f8cfcbd15886768540a6991c
  click c01f1c9857d132158f8cfcbd15886768540a6991c "https://github.com/ziyu59151/fomhd/commit/01f1c9857d132158f8cfcbd15886768540a6991c"
  c0ec6b9320a937d7a7e4453c17a0a560b09ad1d53["0ec6b932 · Merge pull request #35;1 from ziyu59151/codex/a1-rect-wall-curl"]
  c73fb2a048813d622b643986cf6ea58dfcb82e139 --> c0ec6b9320a937d7a7e4453c17a0a560b09ad1d53
  c01f1c9857d132158f8cfcbd15886768540a6991c --> c0ec6b9320a937d7a7e4453c17a0a560b09ad1d53
  click c0ec6b9320a937d7a7e4453c17a0a560b09ad1d53 "https://github.com/ziyu59151/fomhd/commit/0ec6b9320a937d7a7e4453c17a0a560b09ad1d53"
  c667cd576b8ccaf295155390c977b5dcc9f0619f3["667cd576 · docs: update branch graph [skip ci]"]
  c73fb2a048813d622b643986cf6ea58dfcb82e139 --> c667cd576b8ccaf295155390c977b5dcc9f0619f3
  click c667cd576b8ccaf295155390c977b5dcc9f0619f3 "https://github.com/ziyu59151/fomhd/commit/667cd576b8ccaf295155390c977b5dcc9f0619f3"
  cb737e04ea9913aad86938a2f3556a8ade85f8b44["b737e04e · docs: update branch graph [skip ci]"]
  c667cd576b8ccaf295155390c977b5dcc9f0619f3 --> cb737e04ea9913aad86938a2f3556a8ade85f8b44
  click cb737e04ea9913aad86938a2f3556a8ade85f8b44 "https://github.com/ziyu59151/fomhd/commit/b737e04ea9913aad86938a2f3556a8ade85f8b44"
  c9f4fd7a9a57f3d3b3103f5f1ffefe1d9617647c8["9f4fd7a9 · docs: update branch graph [skip ci]"]
  cb737e04ea9913aad86938a2f3556a8ade85f8b44 --> c9f4fd7a9a57f3d3b3103f5f1ffefe1d9617647c8
  click c9f4fd7a9a57f3d3b3103f5f1ffefe1d9617647c8 "https://github.com/ziyu59151/fomhd/commit/9f4fd7a9a57f3d3b3103f5f1ffefe1d9617647c8"
  ceaf2062e6755bb5e8815f08074adbeec56962059["eaf2062e · docs: update branch graph [skip ci]"]
  c9f4fd7a9a57f3d3b3103f5f1ffefe1d9617647c8 --> ceaf2062e6755bb5e8815f08074adbeec56962059
  click ceaf2062e6755bb5e8815f08074adbeec56962059 "https://github.com/ziyu59151/fomhd/commit/eaf2062e6755bb5e8815f08074adbeec56962059"
  c1579dd455c41dbfecbad23d5f7aaf96eba0464d1["1579dd45 · docs: update branch graph [skip ci]"]
  ceaf2062e6755bb5e8815f08074adbeec56962059 --> c1579dd455c41dbfecbad23d5f7aaf96eba0464d1
  click c1579dd455c41dbfecbad23d5f7aaf96eba0464d1 "https://github.com/ziyu59151/fomhd/commit/1579dd455c41dbfecbad23d5f7aaf96eba0464d1"
  cd962c9ce0b81263cdae9ce8ce55ac3ae0df7d234["d962c9ce · docs: update branch graph [skip ci]"]
  c1579dd455c41dbfecbad23d5f7aaf96eba0464d1 --> cd962c9ce0b81263cdae9ce8ce55ac3ae0df7d234
  click cd962c9ce0b81263cdae9ce8ce55ac3ae0df7d234 "https://github.com/ziyu59151/fomhd/commit/d962c9ce0b81263cdae9ce8ce55ac3ae0df7d234"
  ca2d55c4b78a770fb4b70fe0f29de8ce409039d94["a2d55c4b · Merge pull request #35;3 from ziyu59151/codex/a1-rect-wall-curl"]
  cd962c9ce0b81263cdae9ce8ce55ac3ae0df7d234 --> ca2d55c4b78a770fb4b70fe0f29de8ce409039d94
  c01f1c9857d132158f8cfcbd15886768540a6991c --> ca2d55c4b78a770fb4b70fe0f29de8ce409039d94
  click ca2d55c4b78a770fb4b70fe0f29de8ce409039d94 "https://github.com/ziyu59151/fomhd/commit/a2d55c4b78a770fb4b70fe0f29de8ce409039d94"
  c58bc6d907690a9106bd3c68d2f3264a99403a38d["58bc6d90 · docs: update branch graph [skip ci]"]
  ca2d55c4b78a770fb4b70fe0f29de8ce409039d94 --> c58bc6d907690a9106bd3c68d2f3264a99403a38d
  click c58bc6d907690a9106bd3c68d2f3264a99403a38d "https://github.com/ziyu59151/fomhd/commit/58bc6d907690a9106bd3c68d2f3264a99403a38d"
  c3e67390dee756270b3026079f7631a58da51a88a["3e67390d · docs: update branch graph [skip ci]"]
  c58bc6d907690a9106bd3c68d2f3264a99403a38d --> c3e67390dee756270b3026079f7631a58da51a88a
  click c3e67390dee756270b3026079f7631a58da51a88a "https://github.com/ziyu59151/fomhd/commit/3e67390dee756270b3026079f7631a58da51a88a"
  c022605cea3b4514f6dda64fbe9a3a78e56b7be5d["022605ce · docs: update branch graph [skip ci]"]
  c3e67390dee756270b3026079f7631a58da51a88a --> c022605cea3b4514f6dda64fbe9a3a78e56b7be5d
  click c022605cea3b4514f6dda64fbe9a3a78e56b7be5d "https://github.com/ziyu59151/fomhd/commit/022605cea3b4514f6dda64fbe9a3a78e56b7be5d"
  cac252313a649bedec5557f9b69858a61de2ddf5a["ac252313 · docs: update branch graph [skip ci]"]
  c022605cea3b4514f6dda64fbe9a3a78e56b7be5d --> cac252313a649bedec5557f9b69858a61de2ddf5a
  click cac252313a649bedec5557f9b69858a61de2ddf5a "https://github.com/ziyu59151/fomhd/commit/ac252313a649bedec5557f9b69858a61de2ddf5a"
  cfa1462f6445b4e7a45b4d82e238ca04ed52ca1f1["fa1462f6 · docs: update branch graph [skip ci]"]
  cac252313a649bedec5557f9b69858a61de2ddf5a --> cfa1462f6445b4e7a45b4d82e238ca04ed52ca1f1
  click cfa1462f6445b4e7a45b4d82e238ca04ed52ca1f1 "https://github.com/ziyu59151/fomhd/commit/fa1462f6445b4e7a45b4d82e238ca04ed52ca1f1"
  c6be975994ca2a284305d4fee1f8853fd82b17112["6be97599 · docs: update branch graph [skip ci]"]
  cfa1462f6445b4e7a45b4d82e238ca04ed52ca1f1 --> c6be975994ca2a284305d4fee1f8853fd82b17112
  click c6be975994ca2a284305d4fee1f8853fd82b17112 "https://github.com/ziyu59151/fomhd/commit/6be975994ca2a284305d4fee1f8853fd82b17112"
  cb7b45c306620f1e1cf1fd0f523d4db57e3ea600a["b7b45c30 · Add the fixed rectangular outer wall to A0"]
  c9a38bf41e7bc1fc50186c81967312a7ec5f028e8 --> cb7b45c306620f1e1cf1fd0f523d4db57e3ea600a
  click cb7b45c306620f1e1cf1fd0f523d4db57e3ea600a "https://github.com/ziyu59151/fomhd/commit/b7b45c306620f1e1cf1fd0f523d4db57e3ea600a"
  c60a2252fdce390780139bf18e7279fe850706188["60a2252f · Record A0 rectangular wall verification and source identity"]
  cb7b45c306620f1e1cf1fd0f523d4db57e3ea600a --> c60a2252fdce390780139bf18e7279fe850706188
  click c60a2252fdce390780139bf18e7279fe850706188 "https://github.com/ziyu59151/fomhd/commit/60a2252fdce390780139bf18e7279fe850706188"
  cbc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369["bc809bdd · Pair rectangular-wall curls with an affine-exact magnetic norm"]
  c60a2252fdce390780139bf18e7279fe850706188 --> cbc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369
  click cbc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369 "https://github.com/ziyu59151/fomhd/commit/bc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369"
  c803b332ce45750b8c6e759f7405ba76ed87ee213["803b332c · Scale complete synchronous A0 steps on GPU"]
  cbc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369 --> c803b332ce45750b8c6e759f7405ba76ed87ee213
  click c803b332ce45750b8c6e759f7405ba76ed87ee213 "https://github.com/ziyu59151/fomhd/commit/803b332ce45750b8c6e759f7405ba76ed87ee213"
  c95e5b56f23be59f46bbf6f47206d23b23c34645c["95e5b56f · Preserve finite-solution and final wall projection gates"]
  c803b332ce45750b8c6e759f7405ba76ed87ee213 --> c95e5b56f23be59f46bbf6f47206d23b23c34645c
  click c95e5b56f23be59f46bbf6f47206d23b23c34645c "https://github.com/ziyu59151/fomhd/commit/95e5b56f23be59f46bbf6f47206d23b23c34645c"
  cac8b48003416357fbc8312d19bdfaa1deebcaa0d["ac8b4800 · Publish the reproducible A0 rectangular wall benchmark"]
  cbc809bdd3aa5253e1ffa60c21e7dcf5e7c84d369 --> cac8b48003416357fbc8312d19bdfaa1deebcaa0d
  click cac8b48003416357fbc8312d19bdfaa1deebcaa0d "https://github.com/ziyu59151/fomhd/commit/ac8b48003416357fbc8312d19bdfaa1deebcaa0d"
  c30e44557474bb51ac755d9921a6a1530e0dab1ae["30e44557 · Merge pull request #35;2 from ziyu59151/codex/a0-rect-wall-curl"]
  c9a38bf41e7bc1fc50186c81967312a7ec5f028e8 --> c30e44557474bb51ac755d9921a6a1530e0dab1ae
  cac8b48003416357fbc8312d19bdfaa1deebcaa0d --> c30e44557474bb51ac755d9921a6a1530e0dab1ae
  click c30e44557474bb51ac755d9921a6a1530e0dab1ae "https://github.com/ziyu59151/fomhd/commit/30e44557474bb51ac755d9921a6a1530e0dab1ae"
  c619224999dfd8f1eb93dd06f24d7d42ed416e1b0["61922499 · Merge remote-tracking branch #39;origin/release/a0#39; into codex/a0-rk4-scaling"]
  c95e5b56f23be59f46bbf6f47206d23b23c34645c --> c619224999dfd8f1eb93dd06f24d7d42ed416e1b0
  c30e44557474bb51ac755d9921a6a1530e0dab1ae --> c619224999dfd8f1eb93dd06f24d7d42ed416e1b0
  click c619224999dfd8f1eb93dd06f24d7d42ed416e1b0 "https://github.com/ziyu59151/fomhd/commit/619224999dfd8f1eb93dd06f24d7d42ed416e1b0"
  c272f46d462e0021cfadef51bf9b1e6ac3053d279["272f46d4 · Publish the synchronous A0 performance replay and evidence"]
  c619224999dfd8f1eb93dd06f24d7d42ed416e1b0 --> c272f46d462e0021cfadef51bf9b1e6ac3053d279
  click c272f46d462e0021cfadef51bf9b1e6ac3053d279 "https://github.com/ziyu59151/fomhd/commit/272f46d462e0021cfadef51bf9b1e6ac3053d279"
  cc2af09e257b8b5159070482dba0627fe2c8c9f08["c2af09e2 · Record guarded same-device replay and measured A0 capacity"]
  c272f46d462e0021cfadef51bf9b1e6ac3053d279 --> cc2af09e257b8b5159070482dba0627fe2c8c9f08
  click cc2af09e257b8b5159070482dba0627fe2c8c9f08 "https://github.com/ziyu59151/fomhd/commit/c2af09e257b8b5159070482dba0627fe2c8c9f08"
  cd02e51a19b98d4925db86dc89ae8faacf0148f34["d02e51a1 · Merge pull request #35;4 from ziyu59151/codex/a0-rk4-scaling"]
  c30e44557474bb51ac755d9921a6a1530e0dab1ae --> cd02e51a19b98d4925db86dc89ae8faacf0148f34
  cc2af09e257b8b5159070482dba0627fe2c8c9f08 --> cd02e51a19b98d4925db86dc89ae8faacf0148f34
  click cd02e51a19b98d4925db86dc89ae8faacf0148f34 "https://github.com/ziyu59151/fomhd/commit/d02e51a19b98d4925db86dc89ae8faacf0148f34"
  ca3ac3c1766a0a34d1d14111fc9b622f375cf299f["a3ac3c17 · Publish the rectangular A0 eigenmode, resolution and wall controls"]
  cd02e51a19b98d4925db86dc89ae8faacf0148f34 --> ca3ac3c1766a0a34d1d14111fc9b622f375cf299f
  click ca3ac3c1766a0a34d1d14111fc9b622f375cf299f "https://github.com/ziyu59151/fomhd/commit/a3ac3c1766a0a34d1d14111fc9b622f375cf299f"
  c1d5f684d56416b054135b894aae5485f3bac532c["1d5f684d · Merge pull request #35;5 from ziyu59151/codex/a0-rectangle-cae-evidence"]
  cd02e51a19b98d4925db86dc89ae8faacf0148f34 --> c1d5f684d56416b054135b894aae5485f3bac532c
  ca3ac3c1766a0a34d1d14111fc9b622f375cf299f --> c1d5f684d56416b054135b894aae5485f3bac532c
  click c1d5f684d56416b054135b894aae5485f3bac532c "https://github.com/ziyu59151/fomhd/commit/1d5f684d56416b054135b894aae5485f3bac532c"
  b0(["codex/a0-benchmark-gallery"]) -.-> c0f094115c385781c0356b333dc545539c2ed43d8
  b1(["codex/a0-grid-convergence"]) -.-> c81e89dbb9f58e36ff73be89e2e1912e7354920c4
  b2(["codex/a0-nl-cellplots"]) -.-> c0ff316a7ac35d7508b426fac3b268d99713dd1cb
  b3(["codex/a0-rect-wall-curl"]) -.-> cac8b48003416357fbc8312d19bdfaa1deebcaa0d
  b4(["codex/a0-rectangle-cae-evidence"]) -.-> ca3ac3c1766a0a34d1d14111fc9b622f375cf299f
  b5(["codex/a0-report-closeout"]) -.-> cb74e85782de736522730d95b2f329be40134fe7a
  b6(["codex/a0-rk4-scaling"]) -.-> cc2af09e257b8b5159070482dba0627fe2c8c9f08
  b7(["codex/a0-runtime-guidance"]) -.-> cf514d38fd4f51a92ca93f68035896675b11dffe1
  b8(["codex/a0-tokamak-cae"]) -.-> cf487d8c5fecb54fc26dfe5600a72cf0b1933b844
  b9(["codex/a0-warp"]) -.-> cb2dc5797fbe11ef32bb0e42d1b52a411591370e8
  b10(["codex/a050-clt-intercept-wall"]) -.-> cf6175c2876802160cec4f901ac080fa657654198
  b11(["codex/a050-polygon-wall-research"]) -.-> cd323f875c76b7e6934af423dcf6b21653c3b4f47
  b12(["codex/a050-two-fluid-experiments"]) -.-> c0ec6b9320a937d7a7e4453c17a0a560b09ad1d53
  b13(["codex/a1-benchmark-report"]) -.-> cece24d48046895abc48985b2a9515be6f5311e20
  b14(["codex/a1-multirate-ion"]) -.-> cb54c31e1ff47d318ce0b86c862c76e1aa0d156a1
  b15(["codex/a1-rect-wall-curl"]) -.-> c01f1c9857d132158f8cfcbd15886768540a6991c
  b16(["codex/a1-t0"]) -.-> cb9af715ab2cb75f661b2254a2ea05d2179334599
  b17(["codex/a1-t1-waveform"]) -.-> c4c75394ec2f540cf224b91b0cb08766c34d0a506
  b18(["codex/a1-t2-electron-field"]) -.-> cbe5f1c07913409bfd377f22e9ea9d69ac614b158
  b19(["main"]) -.-> c6be975994ca2a284305d4fee1f8853fd82b17112
  b20(["reference/a0-native"]) -.-> c1840cfa60e3b29401c9ea55a4ee8876351f582c1
  b21(["release/a0"]) -.-> c1d5f684d56416b054135b894aae5485f3bac532c
  b22(["release/a1-t0"]) -.-> cfc9a19e26fc2fb6d601c39e18a11a095386e3038
```

Regenerate after fetching:

```bash
git fetch origin --prune
python3 tools/branch_graph.py --output docs/BRANCH_GRAPH.md
```

Local terminal graph: `git graph` (see [development policy](development.md)).
