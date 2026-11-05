"""Branch visualization must reflect real Git ancestry, including merges."""

import importlib.util
from pathlib import Path
import subprocess


def test_real_fork_merge_and_old_head(tmp_path):
    script = Path(__file__).resolve().parents[2] / "tools/branch_graph.py"
    spec = importlib.util.spec_from_file_location("branch_graph", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Graph test")
    git("commit", "--allow-empty", "-m", "root")
    root = git("rev-parse", "HEAD")
    git("branch", "reference/native")
    git("switch", "-c", "feature")
    git("commit", "--allow-empty", "-m", 'feature "quoted"')
    feature = git("rev-parse", "HEAD")
    git("switch", "main")
    git("commit", "--allow-empty", "-m", "main development")
    main = git("rev-parse", "HEAD")
    git("merge", "--no-ff", "feature", "-m", "merge")
    merge = git("rev-parse", "HEAD")
    graph = module.render(tmp_path, limit=2, ref_prefix="refs/heads")
    for name in ("main", "feature", "reference/native"):
        assert name in graph
    assert f"c{main} --> c{merge}" in graph
    assert f"c{feature} --> c{merge}" in graph
    assert root[:8] in graph
    assert "Earlier history" in graph
    assert "#quot;quoted#quot;" in graph
    assert module._label("user's `x` #1 & <a>") == "user#39;s #96;x#96; #35;1 #amp; #lt;a#gt;"
    assert '```mermaid' in graph
    assert module.unchanged_snapshot(tmp_path, graph, module.branch_heads(tmp_path, "refs/heads"))
    output = tmp_path / "docs/BRANCH_GRAPH.md"
    output.parent.mkdir()
    output.write_text(graph)
    git("add", "docs/BRANCH_GRAPH.md")
    git("commit", "-m", "docs: update branch graph [skip ci]")
    assert module.unchanged_snapshot(tmp_path, graph, module.branch_heads(tmp_path, "refs/heads"))
    git("commit", "--allow-empty", "-m", "new source event")
    assert not module.unchanged_snapshot(tmp_path, graph, module.branch_heads(tmp_path, "refs/heads"))
    assert not module.unchanged_snapshot(tmp_path, "bad metadata", module.branch_heads(tmp_path, "refs/heads"))
