#!/usr/bin/env python3
"""Render actual fetched Git branch ancestry as GitHub Mermaid Markdown."""

import argparse
import json
from pathlib import Path
import subprocess


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _label(text):
    entities = {"'": "#39;", "`": "#96;", "#": "#35;", "&": "#amp;",
                "<": "#lt;", ">": "#gt;", '"': "#quot;", "\n": " "}
    return "".join(entities.get(char, char) for char in text)


def branch_heads(repo, ref_prefix="refs/remotes/origin"):
    refs = {}
    rows = _git(repo, "for-each-ref", "--format=%(refname) %(objectname) %(symref)", ref_prefix)
    for row in rows.splitlines():
        parts = row.split()
        if len(parts) == 2:
            refs[parts[0].removeprefix(ref_prefix + "/")] = parts[1]
    if not refs:
        raise ValueError(f"no branches under {ref_prefix}; fetch origin first")
    return refs


def unchanged_snapshot(repo, existing, refs):
    """Ignore only the one graph-only commit following the recorded snapshot."""
    prefix = "<!-- branch-graph-refs: "
    line = next((line for line in existing.splitlines() if line.startswith(prefix)), "")
    try:
        previous = json.loads(line.removeprefix(prefix).removesuffix(" -->"))
    except ValueError:
        return False
    if not isinstance(previous, dict) or previous.keys() != refs.keys():
        return False
    for name, sha in refs.items():
        if sha == previous[name]:
            continue
        parents = _git(repo, "show", "-s", "--format=%P", sha).split()
        if parents != [previous[name]]:
            return False
        changed = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", sha).splitlines()
        if changed != ["docs/BRANCH_GRAPH.md"]:
            return False
    return True


def render(repo, *, limit=120, ref_prefix="refs/remotes/origin"):
    if not 1 <= limit <= 120:
        raise ValueError("limit must be in [1, 120]")
    refs = branch_heads(repo, ref_prefix)
    commits = _git(repo, "rev-list", "--topo-order", f"--max-count={limit}", *refs.values()).splitlines()
    commits = list(dict.fromkeys([*commits, *refs.values()]))
    details = {}
    for sha in commits:
        parents, subject = _git(repo, "show", "-s", "--format=%P%x00%s", sha).split("\0", 1)
        details[sha] = (parents.split(), subject)
    lines = [
        "# FOMHD branch history", "",
        "This graph shows actual commit ancestry and every fetched branch head.",
        f"It includes up to {limit} recent commits plus older branch heads; omitted ancestry is marked.",
        "The graph is a generated snapshot, preceding its own rendering commit.",
        "Main-branch pushes refresh it automatically; other branch updates are also polled hourly.",
        "For the live GitHub interface, use [Network](https://github.com/ziyu59151/fomhd/network)",
        "(private-repository plan restrictions may apply).", "",
        "<!-- branch-graph-refs: " + json.dumps(refs, sort_keys=True) + " -->",
        "", "```mermaid", "flowchart LR",
    ]
    missing = set()
    for sha in reversed(commits):
        parents, subject = details[sha]
        lines.append(f'  c{sha}["{sha[:8]} · {_label(subject[:100])}"]')
        for parent in parents:
            if parent not in details and parent not in missing:
                lines.append(f'  c{parent}["Earlier history · {parent[:8]}"]')
                missing.add(parent)
            lines.append(f"  c{parent} --> c{sha}")
        lines.append(f'  click c{sha} "https://github.com/ziyu59151/fomhd/commit/{sha}"')
    for index, (name, sha) in enumerate(sorted(refs.items())):
        lines.append(f'  b{index}(["{_label(name)}"]) -.-> c{sha}')
    lines += ["```", "", "Regenerate after fetching:", "", "```bash",
              "git fetch origin --prune", "python3 tools/branch_graph.py --output docs/BRANCH_GRAPH.md", "```", "",
              "Local terminal graph: `git graph` (see [development policy](development.md)).", ""]
    rendered = "\n".join(lines)
    if len(rendered) > 48000 or sum("-->" in line or "-.->" in line for line in lines) > 450:
        raise ValueError("graph exceeds the GitHub Mermaid budget; choose a smaller --limit")
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.limit <= 120:
        parser.error("limit must be in [1, 120]")
    if args.output.exists():
        existing = args.output.read_text()
        if (f"It includes up to {args.limit} recent commits" in existing
                and unchanged_snapshot(args.repo, existing, branch_heads(args.repo))):
            return
    rendered = render(args.repo, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
