# Git branch graph generation

Traceability: `A0-TOOL-BRANCH_GRAPH`.

## A0-TOOL-GRAPH-01

Read fetched remote branch refs (excluding symbolic HEAD), then read their real
commit ancestry with Git. Include each branch head even when it lies outside
the bounded recent-history window. Draw commit nodes and parent-to-child edges
as a GitHub-rendered Mermaid flowchart; attach named branch nodes to their
actual heads. The graph is a snapshot, with omitted history explicitly marked.
It is not a synthetic release roadmap. Escape labels, link commits to GitHub,
and keep source/render generation independent of the numerical environment.

Write only the requested Markdown output. A GitHub Actions workflow regenerates
the graph after branch pushes/deletions/creations and on manual request. An hourly
default-branch schedule also catches changes on old branches without the workflow.
Record the source branch refs in a Markdown comment; when comparing snapshots,
ignore one single-parent commit that changes only the graph file. Unchanged refs
therefore do not create another commit. The graph predates its rendering commit.
Tests construct a real local fork-and-merge history and verify its parent edges
and all branch labels, including an old branch outside the recent window.

Validate the history limit before idempotence; a changed limit regenerates. Mermaid labels use its own entities. Reject diagrams above 48000 characters or450 edges. Publication retries at most3 times, fetching fresh main and regenerating after a concurrent push, without force-pushing.
