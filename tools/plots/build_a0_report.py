#!/usr/bin/env python3
"""Build a bounded static HTML report from a reviewed A0 JSON manifest."""

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

SOURCE_BADGES = {
    "nonlinear_evolution": ("NL | FOMHD time evolution", "nl"),
    "production_linearization": ("LIN | FOMHD discrete Jacobian", "lin"),
    "independent_reference": ("REF | independent model", "ref"),
    "verification": ("VER | verification", "verify"),
    "mixed_comparison": ("MIX | explicit components", "mixed"),
}
PAGE_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
RESERVED_IDS = {
    "main",
    "report-title",
    "introduction",
    "notation",
    "archive",
    "report-index",
    "conclusions",
}


class ReportError(ValueError):
    """Raised when a manifest or referenced artifact is invalid."""


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportError(f"{where} must be an object")
    return value


def _items(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReportError(f"{where} must be an array")
    return value


def _text(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReportError(f"{where}.{key} must be nonempty text")
    return value


def _scalar(value: Any, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ReportError(f"{where} must be a string or number")
    if isinstance(value, str) and not value.strip():
        raise ReportError(f"{where} must not be blank")
    return str(value)


def _link(
    value: Any,
    key: str,
    where: str,
    base: Path,
    output_dir: Path,
    output: Path,
    *,
    file_required: bool,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportError(f"{where}.{key} must be nonempty local path text")
    source = Path(value)
    if source.is_absolute():
        raise ReportError(f"{where}.{key} must be manifest-relative")
    target = (base / source).resolve()
    if target == output.resolve():
        raise ReportError(f"{where}.{key} cannot overwrite an input artifact")
    if file_required and not target.is_file():
        raise ReportError(f"{where}.{key} file does not exist: {value}")
    if not target.exists():
        raise ReportError(f"{where}.{key} path does not exist: {value}")
    relative = os.path.relpath(target, output_dir)
    return quote(Path(relative).as_posix(), safe="/:@-._~")


def _number(figure: dict[str, Any], where: str) -> int:
    value = figure.get("number")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ReportError(f"{where}.number must be a positive integer")
    return value


def _badge(figure: dict[str, Any], where: str) -> tuple[str, str]:
    kind = figure.get("source_kind")
    if kind not in SOURCE_BADGES:
        raise ReportError(f"{where}.source_kind must be one of {sorted(SOURCE_BADGES)}")
    label, css_class = SOURCE_BADGES[kind]
    components = figure.get("components")
    if kind == "mixed_comparison":
        values = _items(components, f"{where}.components")
        if not values or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            raise ReportError(f"{where}.components must list named components")
    elif components is not None:
        values = _items(components, f"{where}.components")
    else:
        return label, css_class
    details = "; ".join(_esc(_scalar(item, f"{where}.components")) for item in values)
    return f"{label}: {details}", css_class


def _metadata(figure: dict[str, Any], where: str) -> str:
    rows: list[str] = []
    if "grid" in figure:
        grid = _object(figure["grid"], f"{where}.grid")
        for axis in ("n_r", "n_z"):
            if axis not in grid:
                raise ReportError(f"{where}.grid.{axis} is required")
            value = _scalar(grid[axis], f"{where}.grid.{axis}")
            rows.append(f"<dt>{axis.upper()}</dt><dd>{_esc(value)}</dd>")
    fields = (
        ("n_phi", "n_phi (toroidal harmonic)"),
        ("k", "K (retained harmonic cutoff)"),
        ("n_phi_quadrature", "N_phi quadrature"),
        ("device", "Device"),
    )
    for key, label in fields:
        if key in figure and figure[key] is not None:
            value = figure[key]
            text = (
                _text(figure, key, where)
                if isinstance(value, str)
                else _scalar(value, f"{where}.{key}")
            )
            rows.append(f"<dt>{label}</dt><dd>{_esc(text)}</dd>")
    status = figure.get("status")
    if status is not None:
        status = _text(figure, "status", where)
        normalized_status = status.casefold()
        warning = any(
            word in normalized_status
            for word in (
                "unavailable",
                "nonterminal",
                "pending",
                "provisional",
                "incomplete",
                "not final",
                "failed",
                "running",
                "initializing",
                "budget_stopped",
                "budget-stopped",
                "budget stopped",
            )
        )
        prefix = "<strong>Limited/provisional evidence:</strong> " if warning else ""
        css_class = "status warning" if warning else "status"
        rows.append(
            f'<dt>Status</dt><dd><span class="{css_class}">'
            f"{prefix}{_esc(status)}</span></dd>"
        )
    return f'<dl class="figure-metadata">{"".join(rows)}</dl>' if rows else ""


def _panels(figure: dict[str, Any], where: str) -> str:
    if "panels" not in figure:
        return ""
    seen: set[str] = set()
    panels: list[str] = []
    for index, raw in enumerate(_items(figure["panels"], f"{where}.panels")):
        panel_where = f"{where}.panels[{index}]"
        panel = _object(raw, panel_where)
        letter = _text(panel, "letter", panel_where)
        if not re.fullmatch(r"[A-Za-z]", letter) or letter in seen:
            raise ReportError(f"{panel_where}.letter must be a unique letter")
        seen.add(letter)
        panels.append(
            f'<li><span class="panel-ref">({_esc(letter)})</span> '
            f"{_esc(_text(panel, 'description', panel_where))}</li>"
        )
    return f'<ul class="panel-list">{"".join(panels)}</ul>'


def _figure(
    raw: Any, where: str, base: Path, output_dir: Path, output: Path
) -> tuple[int, str]:
    figure = _object(raw, where)
    number = _number(figure, where)
    title = _text(figure, "title", where)
    caption = _text(figure, "caption", where)
    badge, badge_class = _badge(figure, where)
    image = _link(
        figure.get("image"),
        "image",
        where,
        base,
        output_dir,
        output,
        file_required=True,
    )
    links = ""
    for key in ("svg", "data", "provenance"):
        if key in figure and figure[key] is not None:
            href = _link(
                figure[key], key, where, base, output_dir, output, file_required=True
            )
            links += f'<a href="{href}">{key.upper()}</a> '
    links = f'<p class="artifact-links">Files: {links}</p>' if links else ""
    body = (
        f'<figure id="figure-{number}"><h3>Figure {number}: {_esc(title)}</h3>'
        f'<img src="{image}" alt="{_esc(title)}">'
        f'<figcaption><span class="source-badge {badge_class}">{badge}</span>'
        f'<span class="figure-caption">{_esc(caption)}</span>'
        f"{_metadata(figure, where)}{_panels(figure, where)}{links}"
        f"</figcaption></figure>"
    )
    collapsed = figure.get("collapsed", False)
    if not isinstance(collapsed, bool):
        raise ReportError(f"{where}.collapsed must be true or false")
    if not collapsed:
        return number, body
    summary = f"Figure {number}: {title} - {SOURCE_BADGES[figure['source_kind']][0]}"
    return number, (
        '<details class="figure-details">'
        f"<summary>{_esc(summary)}</summary>{body}</details>"
    )


def _page(
    raw: Any,
    position: int,
    total: int,
    pages: list[dict[str, Any]],
    numbers: set[int],
    page_ids: list[str],
    base: Path,
    output_dir: Path,
    output: Path,
) -> tuple[str, str, str]:
    where = f"pages[{position}]"
    page = _object(raw, where)
    page_id = _text(page, "id", where)
    if not PAGE_ID.fullmatch(page_id):
        raise ReportError(f"{where}.id must be an HTML-safe identifier")
    title = _text(page, "title", where)
    intro = _text(page, "introduction", where)
    notes = _items(page.get("notes", []), f"{where}.notes")
    figures = _items(page.get("figures", []), f"{where}.figures")
    rendered = []
    for index, figure in enumerate(figures):
        number, body = _figure(
            figure, f"{where}.figures[{index}]", base, output_dir, output
        )
        if number in numbers:
            raise ReportError(f"duplicate figure number: {number}")
        numbers.add(number)
        if f"figure-{number}" in page_ids:
            raise ReportError(f"figure id collides with page id: {number}")
        rendered.append(body)
    previous = f"page-{pages[position - 1]['id']}" if position else "report-index"
    following = (
        f"page-{pages[position + 1]['id']}"
        if position + 1 < len(pages)
        else "report-index"
    )
    nav = (
        '<nav class="page-nav" aria-label="Page navigation">'
        '<a href="#report-index">Index</a>'
        f'<a href="#{_esc(previous)}">Previous</a>'
        f'<a href="#{_esc(following)}">Next</a></nav>'
    )
    note_html = "".join(f"<li>{_esc(_scalar(note, where))}</li>" for note in notes)
    body = (
        f'<article class="report-page" id="page-{_esc(page_id)}">'
        f'<p class="page-counter">Page {position + 1} / {total}</p>{nav}'
        f'<h2>{_esc(title)}</h2><p class="prose">{_esc(intro)}</p>'
        f"{''.join(rendered)}"
        f"{f'<h3>Notes</h3><ul>{note_html}</ul>' if note_html else ''}"
        "</article>"
    )
    return page_id, title, body


def render_report(manifest: dict[str, Any], manifest_path: Path, output: Path) -> str:
    """Validate and render a manifest without writing the output."""
    manifest = _object(manifest, "manifest")
    base, output_dir = manifest_path.resolve().parent, output.parent.resolve()
    title = _text(manifest, "title", "manifest")
    intro = _text(manifest, "introduction", "manifest")
    notation = _items(manifest.get("notation"), "manifest.notation")
    if not notation:
        raise ReportError("manifest.notation must not be empty")
    archive = _object(manifest.get("archive"), "manifest.archive")
    archive_href = _link(
        archive.get("path"),
        "path",
        "manifest.archive",
        base,
        output_dir,
        output,
        file_required=False,
    )
    archive_label = _text(archive, "label", "manifest.archive")
    pages = _items(manifest.get("pages"), "manifest.pages")
    if not pages:
        raise ReportError("manifest.pages must not be empty")
    page_objects = [_object(page, f"pages[{i}]") for i, page in enumerate(pages)]
    page_ids = [_text(page, "id", f"pages[{i}]") for i, page in enumerate(page_objects)]
    if len(page_ids) != len(set(page_ids)):
        raise ReportError("duplicate page id")
    if RESERVED_IDS.intersection(page_ids):
        raise ReportError("reserved or duplicate page id")
    conclusions = _text(manifest, "conclusions", "manifest")
    issues: list[str] = []
    for index, raw in enumerate(
        _items(manifest.get("known_issues", []), "manifest.known_issues")
    ):
        where = f"manifest.known_issues[{index}]"
        issue = _object(raw, where)
        href = _link(
            issue.get("path"),
            "path",
            where,
            base,
            output_dir,
            output,
            file_required=False,
        )
        issues.append(
            f'<li><a href="{href}">{_esc(_text(issue, "label", where))}</a></li>'
        )
    rows: list[str] = []
    for index, raw in enumerate(notation):
        where = f"manifest.notation[{index}]"
        row = _object(raw, where)
        rows.append(
            f"<tr><td>{_esc(_text(row, 'symbol', where))}</td>"
            f"<td>{_esc(_text(row, 'meaning', where))}</td></tr>"
        )
    numbers: set[int] = set()
    page_data = [
        _page(
            page,
            position,
            len(page_objects),
            page_objects,
            numbers,
            page_ids,
            base,
            output_dir,
            output,
        )
        for position, page in enumerate(page_objects)
    ]
    index = "".join(
        f'<li><a href="#page-{_esc(pid)}">{_esc(page_title)}</a></li>'
        for pid, page_title, _ in page_data
    )
    issue_html = f"<h3>Known issues</h3><ul>{''.join(issues)}</ul>" if issues else ""
    style = """<style>
:root{color-scheme:light;font-family:system-ui,sans-serif;line-height:1.5;color:#17202a;background:#f5f7f8}
*{box-sizing:border-box}body{max-width:1240px;margin:0 auto;padding:2rem 1rem 4rem}a{color:#075985}a:focus{outline:3px solid #f59e0b;outline-offset:2px}.prose{white-space:pre-line}
.skip{position:absolute;left:-9999px}.skip:focus{left:1rem;top:1rem;background:white;padding:.5rem;z-index:2}section,article{background:white;border:1px solid #cbd5e1;border-radius:.6rem;padding:clamp(1rem,3vw,2rem);margin:1.5rem 0}
table{border-collapse:collapse;width:100%}th,td{border:1px solid #cbd5e1;padding:.55rem;text-align:left}th{background:#e2e8f0}.page-nav{display:flex;gap:1.2rem;flex-wrap:wrap}.page-counter{font-weight:700;color:#334155}
.report-page figure{border-top:1px solid #cbd5e1;padding-top:1.5rem;margin:1.5rem 0}.report-page img{display:block;width:100%;height:auto;max-width: 1100px;margin:0 auto;background:white}
figcaption p{margin:.6rem 0}.source-badge{display:inline-block;font-weight:700;border:2px solid;border-radius:.35rem;padding:.25rem .55rem;margin:.25rem 0}.nl{color:#005b4f;background:#ccfbf1;border-color:#0f766e}.lin{color:#581c87;background:#f3e8ff;border-color:#7e22ce}
.ref{color:#273240;background:#e5e7eb;border-color:#64748b}.verify{color:#713f12;background:#fef3c7;border-color:#b45309}.mixed{color:#1e3a5f;background:#e0f2fe;border-color:#0369a1}.figure-caption{display:block}
.figure-metadata{display:grid;grid-template-columns:max-content 1fr;gap:.25rem 1rem}.figure-metadata dt{font-weight:700}.figure-metadata dd{margin:0}.status.warning{background:#fff7ed;color:#9a3412;border:1px solid #ea580c;padding:.2rem .4rem}
.panel-ref{font-weight:700}.figure-details{margin:1.5rem 0}.figure-details>summary{cursor:pointer;font-weight:700;padding:.75rem;background:#eef2f7;border-radius:.4rem}footer{color:#475569;font-size:.95rem}
@page{size:A4;margin:16mm;@bottom-center{content:"Page " counter(page) " of " counter(pages)}}@media print{.skip,.page-nav{display:none}.report-page{break-inside:avoid}.report-page+.report-page{break-before:page}}
</style>"""
    return (
        '<!doctype html>\n<html lang="en">\n<head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(title)}</title>{style}</head><body>"
        '<a class="skip" href="#main">Skip to report</a>'
        f'<header><h1 id="report-title">{_esc(title)}</h1>'
        f'<p class="prose">{_esc(intro)}</p></header><main id="main">'
        '<section id="notation"><h2>Notation</h2><table><thead><tr>'
        "<th>Symbol</th><th>Meaning</th></tr></thead><tbody>"
        f"{''.join(rows)}</tbody></table></section>"
        f'<section id="archive"><h2>Archive</h2>'
        f'<a href="{archive_href}">{_esc(archive_label)}</a></section>'
        '<nav id="report-index" aria-label="Report pages">'
        f"<h2>Report pages</h2><ol>{index}</ol></nav></main>"
        f"{''.join(body for _, _, body in page_data)}"
        f'<section id="conclusions"><h2>Conclusions</h2>'
        f'<p class="prose">{_esc(conclusions)}</p>{issue_html}</section>'
        "<footer>Supplied figures support review but do not by themselves "
        "establish convergence or scientific acceptance.</footer>"
        "</body></html>\n"
    )


def _backup(output: Path) -> Path:
    digest = hashlib.sha256(output.read_bytes()).hexdigest()[:12]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    candidate = output.with_name(f"{output.name}.bak-{stamp}-{digest}")
    suffix = 1
    while candidate.exists():
        candidate = output.with_name(f"{output.name}.bak-{stamp}-{digest}-{suffix}")
        suffix += 1
    shutil.copy2(output, candidate)
    return candidate


def _write_atomic(rendered: str, output: Path, replace: bool) -> None:
    data = rendered.encode("utf-8")
    existed = output.exists()
    if existed and not output.is_file():
        raise FileExistsError(f"output is not a regular file: {output}")
    if existed:
        if output.read_bytes() == data:
            return
        if not replace:
            raise FileExistsError(f"output exists; pass --replace: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if existed:
            _backup(output)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_report(manifest_path: Path, output: Path, *, replace: bool = False) -> str:
    """Validate, render and atomically write a report."""
    manifest_path, output = Path(manifest_path), Path(output)
    if manifest_path.resolve() == output.resolve():
        raise ReportError("output cannot overwrite the manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ReportError(f"invalid JSON manifest: {error}") from error
    rendered = render_report(manifest, manifest_path, output)
    _write_atomic(rendered, output, replace)
    return rendered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        build_report(arguments.manifest, arguments.output, replace=arguments.replace)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
