#!/usr/bin/env python3
"""Build a bounded, offline A0-style T2-E field gallery."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

BLUE = "#0072B2"
ORANGE = "#D55E00"
ACCENT = "#00796B"
DARK = "#37474F"
DEFAULT_COLORS = (BLUE, ORANGE, ACCENT, DARK)
SOURCE_KINDS = {
    "nonlinear_evolution",
    "production_linearization",
    "independent_reference",
    "verification",
    "mixed_comparison",
}
RESERVED_IDS = {
    "main",
    "report-title",
    "introduction",
    "notation",
    "archive",
    "report-index",
    "conclusions",
}
ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
STYLE_RE = re.compile(r"<style>(.*?)</style>", re.DOTALL)
CJK_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uac00-\ud7af\uf900-\ufaff"
    r"\U00020000-\U0003134f]"
)


class InputError(ValueError):
    """Raised when reviewed gallery data violates the renderer contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sources(source_root: Path) -> tuple[Any, Any]:
    plot_dir = source_root / "tools/plots"
    a0_path = plot_dir / "build_a0_report.py"
    t0_path = plot_dir / "build_a1_t0_report.py"
    if not a0_path.is_file() or not t0_path.is_file():
        raise InputError(f"source root does not contain both canonical builders: {source_root}")
    a0 = load_module("a1_t2_a0_renderer", a0_path)
    t0 = load_module("a1_t2_t0_style", t0_path)
    for function in (
        "configure_matplotlib",
        "style_axis",
        "panel_label",
        "save_figure",
    ):
        if not callable(getattr(t0, function, None)):
            raise InputError(f"A1 T0 builder does not expose {function}")
    if not callable(getattr(a0, "render_report", None)):
        raise InputError("A0 builder does not expose render_report")
    if set(a0.SOURCE_BADGES) != SOURCE_KINDS:
        raise InputError("A0 source badge enum differs from the expected enum")
    return a0, t0


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    return value


def _items(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise InputError(f"{where} must be an array")
    return value


def _text(mapping: dict[str, Any], key: str, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}.{key} must be nonempty text")
    if CJK_RE.search(value):
        raise InputError(f"{where}.{key} must be English")
    return value


def _optional_text(mapping: dict[str, Any], key: str, where: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where}.{key} must be nonempty text when supplied")
    if CJK_RE.search(value):
        raise InputError(f"{where}.{key} must be English")
    return value


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{where} must be finite")
    return result


def _number_list(value: Any, where: str) -> list[float]:
    values = _items(value, where)
    if not values:
        raise InputError(f"{where} must not be empty")
    return [
        _finite_number(item, f"{where}[{index}]")
        for index, item in enumerate(values)
    ]


def _validate_json_values(value: Any, where: str) -> None:
    if isinstance(value, str):
        if CJK_RE.search(value):
            raise InputError(f"{where} must be English")
    elif isinstance(value, float) and not math.isfinite(value):
        raise InputError(f"{where} must be finite")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise InputError(f"{where} keys must be strings")
            _validate_json_values(key, f"{where} key")
            _validate_json_values(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_values(item, f"{where}[{index}]")


def _series(raw: Any, where: str) -> None:
    series = _object(raw, where)
    _text(series, "label", where)
    kind = series.get("kind", "line")
    if kind != "line":
        raise InputError(f"{where}.kind must be 'line'; bar plots are not implemented")
    x_values = _number_list(series.get("x"), f"{where}.x")
    y_values = _number_list(series.get("y"), f"{where}.y")
    if len(x_values) != len(y_values):
        raise InputError(f"{where}.x and {where}.y must have equal lengths")
    for key in ("color", "marker", "linestyle"):
        _optional_text(series, key, where)


def _panel(raw: Any, where: str) -> None:
    panel = _object(raw, where)
    _text(panel, "title", where)
    _text(panel, "xlabel", where)
    _text(panel, "ylabel", where)
    for key in ("xscale", "yscale"):
        scale = _optional_text(panel, key, where)
        if scale is not None and scale not in {"linear", "log", "symlog", "logit"}:
            raise InputError(f"{where}.{key} is not a supported Matplotlib scale")
    series = _items(panel.get("series"), f"{where}.series")
    if not series:
        raise InputError(f"{where}.series must not be empty")
    for index, item in enumerate(series):
        _series(item, f"{where}.series[{index}]")
    if panel.get("horizontal_lines") is None:
        return
    for index, raw_line in enumerate(
        _items(panel["horizontal_lines"], f"{where}.horizontal_lines")
    ):
        line_where = f"{where}.horizontal_lines[{index}]"
        line = _object(raw_line, line_where)
        _finite_number(line.get("y"), f"{line_where}.y")
        _text(line, "label", line_where)
        _optional_text(line, "color", line_where)


def _figure(raw: Any, where: str) -> dict[str, Any]:
    figure = _object(raw, where)
    figure_id = _text(figure, "id", where)
    if not ID_RE.fullmatch(figure_id):
        raise InputError(f"{where}.id must be an HTML-safe identifier")
    if figure_id in RESERVED_IDS:
        raise InputError(f"{where}.id is reserved")
    _text(figure, "title", where)
    source_kind = _text(figure, "source_kind", where)
    if source_kind not in SOURCE_KINDS:
        raise InputError(f"{where}.source_kind is not an A0 evidence kind")
    if figure.get("components") is not None:
        components = _items(figure["components"], f"{where}.components")
        if not components:
            raise InputError(f"{where}.components must not be empty when supplied")
        for index, component in enumerate(components):
            if not isinstance(component, str) or not component.strip():
                raise InputError(
                    f"{where}.components[{index}] must be nonempty text"
                )
    elif source_kind == "mixed_comparison":
        raise InputError(f"{where}.components is required for mixed_comparison")
    for key in ("caption", "status", "device", "question", "controls", "criterion"):
        _text(figure, key, where)
    panels = _items(figure.get("panels"), f"{where}.panels")
    if not 1 <= len(panels) <= 4:
        raise InputError(f"{where}.panels must contain one to four panels")
    for index, panel in enumerate(panels):
        _panel(panel, f"{where}.panels[{index}]")
    return figure


def validate_data(data: Any) -> dict[str, Any]:
    """Validate the parent-reviewed data contract without changing values."""
    root = _object(data, "data")
    _validate_json_values(root, "data")
    for key in ("title", "introduction", "conclusions"):
        _text(root, key, "data")

    known_issues = _items(root.get("known_issues"), "data.known_issues")
    for index, raw in enumerate(known_issues):
        where = f"data.known_issues[{index}]"
        issue = _object(raw, where)
        _text(issue, "label", where)
        _text(issue, "path", where)

    archive = _object(root.get("archive"), "data.archive")
    _text(archive, "label", "data.archive")
    _text(archive, "path", "data.archive")

    notation = _items(root.get("notation"), "data.notation")
    if not notation:
        raise InputError("data.notation must not be empty")
    for index, raw in enumerate(notation):
        where = f"data.notation[{index}]"
        item = _object(raw, where)
        _text(item, "symbol", where)
        _text(item, "meaning", where)

    figures = _items(root.get("figures"), "data.figures")
    if not figures:
        raise InputError("data.figures must not be empty")
    seen: set[str] = set()
    for index, raw in enumerate(figures):
        figure = _figure(raw, f"data.figures[{index}]")
        if figure["id"] in seen:
            raise InputError(f"duplicate figure id: {figure['id']}")
        seen.add(figure["id"])
    return root


def load_json(data_path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise InputError(f"non-finite JSON constant is not allowed: {value}")

    try:
        value = json.loads(
            data_path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise InputError(f"invalid JSON data: {error}") from error
    return _object(value, "data")


def render_figures(data: dict[str, Any], output: Path, t0: Any) -> None:
    plt = t0.configure_matplotlib()
    for figure in data["figures"]:
        panels = figure["panels"]
        columns = 1 if len(panels) == 1 else 2
        rows = math.ceil(len(panels) / columns)
        fig, grid = plt.subplots(
            rows,
            columns,
            figsize=(11.8, 4.8 if rows == 1 else 8.2),
            layout="constrained",
            squeeze=False,
        )
        try:
            fig.suptitle(figure["title"], fontsize=10.5)
            axes = list(grid.flat)
            for index, panel in enumerate(panels):
                axis = axes[index]
                for series_index, series in enumerate(panel["series"]):
                    axis.plot(
                        series["x"],
                        series["y"],
                        color=series.get("color")
                        or DEFAULT_COLORS[series_index % len(DEFAULT_COLORS)],
                        marker=series.get("marker"),
                        linestyle=series.get("linestyle") or "-",
                        label=series["label"],
                    )
                for line in panel.get("horizontal_lines", []):
                    axis.axhline(
                        line["y"],
                        color=line.get("color") or DARK,
                        linestyle="--",
                        label=line["label"],
                    )
                if panel.get("xscale") is not None:
                    axis.set_xscale(panel["xscale"])
                if panel.get("yscale") is not None:
                    axis.set_yscale(panel["yscale"])
                axis.set_title(panel["title"])
                axis.set_xlabel(panel["xlabel"])
                axis.set_ylabel(panel["ylabel"])
                axis.legend(loc="best")
                t0.style_axis(axis)
                t0.panel_label(axis, chr(ord("a") + index))
            for axis in axes[len(panels) :]:
                axis.set_visible(False)
            t0.save_figure(fig, output, figure["id"])
        finally:
            plt.close(fig)


def _manifest_link(
    value: str,
    data_dir: Path,
    data_path: Path,
    output: Path,
    where: str,
) -> str:
    source = Path(value)
    target = (source if source.is_absolute() else data_dir / source).resolve()
    if not target.exists():
        raise InputError(f"{where} local path does not exist: {target}")
    if target == output.resolve():
        raise InputError(f"{where} cannot link to the output directory")
    if target == data_path.resolve():
        return "data.json"
    return Path(os.path.relpath(target, output)).as_posix()


def build_manifest(data: dict[str, Any], data_path: Path, output: Path) -> dict[str, Any]:
    data_dir = data_path.parent
    figure_records = []
    pages = []
    for number, figure in enumerate(data["figures"], start=1):
        panel_records = [
            {
                "letter": chr(ord("a") + index),
                "description": (
                    f"{panel['title']}; x-axis: {panel['xlabel']}; "
                    f"y-axis: {panel['ylabel']}."
                ),
            }
            for index, panel in enumerate(figure["panels"])
        ]
        record: dict[str, Any] = {
            "number": number,
            "title": figure["title"],
            "caption": (
                f"Question: {figure['question']} "
                f"Controls: {figure['controls']} "
                f"Criterion: {figure['criterion']} "
                f"Supplied caption: {figure['caption']}"
            ),
            "source_kind": figure["source_kind"],
            "status": figure["status"],
            "device": figure["device"],
            "panels": panel_records,
            "image": f"{figure['id']}.png",
            "svg": f"{figure['id']}.svg",
            "data": "data.json",
        }
        if figure.get("components") is not None:
            record["components"] = list(figure["components"])
        figure_records.append(record)
        pages.append(
            {
                "id": figure["id"],
                "title": figure["title"],
                "introduction": figure["question"],
                "figures": [record],
            }
        )

    known_issues = []
    for index, issue in enumerate(data["known_issues"]):
        where = f"data.known_issues[{index}]"
        known_issues.append(
            {
                "label": issue["label"],
                "path": _manifest_link(
                    issue["path"], data_dir, data_path, output, where
                ),
            }
        )
    return {
        "title": data["title"],
        "introduction": data["introduction"],
        "notation": [
            {"symbol": item["symbol"], "meaning": item["meaning"]}
            for item in data["notation"]
        ],
        "archive": {
            "label": data["archive"]["label"],
            "path": _manifest_link(
                data["archive"]["path"],
                data_dir,
                data_path,
                output,
                "data.archive",
            ),
        },
        "pages": pages,
        "conclusions": data["conclusions"],
        "known_issues": known_issues,
    }


def embed_svgs(
    rendered: str,
    figures: list[dict[str, Any]],
    output: Path,
) -> str:
    for figure in figures:
        name = figure["id"]
        payload = base64.b64encode((output / f"{name}.svg").read_bytes()).decode("ascii")
        data_uri = f"data:image/svg+xml;base64,{payload}"
        needle = f'src="{name}.png"'
        if rendered.count(needle) != 1:
            raise RuntimeError(f"expected one rendered PNG source for {name}")
        rendered = rendered.replace(needle, f'src="{data_uri}"')
    if any(f'src="{figure["id"]}.png"' in rendered for figure in figures):
        raise RuntimeError("a raster image source remains in the gallery HTML")
    return rendered


def _style(path: Path) -> bytes:
    match = STYLE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise RuntimeError(f"no embedded style block found in {path}")
    return match.group(1).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_output(
    data: dict[str, Any],
    output: Path,
    renderer_path: Path,
) -> None:
    html_path = output / "index.html"
    html = html_path.read_text(encoding="utf-8")
    if _style(html_path) != _style(renderer_path):
        raise RuntimeError("rendered HTML CSS differs from the A0 renderer CSS")
    if CJK_RE.search(html):
        raise RuntimeError("rendered HTML contains CJK text")
    if "http://" in html.casefold() or "https://" in html.casefold():
        raise RuntimeError("rendered HTML contains a remote asset")
    for figure in data["figures"]:
        name = figure["id"]
        png = output / f"{name}.png"
        svg = output / f"{name}.svg"
        if not png.is_file() or png.stat().st_size == 0:
            raise RuntimeError(f"PNG output is missing or empty: {png}")
        if not svg.is_file() or svg.stat().st_size == 0:
            raise RuntimeError(f"SVG output is missing or empty: {svg}")
        payload = base64.b64encode(svg.read_bytes()).decode("ascii")
        if html.count(f'src="data:image/svg+xml;base64,{payload}"') != 1:
            raise RuntimeError(f"HTML does not embed the exported SVG exactly: {svg}")
        if f'href="{name}.svg"' not in html:
            raise RuntimeError(f"HTML lacks the raw SVG artifact link: {svg}")
        if 'href="data.json"' not in html:
            raise RuntimeError("HTML lacks a reviewed-data artifact link")


def build(data_path: Path, output: Path, source_root: Path) -> dict[str, Any]:
    data_path, output, source_root = (
        data_path.resolve(),
        output.resolve(),
        source_root.resolve(),
    )
    if output.exists():
        raise FileExistsError(f"refusing existing output path: {output}")
    if not data_path.is_file():
        raise InputError(f"data path is not a file: {data_path}")

    a0, t0 = load_sources(source_root)
    renderer_path = source_root / "tools/plots/build_a0_report.py"
    style_builder_path = source_root / "tools/plots/build_a1_t0_report.py"
    renderer_hash = sha256_file(renderer_path)
    style_builder_hash = sha256_file(style_builder_path)
    data_hash = sha256_file(data_path)
    data = validate_data(load_json(data_path))
    manifest = build_manifest(data, data_path, output)

    output.mkdir(parents=True, exist_ok=False)
    old_mplconfig = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="a1-t2-matplotlib-") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        try:
            render_figures(data, output, t0)
        finally:
            if old_mplconfig is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_mplconfig

    shutil.copyfile(data_path, output / "data.json")
    manifest_path = output / "manifest.json"
    _write_json(manifest_path, manifest)
    html_path = output / "index.html"
    rendered = a0.render_report(manifest, manifest_path, html_path)
    rendered = embed_svgs(rendered, data["figures"], output)
    html_path.write_text(rendered, encoding="utf-8")

    if sha256_file(data_path) != data_hash:
        raise RuntimeError("reviewed input data changed during rendering")
    if sha256_file(renderer_path) != renderer_hash:
        raise RuntimeError("canonical A0 renderer changed during rendering")
    if sha256_file(style_builder_path) != style_builder_hash:
        raise RuntimeError("canonical A1 T0 style builder changed during rendering")
    if (output / "data.json").read_bytes() != data_path.read_bytes():
        raise RuntimeError("copied data.json is not byte-identical to reviewed input")
    verify_output(data, output, renderer_path)
    return {
        "status": "ok",
        "output": str(output),
        "figures": len(data["figures"]),
        "panels": sum(len(figure["panels"]) for figure in data["figures"]),
        "html_bytes": html_path.stat().st_size,
        "data_sha256": data_hash,
        "renderer_sha256": renderer_hash,
        "style_builder_sha256": style_builder_hash,
        "css_sha256": hashlib.sha256(_style(html_path)).hexdigest(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_root = (
        args.source_root
        or Path(__file__).resolve().parents[2]
    ).resolve()
    try:
        result = build(args.data, args.output, source_root)
    except FileExistsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (InputError, KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
