#!/usr/bin/env python3
"""Assert-based offline checks for an A1 T0 A0-style report."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote

STYLE_RE = re.compile(r"<style>(.*?)</style>", re.DOTALL)
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
FIGURE_NUMBERS = list(range(1, 8))
PAGE_IDS = [
    "scope-protocol",
    "resolved-frequency",
    "micro-response",
    "target-accuracy",
    "high-mode",
    "fixed-H-and-energy",
    "linear-qualification-cost",
    "legacy-and-next-question",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("a1_t0_builder_under_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"cannot load builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (sha256_file(path), path.stat().st_size)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ReportScan(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.image_sources: list[str] = []
        self.article_ids: list[str] = []
        self.figure_ids: list[str] = []
        self.heading_texts: list[tuple[str, str]] = []
        self.badge_count = 0
        self.script_count = 0
        self.link_count = 0
        self._heading: str | None = None
        self._heading_chunks: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            self.ids.append(element_id)
            if tag == "article" and element_id.startswith("page-"):
                self.article_ids.append(element_id)
            if tag == "figure" and element_id.startswith("figure-"):
                self.figure_ids.append(element_id)
        if tag == "a" and values.get("href"):
            self.hrefs.append(str(values["href"]))
        if tag == "img":
            self.image_sources.append(str(values.get("src", "")))
        if tag in {"h2", "h3"}:
            self._heading = tag
            self._heading_chunks = []
        if tag == "span":
            classes = str(values.get("class", "")).split()
            if "source-badge" in classes:
                self.badge_count += 1
        if tag == "script":
            self.script_count += 1
        if tag == "link":
            self.link_count += 1

    def handle_data(self, data: str) -> None:
        if self._heading is not None:
            self._heading_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._heading == tag:
            self.heading_texts.append(
                (tag, " ".join("".join(self._heading_chunks).split()))
            )
            self._heading = None
            self._heading_chunks = []


def parse_png(path: Path) -> tuple[int, int, float | None]:
    payload = path.read_bytes()
    assert payload.startswith(b"\x89PNG\r\n\x1a\n"), f"invalid PNG signature: {path}"
    offset = 8
    width = height = None
    dpi_x = None
    while offset < len(payload):
        assert offset + 12 <= len(payload), f"truncated PNG chunk: {path}"
        length = int.from_bytes(payload[offset : offset + 4], "big")
        kind = payload[offset + 4 : offset + 8]
        body = payload[offset + 8 : offset + 8 + length]
        assert offset + 12 + length <= len(payload), f"truncated PNG body: {path}"
        if kind == b"IHDR":
            assert length == 13, f"invalid PNG IHDR: {path}"
            width = int.from_bytes(body[0:4], "big")
            height = int.from_bytes(body[4:8], "big")
        if kind == b"pHYs":
            assert length == 9, f"invalid PNG pHYs: {path}"
            pixels_per_meter = int.from_bytes(body[0:4], "big")
            dpi_x = pixels_per_meter * 0.0254
        offset += 12 + length
        if kind == b"IEND":
            break
    assert width is not None and height is not None, f"PNG has no IHDR: {path}"
    return width, height, dpi_x


def check_svg(path: Path) -> None:
    payload = path.read_bytes()
    root = ET.fromstring(payload)
    assert root.tag.rsplit("}", 1)[-1] == "svg", f"invalid SVG root: {path}"
    local_tags = {element.tag.rsplit("}", 1)[-1] for element in root.iter()}
    assert "text" in local_tags, f"SVG has no text elements: {path}"
    assert "image" not in local_tags, f"SVG contains an embedded raster image: {path}"


def check_report_scan(scan: ReportScan, html_text: str, output: Path) -> None:
    assert len(scan.ids) == len(set(scan.ids)), "HTML contains duplicate ids"
    assert scan.article_ids == [f"page-{page_id}" for page_id in PAGE_IDS]
    assert scan.figure_ids == [
        f"figure-{number}" for number in FIGURE_NUMBERS
    ]
    assert scan.badge_count == 7, "HTML must contain seven evidence badges"
    assert scan.script_count == 0, "HTML must not contain scripts"
    assert scan.link_count == 0, "HTML must not load linked runtime assets"
    assert "http://" not in html_text.casefold(), "HTML contains a remote HTTP asset"
    assert "https://" not in html_text.casefold(), "HTML contains a remote HTTPS asset"
    assert "//cdn" not in html_text.casefold(), "HTML contains a CDN reference"
    assert not CJK_RE.search(html_text), "HTML contains non-English CJK text"

    ids = set(scan.ids)
    for href in scan.hrefs:
        if href.startswith("#"):
            assert href[1:] in ids, f"broken fragment href: {href}"
            continue
        assert "://" not in href, f"remote href is not allowed: {href}"
        local = unquote(href.split("#", 1)[0])
        assert (output / local).exists(), f"missing local HTML link target: {href}"

    assert len(scan.image_sources) == 7
    assert all(
        source.startswith("data:image/svg+xml;base64,")
        for source in scan.image_sources
    )
    embedded = {
        base64.b64decode(source.split(",", 1)[1])
        for source in scan.image_sources
    }
    svg_files = {
        path.read_bytes()
        for path in output.glob("*.svg")
        if path.is_file()
    }
    assert embedded == svg_files, "HTML embedded SVGs differ from exported SVGs"

    figure_headings = [
        text
        for tag, text in scan.heading_texts
        if tag == "h3" and text.startswith("Figure ")
    ]
    assert len(figure_headings) == 7, "HTML must contain seven figure headings"
    for number, heading in zip(FIGURE_NUMBERS, figure_headings, strict=True):
        assert heading.startswith(f"Figure {number}:"), heading


def check_manifest(
    manifest: dict[str, Any],
    output: Path,
    builder: Any,
) -> None:
    pages = manifest["pages"]
    assert len(pages) == 8
    assert [page["id"] for page in pages] == PAGE_IDS
    figures = [figure for page in pages for figure in page.get("figures", [])]
    assert [figure["number"] for figure in figures] == FIGURE_NUMBERS
    assert len({figure["title"] for figure in figures}) == 7
    for figure in figures:
        assert (output / figure["image"]).is_file()
        assert (output / figure["svg"]).is_file()
        assert (output / figure["data"]).is_file()
        assert (output / figure["provenance"]).is_file()
        assert figure["caption"].strip()
        assert figure["source_kind"] in {
            "nonlinear_evolution",
            "production_linearization",
            "independent_reference",
            "verification",
            "mixed_comparison",
        }
    assert "LIN" in figures[6]["caption"]
    assert "equal-accuracy speedup" in figures[6]["caption"]
    assert "last accepted sample" in figures[3]["caption"]
    assert "absolute" in figures[3]["caption"]
    expected = builder.build_manifest(
        json.loads((output / "data.json").read_text(encoding="utf-8"))
    )
    assert manifest == expected, "manifest differs from source-data reconstruction"


def check_values(data: dict[str, Any], builder: Any) -> None:
    target_cases = builder.target_accuracy_cases(data)
    expected_target = {
        case["id"]
        for case in data["cases"]
        if case["seed"] == "target"
        and case["completed"]
        and case["max_target_error"] is not None
        and case["max_phase_rad"] is not None
        and case["max_amplitude_drift"] is not None
    }
    assert {case["id"] for case in target_cases} == expected_target
    boundary = {"completed": False, "rejected_time_over_T": 1.0}
    assert builder._accepted_indices(boundary, [0.0, 0.5, 1.0, 1.5]) == [0, 1]
    boundary["time_over_T"] = [0.0, 0.5, 1.0, 1.5]
    assert builder.accepted_horizon(boundary) == 0.5
    rejected = [case for case in data["cases"] if not case["completed"]]
    for case in rejected:
        if case["rejected_time_over_T"] is None:
            continue
        times, norms = builder.accepted_history(
            case,
            "time_over_T",
            "high_absolute_norm",
        )
        assert len(times) == len(norms)
        assert all(
            time < case["rejected_time_over_T"] for time in times
        )
        energy_times, residuals = builder.accepted_history(
            case,
            "energy_time_over_T",
            "total_energy_residual_J",
        )
        assert len(energy_times) == len(residuals)
        assert all(
            time < case["rejected_time_over_T"]
            for time in energy_times
        )
    for case in target_cases:
        assert case["max_target_error"] == data["cases"][
            data["cases"].index(case)
        ]["max_target_error"]
        assert case["max_phase_rad"] is not None
        assert case["max_amplitude_drift"] is not None


def check_provenance(
    data_path: Path,
    output: Path,
    builder_path: Path,
) -> None:
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "complete"
    assert provenance["compact_input"]["path"] == str(data_path.resolve())
    assert provenance["compact_input"]["sha256"] == sha256_file(data_path)
    assert provenance["builder"]["path"] == str(builder_path.resolve())
    assert provenance["builder"]["sha256"] == sha256_file(builder_path)
    assert provenance["render_policy"]["runtime_network_used"] is False
    assert provenance["render_policy"]["solver_used"] is False
    assert provenance["render_policy"]["figures"] == 7
    assert provenance["render_policy"]["png_dpi"] == 160
    assert provenance["render_policy"]["svg_fonttype"] == "none"
    for item in provenance["verified_input_provenance"]:
        path = Path(item["path"])
        assert path.is_file(), f"provenance input disappeared: {path}"
        assert item["sha256"] == sha256_file(path)
    for key in ("renderer", "a0_html"):
        path = Path(provenance["canonical_sources"][f"{key}_path"])
        assert path.is_file()
        assert (
            provenance["canonical_sources"][f"{key}_sha256"] == sha256_file(path)
        )
    for full in ("plot_tokamak_cae", "plot_cae_comparison"):
        path = Path(provenance["canonical_sources"][f"{full}_path"])
        assert path.is_file()
        assert (
            provenance["canonical_sources"][f"{full}_sha256"] == sha256_file(path)
        )
    records = {item["path"]: item for item in provenance["outputs"]}
    expected_outputs = {
        "a1-benchmark.html",
        "manifest.json",
        "data.json",
        *{
            f"{name}.{suffix}"
            for name in (
                "fig01_resolved_frequencies",
                "fig02_micro_response",
                "fig03_target_accuracy",
                "fig04_high_mode_history",
                "fig05_fixed_H_refinement",
                "fig06_energy_residuals",
                "fig07_linear_and_costs",
            )
            for suffix in ("png", "svg")
        },
    }
    assert set(records) == expected_outputs
    for relative, item in records.items():
        path = output / relative
        assert item["sha256"] == sha256_file(path)
        assert item["bytes"] == path.stat().st_size
    assert provenance["copied_data"]["sha256"] == sha256_file(output / "data.json")


def check_refusal(
    data_path: Path,
    output: Path,
    builder_path: Path,
) -> None:
    before = snapshot(output)
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(builder_path),
            "--data",
            str(data_path),
            "--output",
            str(output),
            "--source-root",
            "/home/lxzeng/projects/fomhd",
            "--renderer",
            "/home/lxzeng/projects/fomhd/tools/plots/build_a0_report.py",
            "--a0-html",
            "/home/lxzeng/runs/fomhd/a0-benchmark-figures__20261104T050738Z/index.html",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "refusing existing output path" in result.stderr
    assert snapshot(output) == before, "refusal modified the existing output"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--builder",
        type=Path,
        default=Path(__file__).with_name("build_a1_t0_report.py"),
    )
    parser.add_argument(
        "--a0-html",
        type=Path,
        default=Path(
            "/home/lxzeng/runs/fomhd/"
            "a0-benchmark-figures__20261104T050738Z/index.html"
        ),
    )
    args = parser.parse_args(argv)
    data_path = args.data.resolve()
    output = args.output.resolve()
    builder_path = args.builder.resolve()
    a0_html = args.a0_html.resolve()

    checks: list[str] = []
    assert data_path.is_file()
    assert output.is_dir()
    assert builder_path.is_file()
    assert a0_html.is_file()
    builder = load_module(builder_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    builder.validate_data(data)
    checks.append("compact schema and English fields")

    expected_files = {
        "a1-benchmark.html",
        "manifest.json",
        "provenance.json",
        "data.json",
        *{
            f"{name}.{suffix}"
            for name in builder.FIGURE_NAMES
            for suffix in ("png", "svg")
        },
    }
    actual_files = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files, (
        f"unexpected output files: expected {sorted(expected_files)}, "
        f"found {sorted(actual_files)}"
    )
    assert (output / "data.json").read_bytes() == data_path.read_bytes()
    checks.append("exact output set and byte-identical copied data")

    for number, name in enumerate(builder.FIGURE_NAMES, start=1):
        svg_path = output / f"{name}.svg"
        png_path = output / f"{name}.png"
        check_svg(svg_path)
        width, height, dpi = parse_png(png_path)
        assert width >= 1500 and height >= 600, f"PNG is unexpectedly small: {name}"
        assert dpi is None or math.isclose(dpi, 160.0, rel_tol=0.0, abs_tol=1.0), (
            f"PNG DPI is not approximately 160: {name}: {dpi}"
        )
        assert svg_path.stat().st_size > 1000
        assert png_path.stat().st_size > 10_000
        assert number in FIGURE_NUMBERS
    checks.append("seven structurally readable text-SVG and 160-dpi PNG figures")

    html_text = (output / "a1-benchmark.html").read_text(encoding="utf-8")
    html_bytes = (output / "a1-benchmark.html").read_bytes()
    html_css = STYLE_RE.search(html_text)
    a0_css = STYLE_RE.search(a0_html.read_text(encoding="utf-8"))
    assert html_css is not None and a0_css is not None
    assert html_css.group(1).encode("utf-8") == a0_css.group(1).encode("utf-8")
    scan = ReportScan()
    scan.feed(html_text)
    check_report_scan(scan, html_text, output)
    assert b'<html lang="en">' in html_bytes
    checks.append("eight pages, seven figures, anchors, and embedded SVGs")
    checks.append("canonical A0 CSS byte-for-byte and no runtime network")

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    check_manifest(manifest, output, builder)
    checks.append("manifest structure and evidence language")
    check_values(data, builder)
    checks.append("completed-only endpoints and rejected-prefix boundaries")

    check_provenance(data_path, output, builder_path)
    checks.append("input, script, renderer, style, and output hashes")
    check_refusal(data_path, output, builder_path)
    checks.append("existing-output refusal with no mutation")

    builder_source = builder_path.read_text(encoding="utf-8")
    assert not CJK_RE.search(builder_source)
    banned_imports = (
        "import requests",
        "import socket",
        "import urllib",
        "from urllib",
        "import http",
    )
    assert all(token not in builder_source for token in banned_imports)
    checks.append("English builder source and no network-capable imports")

    for check in checks:
        print(f"PASS: {check}")
    print(f"PASS: {len(checks)} checker groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
