"""Behavioral checks for the bounded A0 static report builder."""

import json
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from tools.plots import build_a0_report as report


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids: list[str] = []
        self.links: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        for name in ("href", "src"):
            if name in values:
                self.links.append(values[name])

    def handle_data(self, data):
        self.text.append(data)


def parse_document(html: str) -> Document:
    document = Document()
    document.feed(html)
    return document


@pytest.fixture
def case(tmp_path):
    root = tmp_path / "case"
    assets = root / "assets"
    assets.mkdir(parents=True)
    (root / "archive").mkdir()
    (root / "known-issue.html").write_text("Known issue")
    image = b"\x89PNG\r\n\x1a\nnot-a-real-image-but-preserved"
    for number in (17, 18, 19, 20, 21):
        (assets / f"figure-{number}.png").write_bytes(image)
    (assets / "figure-17.svg").write_text("<svg/>")
    (assets / "figure-17-data.json").write_text("{}")
    (assets / "figure-17-provenance.json").write_text("{}")
    figures = {
        17: {
            "number": 17,
            "title": "Nonlinear evolution",
            "source_kind": "nonlinear_evolution",
            "image": "assets/figure-17.png",
            "svg": "assets/figure-17.svg",
            "data": "assets/figure-17-data.json",
            "provenance": "assets/figure-17-provenance.json",
            "caption": "Supplied time-evolution evidence.",
            "grid": {"n_r": 64, "n_z": 128},
            "n_phi": 3,
            "k": 96,
            "n_phi_quadrature": 5,
            "device": "cpu:0",
            "status": "available",
            "panels": [
                {"letter": "a", "description": "Stored initial state."},
                {"letter": "b", "description": "Stored final state."},
            ],
        },
        18: {
            "number": 18,
            "title": "Production linearization",
            "source_kind": "production_linearization",
            "image": "assets/figure-18.png",
            "caption": "Discrete Jacobian spectrum.",
            "device": "cuda:0",
            "status": "available",
        },
        19: {
            "number": 19,
            "title": "Independent reference",
            "source_kind": "independent_reference",
            "image": "assets/figure-19.png",
            "caption": "Independent model result.",
            "status": "available",
        },
        20: {
            "number": 20,
            "title": "Verification detail",
            "source_kind": "verification",
            "image": "assets/figure-20.png",
            "caption": "Detailed verification evidence.",
            "status": "unavailable: render pending parent review",
            "collapsed": True,
        },
        21: {
            "number": 21,
            "title": "Mixed comparison",
            "source_kind": "mixed_comparison",
            "components": ["nonlinear phase map", "linearized reference"],
            "image": "assets/figure-21.png",
            "caption": "Comparison of supplied components.",
            "status": "nonterminal",
        },
    }
    manifest = {
        "title": "A0 <b>benchmark</b> report",
        "introduction": "<script>bad()</script> & literal introduction",
        "notation": [
            {"symbol": "N_R", "meaning": "radial grid points"},
            {"symbol": "n_phi", "meaning": "toroidal harmonic"},
        ],
        "archive": {"label": "Retained archive", "path": "archive"},
        "pages": [
            {
                "id": "overview",
                "title": "Overview",
                "introduction": "First logical page.",
                "figures": [figures[17], figures[18]],
                "notes": ["Rendered from supplied files."],
            },
            {
                "id": "results",
                "title": "Results",
                "introduction": "Second logical page.",
                "figures": [figures[19], figures[20], figures[21]],
                "notes": [],
            },
        ],
        "conclusions": "Supplied conclusions; no new numerical result.",
        "known_issues": [
            {"label": "Known issue", "path": "known-issue.html"},
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "output": root / "new-output" / "report.html",
        "assets": assets,
    }


def rewrite_manifest(case, manifest=None):
    case["manifest_path"].write_text(json.dumps(manifest or case["manifest"]))


def assert_local_links_exist(document, output: Path):
    ids = set(document.ids)
    for value in document.links:
        parsed = urlparse(value)
        assert not parsed.scheme
        assert not parsed.netloc
        if value.startswith("#"):
            assert unquote(parsed.fragment) in ids
            continue
        assert parsed.path
        target = output.parent / unquote(parsed.path)
        assert target.exists(), value


def build(case):
    report.build_report(case["manifest_path"], case["output"])
    return case["output"].read_text()


def test_render_escapes_prose_and_uses_unique_page_and_figure_ids(case):
    html = build(case)
    document = parse_document(html)
    visible = "".join(document.text)
    assert "&lt;script&gt;bad()&lt;/script&gt;" in html
    assert "<script>bad()</script>" not in html
    assert "A0 <b>benchmark</b> report" in visible
    assert len(document.ids) == len(set(document.ids))
    assert {"report-index", "page-overview", "page-results"} <= set(document.ids)
    assert {"figure-17", "figure-18", "figure-19", "figure-20", "figure-21"} <= set(
        document.ids
    )
    assert "Figure 17" in visible
    assert "Figure 21" in visible


def test_render_source_badges_do_not_depend_on_file_names(case):
    html = build(case)
    expected = (
        "NL | FOMHD time evolution",
        "LIN | FOMHD discrete Jacobian",
        "REF | independent model",
        "VER | verification",
        "MIX | explicit components",
    )
    for badge in expected:
        assert badge in html
    assert "nonlinear phase map" in html
    assert "linearized reference" in html
    assert 'class="source-badge nl"' in html
    assert 'class="source-badge lin"' in html
    assert 'class="source-badge ref"' in html


def test_render_keeps_grid_harmonic_and_quadrature_fields_separate(case):
    html = build(case)
    assert "<dt>N_R</dt><dd>64</dd>" in html
    assert "<dt>N_Z</dt><dd>128</dd>" in html
    assert "<dt>n_phi (toroidal harmonic)</dt><dd>3</dd>" in html
    assert "<dt>K (retained harmonic cutoff)</dt><dd>96</dd>" in html
    assert "<dt>N_phi quadrature</dt><dd>5</dd>" in html
    assert "K quadrature" not in html
    assert "(a)" in html and "Stored initial state." in html


def test_render_navigation_links_print_css_and_status_labels(case):
    html = build(case)
    document = parse_document(html)
    visible = "".join(document.text)
    assert "Page 1 / 2" in visible
    assert "Page 2 / 2" in visible
    assert visible.count("Index") >= 2
    assert visible.count("Previous") >= 2
    assert visible.count("Next") >= 2
    assert "@page" in html and "counter(page)" in html
    assert "Limited/provisional evidence:" in visible
    assert "render pending parent review" in visible
    assert "<details" in html and "<summary>" in html
    assert "max-width: 1100px" in html
    assert not any(
        name in html
        for name in (
            "filter:",
            "object-fit:",
            "clip-path:",
            "transform:",
        )
    )
    assert_local_links_exist(document, case["output"])


@pytest.mark.parametrize(
    ("status", "cautious"),
    [
        ("running", True),
        ("initializing", True),
        ("budget_stopped", True),
        ("completed", False),
    ],
    ids=["running", "initializing", "budget-stopped", "completed"],
)
def test_limited_statuses_are_visible_without_rewriting_literal_text(
    case, status, cautious
):
    manifest = deepcopy(case["manifest"])
    manifest["pages"] = [manifest["pages"][0]]
    manifest["pages"][0]["figures"] = [manifest["pages"][0]["figures"][0]]
    manifest["pages"][0]["figures"][0]["status"] = status
    rewrite_manifest(case, manifest)

    visible = "".join(parse_document(build(case)).text)
    prefix = "Limited/provisional evidence:"
    assert status in visible
    if cautious:
        assert f"{prefix} {status}" in visible
    else:
        assert prefix not in visible


@pytest.mark.parametrize("number", (17, 18, 19, 20, 21))
def test_build_preserves_underlying_image_bytes(case, number):
    before = (case["assets"] / f"figure-{number}.png").read_bytes()
    build(case)
    assert (case["assets"] / f"figure-{number}.png").read_bytes() == before


def test_unknown_source_kind_is_rejected_before_output_is_touched(case):
    manifest = deepcopy(case["manifest"])
    manifest["pages"][0]["figures"][0]["source_kind"] = "an-inference-bomb"
    rewrite_manifest(case, manifest)
    with pytest.raises(ValueError, match="source_kind"):
        report.build_report(case["manifest_path"], case["output"])
    assert not case["output"].parent.exists()


@pytest.mark.parametrize("missing_field", ("image", "provenance"))
def test_missing_required_or_declared_figure_file_is_rejected(case, missing_field):
    manifest = deepcopy(case["manifest"])
    figure = manifest["pages"][0]["figures"][0]
    (case["root"] / figure[missing_field]).unlink()
    rewrite_manifest(case, manifest)
    with pytest.raises(ValueError, match=missing_field):
        report.build_report(case["manifest_path"], case["output"])
    assert not case["output"].exists()


@pytest.mark.parametrize("duplicate", ("page", "figure"))
def test_duplicate_ids_and_global_figure_numbers_are_rejected(case, duplicate):
    manifest = deepcopy(case["manifest"])
    if duplicate == "page":
        manifest["pages"][1]["id"] = manifest["pages"][0]["id"]
    else:
        manifest["pages"][1]["figures"][0]["number"] = 17
    rewrite_manifest(case, manifest)
    with pytest.raises(ValueError, match=duplicate):
        report.build_report(case["manifest_path"], case["output"])
    assert not case["output"].exists()


def test_no_replace_preserves_existing_report(case, monkeypatch):
    first = build(case)
    changed = deepcopy(case["manifest"])
    changed["title"] = "Changed report"
    rewrite_manifest(case, changed)
    with pytest.raises(FileExistsError):
        report.build_report(case["manifest_path"], case["output"])
    assert case["output"].read_text() == first
    assert not list(case["output"].parent.glob("report.html.bak-*"))
    rewrite_manifest(case)
    monkeypatch.setattr(report.os, "replace", lambda *args: pytest.fail("replaced"))
    report.build_report(case["manifest_path"], case["output"], replace=True)


def test_replace_creates_timestamp_and_hash_named_backup(case):
    first = build(case)
    image_before = (case["assets"] / "figure-17.png").read_bytes()
    changed = deepcopy(case["manifest"])
    changed["title"] = "Replacement title"
    rewrite_manifest(case, changed)
    report.build_report(case["manifest_path"], case["output"], replace=True)
    backups = list(case["output"].parent.glob("report.html.bak-*"))
    assert len(backups) == 1
    assert backups[0].read_text() == first
    assert "Replacement title" in case["output"].read_text()
    assert (case["assets"] / "figure-17.png").read_bytes() == image_before


def test_atomic_failure_keeps_existing_output_and_removes_temporary(case, monkeypatch):
    first = build(case)
    changed = deepcopy(case["manifest"])
    changed["title"] = "Must not land"
    rewrite_manifest(case, changed)

    def fail_replace(source, destination):
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(report.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        report.build_report(case["manifest_path"], case["output"], replace=True)
    assert case["output"].read_text() == first
    assert not list(case["output"].parent.glob(".*.tmp"))
