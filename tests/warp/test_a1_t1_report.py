"""Focused synthetic checks for the bounded A1 T1 gallery builder."""

from __future__ import annotations

import json
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from tools.plots import build_a1_t1_report as report


class Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.links: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if "id" in values:
            self.ids.append(values["id"])
        for key in ("href", "src"):
            if key in values:
                self.links.append(values[key])

    def handle_data(self, data):
        self.text.append(data)


def _gain_series(base: float, ratios):
    return [base * ratio for ratio in ratios]


def _case(
    case_id: str,
    M: int,
    seed: str,
    *,
    coupling: str = "waveform",
    micro: int = 2048,
    rtol: float | None = 1.0e-6,
    completed: bool = True,
    high_sync_vector_error: float | None = None,
    final_high_gain: float | None = None,
    max_high_gain: float | None = None,
) -> dict:
    rejection = None if completed else 0.5 * M / micro
    times = [0.0, 0.25 * M / micro, 0.5 * M / micro, 0.75 * M / micro]
    mixed_max_gain = (
        None if seed == "target" else 1.005 if max_high_gain is None else max_high_gain
    )
    ratios = None
    if mixed_max_gain is not None:
        gain_delta = mixed_max_gain - 1.0
        ratios = [
            1.0,
            1.0 + 0.2 * gain_delta,
            1.0 + 0.6 * gain_delta,
            mixed_max_gain,
        ]
    iterations = [
        {
            "macro_index": macro,
            "iteration": iteration,
            "path_max": (1.0e-3 / (iteration + 1)) * (macro + 1),
            "endpoint_max": (1.0e-3 / (iteration + 1)) * (macro + 1),
            "total_energy_residual_J": 1.0e-12 * (macro + 1),
            "kinetic_energy_residual_J": 5.0e-13 * (macro + 1),
        }
        for macro in (0, 1)
        for iteration in (0, 1, 2)
    ]
    result = {
        "id": case_id,
        "coupling": coupling,
        "M": M,
        "seed": seed,
        "micro_per_period": micro,
        "rtol": rtol,
        "completed": completed,
        "status": "COMPLETED" if completed else "REJECTED_SYNTHETIC_CONTROL",
        "time_over_T": times,
        "target_error": [0.0, 1.0e-6, 2.0e-6, 4.0e-6],
        "high_norm_ratio": ratios,
        "high_absolute_norm": _gain_series(1.0e-10, ratios or [1.0, 1.0, 1.0, 1.0]),
        "total_energy_residual_J": [0.0, 1.0e-12, 2.0e-12, 4.0e-12],
        "kinetic_energy_residual_J": [0.0, 5.0e-13, 8.0e-13, 9.0e-13],
        "energy_time_over_T": times,
        "max_target_error": 4.0e-6,
        "max_high_gain": mixed_max_gain,
        "final_high_gain": (
            None
            if seed == "target"
            else 1.005 if final_high_gain is None else final_high_gain
        ),
        "sweeps_min": 2.0,
        "sweeps_max": 3.0,
        "model_calls": 1000 + M,
        "estimated_peak_bytes": 4096.0 + M,
        "elapsed_s": 0.01 + M * 1.0e-5,
        "iterations": iterations,
        "failure_message": None if completed else "synthetic rejection",
        "rejected_time_over_T": rejection,
    }
    if high_sync_vector_error is not None:
        result["high_sync_vector_error"] = high_sync_vector_error
    return result


def _t0_case(M: int, seed: str) -> dict:
    H = M / 2048.0
    times = [0.0, H / 2.0, H]
    ratio = None if seed == "target" else [1.0, 1.01, 1.02]
    completed = M < 256
    return {
        "id": f"t0-{M}-{seed}",
        "M": M,
        "coupling": "predictor",
        "seed": seed,
        "micro_per_period": 2048,
        "H_over_T": H,
        "completed": completed,
        "status": "COMPLETED" if completed else "UNDETERMINED",
        "time_over_T": times,
        "target_error": [0.0, 1.0e-6, 2.0e-6],
        "high_norm_ratio": ratio,
        "high_absolute_norm": (
            [1.0e-10, 1.0e-10, 1.0e-10]
            if seed == "target"
            else [1.0e-10 * value for value in ratio]
        ),
        "max_target_error": 2.0e-6,
        "max_high_gain": None if seed == "target" else 1.02,
        "rejected_time_over_T": None if completed else H / 2.0,
    }


def _compact_data(root: Path) -> dict:
    evidence = root / "evidence"
    baseline = root / "baseline"
    t0_dir = root / "t0-gallery"
    chinese = evidence / "chinese-draft"
    for path in (evidence, baseline, t0_dir, chinese):
        path.mkdir(parents=True, exist_ok=True)
    (t0_dir / "a1-benchmark.html").write_text(
        "<!doctype html><html lang='en'><body>preserved T0</body></html>",
        encoding="utf-8",
    )
    t0_cases = [
        _t0_case(M, seed)
        for M in (4, 8, 16, 32, 64, 128, 256)
        for seed in ("target", "mixed")
    ]
    (t0_dir / "data.json").write_text(
        json.dumps(
            {
                "protocol": {"micro_per_period": 2048},
                "cases": t0_cases,
            }
        ),
        encoding="utf-8",
    )
    (chinese / "a1-benchmark.html").write_text(
        "<!doctype html><html lang='zh'><body>preserved reference draft</body></html>",
        encoding="utf-8",
    )

    cases = []
    for M in (4, 8, 16, 32, 64, 128, 256):
        for seed in ("target", "mixed"):
            cases.append(
                _case(
                    f"primary-{M}-{seed}",
                    M,
                    seed,
                    completed=not (M == 256 and seed == "target"),
                    max_high_gain=(
                        1.0144201
                        if M == 128 and seed == "mixed"
                        else 1.009 if M == 256 and seed == "mixed" else None
                    ),
                )
            )
    for M, micro in ((128, 2048), (256, 4096), (256, 2048), (512, 4096)):
        for seed in ("target", "mixed"):
            cases.append(
                _case(
                    f"fixed-{M}-{micro}-{seed}",
                    M,
                    seed,
                    micro=micro,
                    max_high_gain=(
                        1.0144201
                        if M == 128 and seed == "mixed"
                        else 1.009 if M == 256 and seed == "mixed" else None
                    ),
                )
            )
    cases.append(
        _case("tolerance-low-target", 128, "target", rtol=1.0e-8)
    )
    cases.append(
        _case(
            "tolerance-low-half-mixed",
            256,
            "mixed",
            micro=4096,
            rtol=1.0e-8,
        )
    )
    for M, sync_base in ((16, 2.0e-4), (32, 4.0e-4)):
        for rtol, sync_scale, gain in (
            (1.0e-10, 1.0, 1.02 + M * 1.0e-5),
            (1.0e-14, 0.25, 1.01 + M * 1.0e-5),
        ):
            cases.append(
                _case(
                    f"tight-{M}-{rtol:.0e}-mixed",
                    M,
                    "mixed",
                    rtol=rtol,
                    high_sync_vector_error=sync_base * sync_scale,
                    final_high_gain=gain,
                )
            )

    linear = []
    for M in (4, 8, 16, 32, 64, 128, 256):
        modulus = 1.0 + M * 1.0e-5
        linear.append(
            {
                "M": M,
                "positive_branch_modulus": modulus,
                "eigenmode_power_gain": modulus**M,
                "actual_seed_final_gain": modulus**M * 1.00001,
                "operator_power_norm": modulus**M * 1.00002,
                "production_gain": modulus**M * 1.00003,
            }
        )

    return {
        "protocol": {
            "T_s": 4.053961499776523e-7,
            "h_s": 4.053961499776523e-7 / 2048.0,
            "cycles": 2,
            "rtol": 1.0e-6,
            "max_iterations": 12,
            "target_error_limit": 0.01,
            "phase_limit_rad": 0.1,
            "amplitude_limit": 0.01,
            "growth_limit": 1.01,
            "energy_budget_J": 8.0e-9,
        },
        "t0_gallery": str((t0_dir / "a1-benchmark.html").resolve()),
        "linear": linear,
        "cases": cases,
        "gpu": {
            "status": "bounded parity available",
            "notes": ["Small synthetic parity row only; no speedup is inferred."],
            "parity_rows": [
                {
                    "M": 4,
                    "accepted_macros": 2,
                    "relative_state_error": 1.0e-12,
                    "relative_mode_error": 2.0e-12,
                    "device": "test:0",
                }
            ],
        },
        "provenance": {
            "source_commit": "a" * 40,
            "source_hashes": {"tools/plots/build_a0_report.py": "b" * 64},
            "evidence_root": str(evidence.resolve()),
            "baseline_root": str(baseline.resolve()),
            "notes": [
                "No solver was run for this synthetic fixture.",
                "The preserved Chinese report is linked as reference context only.",
            ],
        },
    }


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "fixture"
    root.mkdir()
    data = _compact_data(root)
    data_path = root / "compact.json"
    data_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {
        "root": root,
        "data": data,
        "data_path": data_path,
        "output": root / "fresh-output",
        "source_root": Path("/home/lxzeng/projects/fomhd"),
        "a0_html": report.A0_HTML_DEFAULT,
    }


def _build(fixture, output=None):
    return report.build(
        fixture["data_path"],
        output or fixture["output"],
        fixture["source_root"] / "tools/plots/build_a0_report.py",
        fixture["a0_html"],
        (
            fixture["source_root"] / "tools/plots/plot_tokamak_cae.py",
            fixture["source_root"] / "tools/plots/plot_cae_comparison.py",
        ),
    )


def _parse(text: str) -> Document:
    document = Document()
    document.feed(text)
    return document


def test_fresh_build_has_assets_css_links_and_unique_ids(fixture):
    source_hashes = {
        path: report.sha256_file(path)
        for path in (
            fixture["data_path"],
            Path(fixture["data"]["t0_gallery"]).parent / "data.json",
            fixture["source_root"] / "tools/plots/build_a0_report.py",
        )
    }
    result = _build(fixture)
    output = fixture["output"]
    expected = {
        "a1-benchmark.html",
        "manifest.json",
        "data.json",
        "provenance.json",
    }
    expected.update(
        f"{name}.{suffix}"
        for name in report.FIGURE_NAMES
        for suffix in ("png", "svg")
    )
    assert expected <= {path.name for path in output.iterdir()}
    assert result["figures"] == 7
    assert result["pages"] == 8
    assert (output / "data.json").read_bytes() == fixture["data_path"].read_bytes()

    html_path = output / "a1-benchmark.html"
    html = html_path.read_text(encoding="utf-8")
    document = _parse(html)
    assert len(document.ids) == len(set(document.ids))
    assert len([value for value in document.ids if value.startswith("page-")]) == 8
    assert {f"figure-{number}" for number in range(1, 8)} <= set(document.ids)
    assert html.count('src="data:image/svg+xml;base64,') == 7
    assert '<html lang="en">' in html
    assert not report.CJK_RE.search("".join(document.text))
    assert report.style_text(html_path) == report.style_text(fixture["a0_html"])
    assert "No blanket acceptance statement is made" in html
    assert "all soft checks passed" not in html.casefold()
    assert "finite 2T" in html

    ids = set(document.ids)
    for link in document.links:
        parsed = urlparse(link)
        assert not parsed.scheme or parsed.scheme == "data"
        assert not parsed.netloc
        if parsed.scheme == "data":
            continue
        if link.startswith("#"):
            assert unquote(parsed.fragment) in ids
        else:
            assert (html_path.parent / unquote(parsed.path)).exists(), link
    for path, digest in source_hashes.items():
        assert report.sha256_file(path) == digest


def test_manifest_preserves_sources_matrix_and_artifacts(fixture):
    _build(fixture)
    manifest = json.loads(
        (fixture["output"] / "manifest.json").read_text(encoding="utf-8")
    )
    figures = [
        figure
        for page in manifest["pages"]
        for figure in page.get("figures", [])
    ]
    assert len(figures) == 7
    assert {figure["number"] for figure in figures} == set(range(1, 8))
    assert {figure["source_kind"] for figure in figures} >= {
        "mixed_comparison",
        "nonlinear_evolution",
        "verification",
    }
    for figure in figures:
        for key in ("image", "svg", "data", "provenance"):
            assert (fixture["output"] / figure[key]).is_file()
    visible = json.dumps(manifest, ensure_ascii=False)
    assert "M=4..256" in visible
    assert "T0" in visible and "T1" in visible
    assert "Chinese report" in visible
    labels = {item["label"] for item in manifest["known_issues"]}
    assert any("T0 gallery" in label for label in labels)
    assert any("Chinese report" in label for label in labels)

    primary = report.primary_cases(fixture["data"])
    assert primary
    assert {case["coupling"] for case in primary} == {"waveform"}
    t0_data, _ = report.load_t0_data(Path(fixture["data"]["t0_gallery"]))
    for M in (4, 16, 32):
        t1_mixed = report.select_case(
            primary,
            M,
            "mixed",
            2048,
            coupling="waveform",
        )
        t1_residual = report.select_case(
            primary,
            M,
            "mixed",
            2048,
            coupling="waveform",
        )
        t0_mixed = report.select_case(
            t0_data["cases"],
            M,
            "mixed",
            2048,
            coupling="predictor",
        )
        assert t1_mixed is not None and t1_mixed["coupling"] == "waveform"
        assert report.case_gain_history(t1_mixed)[0]
        assert t1_residual is not None and t1_residual["seed"] == "mixed"
        assert t1_residual["iterations"]
        assert t1_residual["energy_time_over_T"]
        assert t0_mixed is not None and t0_mixed["coupling"] == "predictor"
        assert report.case_gain_history(t0_mixed)[0]

    fig2_text = (fixture["output"] / f"{report.FIGURE_NAMES[1]}.svg").read_text(
        encoding="utf-8"
    )
    fig3_text = (fixture["output"] / f"{report.FIGURE_NAMES[2]}.svg").read_text(
        encoding="utf-8"
    )
    fig4_text = (fixture["output"] / f"{report.FIGURE_NAMES[3]}.svg").read_text(
        encoding="utf-8"
    )
    fig5_text = (fixture["output"] / f"{report.FIGURE_NAMES[4]}.svg").read_text(
        encoding="utf-8"
    )
    assert "T1 waveform primary rtol" in fig2_text
    assert "T0" in fig3_text and "T1" in fig3_text
    assert "T1 mixed M=4" in fig4_text
    assert "T1 waveform accepted" in fig5_text


def test_tight_tolerance_controls_and_mixed_residuals_render(fixture):
    data = deepcopy(fixture["data"])
    report.validate_data(data)
    primary = report.primary_cases(data)
    mixed_residual = report.select_case(
        primary,
        4,
        "mixed",
        2048,
        coupling="waveform",
    )
    assert mixed_residual is not None and mixed_residual["seed"] == "mixed"

    tight = [
        case
        for case in data["cases"]
        if case["coupling"] == "waveform"
        and case["seed"] == "mixed"
        and case["micro_per_period"] == 2048
        and case["M"] in {16, 32}
        and case["rtol"] in {1.0e-10, 1.0e-14}
    ]
    assert {(case["M"], case["rtol"]) for case in tight} == {
        (16, 1.0e-10),
        (16, 1.0e-14),
        (32, 1.0e-10),
        (32, 1.0e-14),
    }
    assert all(case["high_sync_vector_error"] is not None for case in tight)

    without_optional = deepcopy(data)
    without_optional["cases"][0].pop("high_sync_vector_error", None)
    report.validate_data(without_optional)
    invalid = deepcopy(data)
    invalid["cases"][0]["high_sync_vector_error"] = -1.0
    with pytest.raises(report.InputError, match="high_sync_vector_error"):
        report.validate_data(invalid)

    _build(fixture)
    fig4_text = (fixture["output"] / f"{report.FIGURE_NAMES[3]}.svg").read_text(
        encoding="utf-8"
    )
    fig6_text = (fixture["output"] / f"{report.FIGURE_NAMES[5]}.svg").read_text(
        encoding="utf-8"
    )
    assert "T1 mixed M=4" in fig4_text
    assert "T1 mixed M=16" in fig4_text
    assert "T1 mixed M=32" in fig4_text
    assert "1e-14" in fig6_text and "1e-10" in fig6_text
    assert "M=16 vs fine RK4" in fig6_text
    assert "M=32 vs fine RK4" in fig6_text
    assert "M=16 final gain" in fig6_text
    assert "M=32 final gain" in fig6_text

    manifest = json.loads(
        (fixture["output"] / "manifest.json").read_text(encoding="utf-8")
    )
    figure6 = next(
        figure
        for page in manifest["pages"]
        for figure in page.get("figures", [])
        if figure["number"] == 6
    )
    assert len(figure6["panels"]) == 4
    assert "true synchronous fine RK4" in figure6["caption"]
    assert "gain is not vector accuracy" in figure6["caption"].casefold()


def test_completed_high_gain_budget_exceedance_is_explicit(fixture):
    data = deepcopy(fixture["data"])
    report.validate_data(data)
    exceedances = report.primary_high_gain_exceedances(data)
    replaced = {(case["M"], case["max_high_gain"]) for case in exceedances}
    assert (128, 1.0144201) in replaced
    assert all(M != 256 for M, _ in replaced)

    _build(fixture)
    output = fixture["output"]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    matrix_notes = "\n".join(
        note
        for page in manifest["pages"]
        if page["id"] == "t0-t1-matrix"
        for note in page["notes"]
    )
    for text in (
        manifest["introduction"],
        manifest["conclusions"],
        matrix_notes,
    ):
        assert "M=128" in text
        assert "1.0144201" in text
        assert "growth_limit=1.01" in text
        assert "completion is execution status" in text.casefold()
        assert "contiguous usable range" in text.casefold()

    fig2_text = (output / f"{report.FIGURE_NAMES[1]}.svg").read_text(
        encoding="utf-8"
    )
    assert "completed M=128" in fig2_text
    assert "1.0144201" in fig2_text
    assert "growth limit 1.01" in fig2_text
    assert "T1 completed, above growth limit" in fig2_text

    figure4 = next(
        figure
        for page in manifest["pages"]
        for figure in page.get("figures", [])
        if figure["number"] == 4
    )
    assert "same-time RK4 midpoint stages are distinct from each other" in (
        figure4["caption"]
    )
    assert "distinct from the converged discrete target" not in figure4["caption"]
    fig4_text = (output / f"{report.FIGURE_NAMES[3]}.svg").read_text(
        encoding="utf-8"
    )
    assert (
        "two same-time RK4 midpoint stages are distinct from each other"
        in fig4_text
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda data: data["cases"][0]["target_error"].pop(), "target_error"),
        (
            lambda data: data["cases"][0]["target_error"].__setitem__(0, float("inf")),
            "finite",
        ),
        (lambda data: data["cases"][1].__setitem__("id", data["cases"][0]["id"]), "duplicate"),
        (
            lambda data: data["cases"][0].__setitem__("max_target_error", -1.0),
            ">= 0",
        ),
        (
            lambda data: data["cases"][0].__setitem__("micro_per_period", 1024),
            "2048 or 4096",
        ),
        (
            lambda data: data["cases"][0]["iterations"][1].__setitem__(
                "iteration", data["cases"][0]["iterations"][0]["iteration"]
            ),
            "duplicate",
        ),
    ],
)
def test_validation_rejects_bad_contract(fixture, mutation, match):
    data = deepcopy(fixture["data"])
    mutation(data)
    with pytest.raises(report.InputError, match=match):
        report.validate_data(data)


def test_target_only_gain_is_null_and_rejection_is_strict(fixture):
    data = deepcopy(fixture["data"])
    report.validate_data(data)
    target = next(
        case
        for case in data["cases"]
        if case["id"] == "primary-4-target"
    )
    assert target["coupling"] == "waveform"
    assert target["high_norm_ratio"] is None
    assert report.case_gain_history(target) == ([], [])
    rejected = next(
        case
        for case in data["cases"]
        if case["id"] == "primary-256-target"
    )
    times = [0.0, rejected["rejected_time_over_T"], 1.0]
    assert report.accepted_indices(rejected, times) == [0]


def test_duplicate_json_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"protocol": {}, "protocol": {}}', encoding="utf-8")
    with pytest.raises(report.InputError, match="duplicate JSON object key"):
        report.load_json(path, "test input")


def test_existing_output_is_refused_without_change(fixture, capsys):
    output = fixture["output"]
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")
    code = report.main(
        [
            "--data",
            str(fixture["data_path"]),
            "--output",
            str(output),
            "--source-root",
            str(fixture["source_root"]),
            "--a0-html",
            str(fixture["a0_html"]),
        ]
    )
    assert code == 2
    assert marker.read_text(encoding="utf-8") == "unchanged"
    assert "refusing existing output" in capsys.readouterr().err
