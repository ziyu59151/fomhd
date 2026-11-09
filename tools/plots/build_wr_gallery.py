#!/usr/bin/env python3
"""Build a bounded offline current-WR benchmark gallery from actual raw JSON."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

sys.dont_write_bytecode = True

SCHEMA = "fomhd-wr-gallery-manifest-v1"
REQUIRED_SOURCES = (
    "cfl",
    "microstep_reference",
    "linear_selection",
    "backend_comparison",
    "parity_v3",
)
OPTIONAL_SOURCES = (
    "periodic_trial",
    "periodic_comparison",
    "matched_gpu_cost",
    "production_precheck",
    "boundary_precheck",
    "reviewed_sections",
    "energy_attribution",
)
CANONICAL_SOURCES = (
    "tools/plots/build_a0_report.py",
    "tools/plots/build_a1_t0_report.py",
    "tools/plots/build_a1_t2_report.py",
    "tools/plots/build_050_sections.py",
)


class InputError(ValueError):
    """The manifest or one immutable source violates the gallery contract."""


class UnsafeOutputError(ValueError):
    """The requested output could overwrite or contain immutable inputs."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class InputIndex:
    def __init__(self) -> None:
        self.records: dict[Path, dict[str, Any]] = {}

    def file(self, path: Path, role: str) -> Path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise InputError(f"required source is absent: {resolved}")
        record = self.records.setdefault(
            resolved,
            {
                "path": str(resolved),
                "sha256": sha256_file(resolved),
                "bytes": resolved.stat().st_size,
                "roles": [],
            },
        )
        if role not in record["roles"]:
            record["roles"].append(role)
            record["roles"].sort()
        return resolved

    def json(self, path: Path, role: str) -> Any:
        resolved = self.file(path, role)

        def reject(value: str) -> None:
            raise InputError(f"nonfinite JSON constant in {resolved}: {value}")

        try:
            return json.loads(
                resolved.read_text(encoding="utf-8"),
                parse_constant=reject,
            )
        except json.JSONDecodeError as error:
            raise InputError(f"invalid JSON {resolved}: {error}") from error

    def npz(self, path: Path, role: str) -> dict[str, np.ndarray]:
        resolved = self.file(path, role)
        try:
            with np.load(resolved, allow_pickle=False) as archive:
                return {key: np.asarray(archive[key]) for key in archive.files}
        except (OSError, ValueError) as error:
            raise InputError(f"invalid NPZ {resolved}: {error}") from error

    def verify_unchanged(self) -> None:
        for path, record in self.records.items():
            if sha256_file(path) != record["sha256"]:
                raise InputError(f"input changed during gallery build: {path}")

    def export(self) -> list[dict[str, Any]]:
        return [self.records[path].copy() for path in sorted(self.records)]


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{where} must be an object")
    return value


def _items(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise InputError(f"{where} must be an array")
    return value


def _text(value: dict[str, Any], key: str, where: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise InputError(f"{where}.{key} must be nonempty text")
    return item


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{where} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise InputError(f"{where} must be finite")
    return number


def _resolve(base: Path, value: str, where: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{where} must be nonempty path text")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def load_manifest(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Path], Path, Path]:
    path = path.resolve()
    if not path.is_file():
        raise InputError(f"manifest is absent: {path}")

    def reject(value: str) -> None:
        raise InputError(f"nonfinite manifest constant: {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except json.JSONDecodeError as error:
        raise InputError(f"invalid manifest JSON: {error}") from error
    manifest = _object(value, "manifest")
    if manifest.get("schema") != SCHEMA:
        raise InputError(f"manifest.schema must be {SCHEMA}")
    for key in ("title", "introduction", "conclusions"):
        _text(manifest, key, "manifest")
    base = path.parent
    raw_root = _resolve(base, _text(manifest, "raw_root", "manifest"), "manifest.raw_root")
    if not raw_root.is_dir():
        raise InputError(f"manifest.raw_root is not a directory: {raw_root}")
    sources = _object(manifest.get("sources"), "manifest.sources")
    paths = {}
    for role in REQUIRED_SOURCES:
        paths[role] = _resolve(
            base,
            _text(sources, role, "manifest.sources"),
            f"manifest.sources.{role}",
        )
    for role in OPTIONAL_SOURCES:
        if role in sources:
            paths[role] = _resolve(
                base,
                _text(sources, role, "manifest.sources"),
                f"manifest.sources.{role}",
            )
    availability = _items(manifest.get("availability"), "manifest.availability")
    if not availability:
        raise InputError("manifest.availability must not be empty")
    for index, raw in enumerate(availability):
        where = f"manifest.availability[{index}]"
        row = _object(raw, where)
        _text(row, "id", where)
        status = _text(row, "status", where)
        if status not in {"available", "partial", "blocked", "not_run"}:
            raise InputError(f"{where}.status is unsupported: {status}")
        _text(row, "detail", where)
    return manifest, paths, raw_root, base


def _safe_output(
    output: Path,
    inputs: list[Path],
    raw_root: Path,
) -> Path:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    try:
        output.relative_to(raw_root.resolve())
    except ValueError:
        pass
    else:
        raise UnsafeOutputError(f"output cannot be inside raw_root: {output}")
    for source in inputs:
        source = source.resolve()
        if output == source or output in source.parents or source in output.parents:
            raise UnsafeOutputError(
                f"output conflicts with immutable input ancestry: {output} / {source}"
            )
    if not output.parent.is_dir():
        raise InputError(f"output parent does not exist: {output.parent}")
    return output


def _json_object(index: InputIndex, path: Path, role: str) -> dict[str, Any]:
    return _object(index.json(path, role), role)


def _json_list(index: InputIndex, path: Path, role: str) -> list[Any]:
    return _items(index.json(path, role), role)


def _series(
    label: str,
    x: list[float],
    y: list[float],
    *,
    marker: str | None = "o",
    linestyle: str = "-",
) -> dict[str, Any]:
    if not x or len(x) != len(y):
        raise InputError(f"series {label} has no paired values")
    if not all(math.isfinite(value) for value in (*x, *y)):
        raise InputError(f"series {label} contains nonfinite values")
    result = {"label": label, "x": x, "y": y, "linestyle": linestyle}
    if marker is not None:
        result["marker"] = marker
    return result


def _panel(
    title: str,
    xlabel: str,
    ylabel: str,
    series: list[dict[str, Any]],
    *,
    xscale: str | None = None,
    yscale: str | None = None,
    horizontal_lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not series:
        raise InputError(f"panel {title} has no actual series")
    result = {
        "title": title,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "series": series,
    }
    if xscale:
        result["xscale"] = xscale
    if yscale:
        result["yscale"] = yscale
    if horizontal_lines:
        result["horizontal_lines"] = horizontal_lines
    return result


def _figure(
    figure_id: str,
    title: str,
    source_kind: str,
    panels: list[dict[str, Any]],
    *,
    question: str,
    fixed: str,
    scan: str,
    reference: str,
    pass_scope: str,
    caption: str,
    status: str,
    device: str,
) -> dict[str, Any]:
    if not 1 <= len(panels) <= 4:
        raise InputError(f"{figure_id} must have one to four panels")
    return {
        "id": figure_id,
        "title": title,
        "source_kind": source_kind,
        "caption": caption,
        "status": status,
        "device": device,
        "question": question,
        "controls": (
            f"Fixed: {fixed} Scan: {scan} Reference: {reference} "
            f"Pass scope: {pass_scope}"
        ),
        "criterion": pass_scope,
        "panels": panels,
    }


def _periodic_method_label(ident: str, case: dict[str, Any]) -> str:
    scheme = _object(case.get("scheme"), f"periodic case {ident} scheme")
    micro_steps = int(scheme.get("M"))
    coupling = str(scheme.get("coupling"))
    device = str(scheme.get("device")).upper()
    if coupling == "synchronous":
        label = f"{device} A0 synchronous M{micro_steps}"
    elif coupling == "waveform":
        label = f"{device} T1 waveform M{micro_steps}"
    elif coupling == "multirate_wr":
        label = f"{device} WR s{int(scheme.get('s'))} M{micro_steps}"
    else:
        raise InputError(f"periodic case {ident} has unsupported coupling {coupling}")
    marker = f"{ident} {case.get('role', '')}".casefold()
    if "fixed" in marker or "half_h" in marker:
        label += " fixed-H h/2"
    return label


def _periodic_metric_series(
    group: dict[str, Any],
    candidates: dict[str, Any],
    cases: dict[str, Any],
    metric: str,
) -> list[dict[str, Any]]:
    result = []
    for ident in sorted(candidates):
        case = _object(cases.get(ident), f"periodic case {ident}")
        values = _object(candidates[ident], f"periodic metrics {ident}")
        result.append(
            _series(
                _periodic_method_label(ident, case),
                [float(int(_object(case["scheme"], "scheme").get("M")))],
                [_finite(values.get(metric), f"{ident}.{metric}")],
            )
        )
    return result


def _periodic_comparison_figures(
    index: InputIndex,
    path: Path,
) -> list[dict[str, Any]]:
    source = _json_object(index, path, "periodic_comparison")
    if source.get("schema") != "fomhd-periodic-summary-v2":
        raise InputError("periodic comparison schema mismatch")
    scope = _object(source.get("scope"), "periodic_comparison.scope")
    nominal = _text(scope, "nominal_horizon", "periodic_comparison.scope")
    contracts = _object(
        source.get("contracts"),
        "periodic_comparison.contracts",
    )
    for key in (
        "all_comparison_sources_admitted",
        "all_complete_physical_contracts_exact",
        "cpu_source_arrays_exact",
    ):
        if contracts.get(key) is not True:
            raise InputError(f"periodic comparison contract is false: {key}")
    if _object(
        contracts.get("physical_parameter_mismatches"),
        "periodic_comparison.contracts.physical_parameter_mismatches",
    ):
        raise InputError("periodic comparison has physical parameter mismatches")
    metric_contract = _object(
        source.get("metric_contract"),
        "periodic_comparison.metric_contract",
    )
    if metric_contract.get("no_mixed_unit_absolute_norm") is not True:
        raise InputError("periodic comparison mixes absolute component units")
    reference = _text(
        _object(source.get("manifest"), "periodic_comparison.manifest"),
        "reference_case",
        "periodic_comparison.manifest",
    )
    cases = _object(source.get("cases"), "periodic_comparison.cases")
    comparisons = _object(
        source.get("comparisons"),
        "periodic_comparison.comparisons",
    )
    groups = _object(
        source.get("target_mode_metrics"),
        "periodic_comparison.target_mode_metrics",
    )
    target = _object(groups.get("target"), "periodic_comparison.target")
    sentinel = _object(groups.get("sentinel"), "periodic_comparison.sentinel")
    if int(target.get("mode")) != 1 or int(sentinel.get("mode")) != 16:
        raise InputError("periodic target/sentinel modes are not mode1/mode16")
    target_candidates = _object(
        target.get("candidates"),
        "periodic_comparison.target.candidates",
    )
    sentinel_candidates = _object(
        sentinel.get("candidates"),
        "periodic_comparison.sentinel.candidates",
    )
    if set(target_candidates) != set(sentinel_candidates):
        raise InputError("periodic target and sentinel candidate sets differ")
    join_times = []
    hard_failures = []
    for ident in sorted(target_candidates):
        case = _object(cases.get(ident), f"periodic case {ident}")
        if case.get("status") != "complete":
            raise InputError(f"periodic case {ident} is not complete")
        comparison = _object(
            comparisons.get(ident),
            f"periodic comparison {ident}",
        )
        if comparison.get("available") is not True:
            raise InputError(f"periodic comparison {ident} is unavailable")
        join = _object(comparison.get("join"), f"periodic join {ident}")
        if join.get("joined") is not True:
            raise InputError(f"periodic comparison {ident} is not joined")
        if _finite(join.get("progress_fraction"), "progress_fraction") != 1.0:
            raise InputError(f"periodic case {ident} is not a full horizon")
        join_times.append(_finite(join.get("time"), f"periodic join {ident} time"))
        if _items(
            comparison.get("physical_parameter_mismatches"),
            f"periodic mismatches {ident}",
        ):
            raise InputError(f"periodic comparison {ident} physical mismatch")
        checks = _object(
            _object(case.get("diagnostics"), f"periodic case {ident} diagnostics").get(
                "checks"
            ),
            f"periodic case {ident} checks",
        )
        failed = int(checks.get("hard_failed"))
        if failed != 0:
            raise InputError(f"periodic case {ident} has {failed} hard failures")
        hard_failures.append(f"{ident}:0")
    if not join_times or any(
        not math.isclose(join_times[0], value, rel_tol=1e-12, abs_tol=1e-30)
        for value in join_times
    ):
        raise InputError("periodic candidates do not share a physical final time")
    reference_case = _object(cases.get(reference), f"periodic reference {reference}")
    if reference_case.get("status") != "complete":
        raise InputError("periodic reference case is not complete")
    devices = sorted(
        {
            str(_object(cases[ident]["scheme"], f"{ident}.scheme").get("device"))
            for ident in target_candidates
        }
    )

    def device_label(device: str) -> str:
        if device == "cpu":
            return "CPU"
        if device == "gpu":
            return "GPU"
        if device.startswith("cuda"):
            return device.replace("cuda", "CUDA", 1)
        return device.upper()

    comparison_device = " and ".join(device_label(device) for device in devices)
    fixed = (
        f"reference={reference}; nominal 2T horizon={nominal}; "
        "all candidates share one common physical final time"
    )
    reference_scope = (
        f"matched synchronous reference {reference}; "
        "2T is a nominal nonlinear-mixture horizon, not a pure-mode growth interval"
    )
    target_figure = _figure(
        "periodic-target-mode1",
        "Periodic target mode1 qualification",
        "verification",
        [
            _panel(
                "Target mode1 normalized complex error",
                "M microsteps per advance",
                "normalized complex difference",
                _periodic_metric_series(
                    target,
                    target_candidates,
                    cases,
                    "normalized_difference_norm",
                ),
                yscale="symlog",
            ),
            _panel(
                "Target mode1 signed phase error",
                "M microsteps per advance",
                "rad",
                _periodic_metric_series(
                    target,
                    target_candidates,
                    cases,
                    "phase_rad",
                ),
                yscale="symlog",
            ),
            _panel(
                "Target mode1 amplitude ratio",
                "M microsteps per advance",
                "candidate / reference",
                _periodic_metric_series(
                    target,
                    target_candidates,
                    cases,
                    "amplitude_ratio",
                ),
            ),
        ],
        question="How close is target mode1 to the matched synchronous 2T reference?",
        fixed=fixed,
        scan=(
            "Named A0, T1, WR s3/s5 and fixed-H cases; M remains the distinct "
            "microstep count. Hard failures: " + ", ".join(hard_failures) + "."
        ),
        reference=reference_scope,
        pass_scope=(
            "Target-mode metric qualification only; the declared budget is a "
            "metric contract, not universal scientific acceptance."
        ),
        caption="Actual target mode1 complex error, phase and amplitude rows.",
        status="completed comparison",
        device=comparison_device,
    )
    sentinel_figure = _figure(
        "periodic-sentinel-mode16",
        "Periodic grid-high mode16 sentinel",
        "verification",
        [
            _panel(
                "Mode16 sentinel normalized complex error",
                "M microsteps per advance",
                "normalized complex difference",
                _periodic_metric_series(
                    sentinel,
                    sentinel_candidates,
                    cases,
                    "normalized_difference_norm",
                ),
                yscale="symlog",
            ),
            _panel(
                "Mode16 sentinel signed phase error",
                "M microsteps per advance",
                "rad",
                _periodic_metric_series(
                    sentinel,
                    sentinel_candidates,
                    cases,
                    "phase_rad",
                ),
                yscale="symlog",
            ),
            _panel(
                "Mode16 sentinel amplitude ratio",
                "M microsteps per advance",
                "candidate / reference",
                _periodic_metric_series(
                    sentinel,
                    sentinel_candidates,
                    cases,
                    "amplitude_ratio",
                ),
            ),
        ],
        question="What does the separate grid-high mode16 sentinel show?",
        fixed=fixed,
        scan="Named methods at their actual M values.",
        reference=reference_scope,
        pass_scope=(
            "Grid-high sentinel evidence only; mode16 damping is not macro "
            "instability and is not target-mode acceptance."
        ),
        caption="Actual mode16 sentinel rows, kept separate from target mode1.",
        status="completed comparison",
        device=comparison_device,
    )
    fixed_ids = [
        ident
        for ident in target_candidates
        if "fixed" in f"{ident} {_object(cases[ident], ident).get('role', '')}".casefold()
        or "half_h" in f"{ident} {_object(cases[ident], ident).get('role', '')}".casefold()
    ]
    base_ids = [
        ident
        for ident in target_candidates
        if ident not in fixed_ids
        and int(_object(_object(cases[ident], ident).get("scheme"), "scheme").get("M"))
        == 8
        and int(_object(_object(cases[ident], ident).get("scheme"), "scheme").get("s"))
        == 3
        and _object(_object(cases[ident], ident).get("scheme"), "scheme").get(
            "coupling"
        )
        == "multirate_wr"
        and _object(_object(cases[ident], ident).get("scheme"), "scheme").get(
            "device"
        )
        == "cpu"
    ]
    if len(fixed_ids) != 1 or len(base_ids) != 1:
        raise InputError("periodic fixed-H comparison requires one base and one fixed case")
    base_id = base_ids[0]
    fixed_id = fixed_ids[0]
    base_scheme = _object(cases[base_id]["scheme"], f"{base_id}.scheme")
    fixed_scheme = _object(cases[fixed_id]["scheme"], f"{fixed_id}.scheme")
    base_m = int(base_scheme["M"])
    fixed_m = int(fixed_scheme["M"])
    if fixed_m != 2 * base_m or not math.isclose(
        2.0 * float(fixed_scheme["h"]),
        float(base_scheme["h"]),
        rel_tol=1e-12,
        abs_tol=0.0,
    ):
        raise InputError("periodic fixed-H case does not retain M and h/2 scaling")
    pair_x = [float(base_m), float(fixed_m)]

    def pair_series(group, metric, label):
        candidates = _object(group.get("candidates"), "periodic pair candidates")
        return _series(
            label,
            pair_x,
            [
                _finite(
                    _object(candidates[base_id], base_id).get(metric),
                    f"{base_id}.{metric}",
                ),
                _finite(
                    _object(candidates[fixed_id], fixed_id).get(metric),
                    f"{fixed_id}.{metric}",
                ),
            ],
        )

    fixed_figure = _figure(
        "periodic-fixed-h-h2",
        "Fixed-H microstep h versus h/2 comparison",
        "verification",
        [
            _panel(
                "Target complex error at M8 h and M16 h/2",
                "M microsteps per advance",
                "normalized complex difference",
                [pair_series(target, "normalized_difference_norm", "target mode1")],
                yscale="symlog",
            ),
            _panel(
                "Target phase at M8 h and M16 h/2",
                "M microsteps per advance",
                "rad",
                [pair_series(target, "phase_rad", "target mode1")],
                yscale="symlog",
            ),
            _panel(
                "Mode16 sentinel complex error at M8 h and M16 h/2",
                "M microsteps per advance",
                "normalized complex difference",
                [
                    pair_series(
                        sentinel,
                        "normalized_difference_norm",
                        "mode16 grid-high sentinel",
                    )
                ],
                yscale="symlog",
            ),
            _panel(
                "Mode16 sentinel phase at M8 h and M16 h/2",
                "M microsteps per advance",
                "rad",
                [
                    pair_series(
                        sentinel,
                        "phase_rad",
                        "mode16 grid-high sentinel",
                    )
                ],
                yscale="symlog",
            ),
        ],
        question="How do target and high-sentinel errors change at fixed H when h is halved?",
        fixed=(
            f"base={base_id} M=8 h; fixed={fixed_id} M=16 h/2; "
            "both retain the same H, physical final time and completed fraction"
        ),
        scan="Two points per panel at distinct actual microstep counts M8 and M16.",
        reference=reference_scope,
        pass_scope=(
            "Fixed-H h/2 numerical control only; mode16 is a separate grid-high "
            "sentinel and its damping is not macro instability."
        ),
        caption=(
            "Actual fixed-H pair. Microstep counts M8 and M16 remain distinct; "
            "no pure-mode growth rate or macro stability claim is made."
        ),
        status="completed comparison",
        device="CPU",
    )
    return [target_figure, sentinel_figure, fixed_figure]


def _cost_method_label(identity: dict[str, Any]) -> str:
    method = str(identity.get("method"))
    micro_steps = int(identity.get("micro_steps"))
    if method == "synchronous":
        return f"A0 synchronous M{micro_steps}"
    if method == "waveform":
        return f"T1 waveform M{micro_steps}"
    if method == "multirate_wr":
        return f"WR s{int(identity.get('ion_nodes'))} M{micro_steps}"
    raise InputError(f"matched cost has unsupported method {method}")


def _matched_gpu_cost_figure(
    index: InputIndex,
    path: Path,
) -> dict[str, Any]:
    source = _json_object(index, path, "matched_gpu_cost")
    if source.get("schema") != "fomhd-matched-wr-cost-v1":
        raise InputError("matched GPU cost schema mismatch")
    if source.get("status") != "completed":
        raise InputError("matched GPU cost is not completed")
    if source.get("physical_input_consistent") is not True:
        raise InputError("matched GPU cost inputs are inconsistent")
    if source.get("all_initial_inputs_nontrivial") is not True:
        raise InputError("matched GPU cost has a trivial initial input")
    config = _object(source.get("config"), "matched_gpu_cost.config")
    limits = _object(source.get("limits"), "matched_gpu_cost.limits")
    warmup_count = int(config.get("warmup"))
    repeats = int(config.get("repeats"))
    if warmup_count != 1 or repeats != 3:
        raise InputError("matched GPU cost is not one warmup plus three repeats")
    if int(limits.get("warmup")) != warmup_count or int(limits.get("repeats")) != repeats:
        raise InputError("matched GPU cost config and limits disagree")
    methods = _items(source.get("methods"), "matched_gpu_cost.methods")
    if len(methods) != 4:
        raise InputError("matched GPU cost must contain four named methods")
    series = []
    labels = []
    medians = []
    reset_seconds = []
    setup_seconds = []
    for position, raw in enumerate(methods):
        method = _object(raw, f"matched_gpu_cost.methods[{position}]")
        if method.get("status") != "accepted":
            raise InputError(f"matched GPU method {position} is not accepted")
        identity = _object(method.get("identity"), f"matched method {position}.identity")
        label = _cost_method_label(identity)
        labels.append(label)
        run_seconds = _object(
            method.get("run_seconds"),
            f"matched method {position}.run_seconds",
        )
        samples = [
            _finite(value, f"{label}.run_seconds.samples[]")
            for value in _items(
                run_seconds.get("samples"),
                f"{label}.run_seconds.samples",
            )
        ]
        if len(samples) != repeats:
            raise InputError(f"{label} run-second sample count mismatch")
        median = _finite(run_seconds.get("median"), f"{label}.run_seconds.median")
        if not math.isclose(median, statistics.median(samples), rel_tol=1e-12, abs_tol=1e-15):
            raise InputError(f"{label} median run seconds do not match samples")
        warmup = _object(method.get("warmup"), f"matched method {position}.warmup")
        reset_seconds.append(_finite(warmup.get("reset_seconds"), "reset_seconds"))
        setup_seconds.append(_finite(warmup.get("setup_seconds"), "setup_seconds"))
        _finite(method.get("context_setup_seconds"), "context_setup_seconds")
        if int(method.get("ledger_resets_inside_physical_run")) != 0:
            raise InputError(f"{label} resets a ledger inside its physical run")
        series.append(
            _series(
                label,
                [float(int(identity["micro_steps"]))],
                [median],
            )
        )
        medians.append(f"{label}={median:.8f} s")
    if len(set(labels)) != 4:
        raise InputError("matched GPU cost method labels are not unique")
    return _figure(
        "matched-gpu-cost-8h",
        "Matched same-GPU complete-horizon cost",
        "verification",
        [
            _panel(
                "Median complete-advance seconds",
                "Declared microsteps per advance",
                "median complete-advance seconds",
                series,
                yscale="log",
            )
        ],
        question="What are the actual matched same-GPU complete-horizon advance medians?",
        fixed=(
            f"points={config.get('points')} common_horizon_microsteps="
            f"{config.get('horizon_microsteps')} warmup={warmup_count} repeats={repeats}"
        ),
        scan="Named methods: " + ", ".join(labels) + ".",
        reference=(
            f"{_text(limits, 'warmup_semantics', 'matched_gpu_cost.limits')}; "
            "every measured repeat starts from the same cloned accepted input"
        ),
        pass_scope=(
            "Observed single-GPU complete-horizon medians; warmup, reset, setup "
            "and context setup are not folded into run_seconds. No speedup or "
            "scientific acceptance claim."
        ),
        caption=(
            "Plotted y is median complete-advance run_seconds over three repeats. "
            "One synchronized full-horizon warmup precedes each method; "
            "reset_seconds and setup_seconds are recorded separately and excluded, "
            "as is context_setup_seconds; ledger resets inside each physical run "
            "are zero. Counter-role erratum: raw full_ion_calls denotes the ion "
            "transport producer, full_stage_calls denotes the stage evaluator, "
            "and full_rhs_calls denotes the full RHS finish. The observed "
            "144 -> 24 values are not a complete-RHS reduction. Medians: "
            + "; ".join(medians)
        ),
        status="completed matched-cost observation",
        device=str(config.get("device")),
    )


def _precheck_output(
    row: dict[str, Any],
    campaign_root: Path,
    where: str,
) -> Path:
    output = row.get("output")
    if output is None:
        command = _items(row.get("command"), f"{where}.command")
        try:
            output = command[command.index("--output") + 1]
        except (ValueError, IndexError) as error:
            raise InputError(f"{where} lacks an exact output path") from error
    if not isinstance(output, str) or not output.strip():
        raise InputError(f"{where}.output must be nonempty path text")
    path = Path(output)
    resolved = (path if path.is_absolute() else campaign_root / path).resolve()
    try:
        resolved.relative_to(campaign_root.resolve())
    except ValueError as error:
        raise InputError(f"{where} output escapes its campaign root") from error
    if not resolved.is_dir():
        raise InputError(f"{where} output directory is absent: {resolved}")
    return resolved


def _precheck_record(
    index: InputIndex,
    row: dict[str, Any],
    campaign_root: Path,
    design: str,
    where: str,
) -> dict[str, Any]:
    output = _precheck_output(row, campaign_root, where)
    run_path = output / "run.json"
    run = _json_object(index, run_path, f"{where} run.json")
    history_path = output / "history.jsonl"
    failure_path = output / "failure.json"
    if history_path.is_file():
        index.file(history_path, f"{where} history.jsonl")
    if failure_path.is_file():
        index.file(failure_path, f"{where} failure.json")
    scheme = _object(
        _object(run.get("identity"), f"{where}.run.identity").get("scheme"),
        f"{where}.run.identity.scheme",
    )
    row_status = str(row.get("status"))
    run_status = str(run.get("status"))
    if row_status != run_status or run_status not in {"partial", "failed"}:
        raise InputError(f"{where} precheck/run status mismatch")
    if run_status == "partial":
        if not history_path.is_file() or history_path.stat().st_size == 0:
            raise InputError(f"{where} accepted prefix lacks first-macro history")
        lines = history_path.read_text(encoding="utf-8").splitlines()
        try:
            history = _object(json.loads(lines[0]), f"{where} history[0]")
        except json.JSONDecodeError as error:
            raise InputError(f"{where} invalid first history row") from error
        if int(history.get("completed_macros")) != 1:
            raise InputError(f"{where} history is not the first accepted macro")
        work = _object(history.get("work"), f"{where}.history.work")
        cost = _finite(
            _object(history.get("timing"), f"{where}.history.timing").get(
                "advance_seconds"
            ),
            f"{where}.history.timing.advance_seconds",
        )
        checks = []
        classification = "acceptedprefix"
    else:
        if not failure_path.is_file():
            raise InputError(f"{where} failed run lacks failure.json")
        failure = _json_object(index, failure_path, f"{where} failure.json")
        if failure.get("code") != "HardGate":
            raise InputError(f"{where} failure is not a hard gate")
        work = _object(failure.get("work"), f"{where}.failure.work")
        cost = _finite(
            failure.get("failure_seconds"),
            f"{where}.failure.failure_seconds",
        )
        checks = [
            _object(item, f"{where}.failure.checks.failures[]")
            for item in _items(
                _object(
                    failure.get("checks"),
                    f"{where}.failure.checks",
                ).get("failures"),
                f"{where}.failure.checks.failures",
            )
            if _object(item, f"{where}.failure.checks.failures[]").get("hard")
            is True
        ]
        coupling = _object(
            failure.get("coupling_metadata"),
            f"{where}.failure.coupling_metadata",
        )
        residual = _items(
            coupling.get("residual_history", []),
            f"{where}.failure.residual_history",
        )
        converged = bool(
            residual
            and _object(residual[-1], f"{where}.residual[-1]").get("converged")
            is True
        )
        classification = (
            "converged-hard-rejected"
            if converged
            else "hard-rejected-before-convergence"
        )
    return {
        "name": _text(row, "name", where),
        "design": design,
        "coupling": str(scheme.get("coupling")),
        "device": str(scheme.get("device")),
        "M": int(scheme.get("M")),
        "s": int(scheme.get("s")),
        "h": float(scheme.get("h")),
        "work": work,
        "cost": cost,
        "checks": checks,
        "classification": classification,
        "output": str(output),
    }


def _first_window_boundary_figures(
    index: InputIndex,
    production_path: Path,
    boundary_path: Path,
) -> list[dict[str, Any]]:
    records = []
    for source_path, design in (
        (production_path, "base h"),
        (boundary_path, "fixed H h/2"),
    ):
        campaign_root = source_path.parent.parent
        rows = _items(
            index.json(source_path, source_path.stem),
            str(source_path),
        )
        for position, raw in enumerate(rows):
            where = f"{source_path.name}[{position}]"
            records.append(
                _precheck_record(
                    index,
                    _object(raw, where),
                    campaign_root,
                    design,
                    where,
                )
            )
    base_h_values = [
        record["h"]
        for record in records
        if record["coupling"] == "multirate_wr" and record["design"] == "base h"
    ]
    if not base_h_values:
        raise InputError("boundary prechecks lack a base-h WR record")
    h_base = base_h_values[0]
    if any(
        not math.isclose(value, h_base, rel_tol=1e-12, abs_tol=0.0)
        for value in base_h_values
    ):
        raise InputError("boundary prechecks contain multiple base h values")
    wr_records = []
    for record in records:
        if record["coupling"] != "multirate_wr":
            continue
        ratio = record["h"] / h_base
        expected_ratio = 1.0 if record["design"] == "base h" else 0.5
        if not math.isclose(ratio, expected_ratio, rel_tol=1e-12, abs_tol=0.0):
            raise InputError(f"{record['name']} h does not match its design")
        effective = record["M"] * ratio
        rounded = round(effective)
        if not math.isclose(effective, rounded, rel_tol=0.0, abs_tol=1e-12):
            raise InputError(f"{record['name']} has noninteger effective base-h M")
        record["effective_M"] = float(rounded)
        wr_records.append(record)
    groups = {
        (design, s): sorted(
            (
                record
                for record in wr_records
                if record["design"] == design and record["s"] == s
            ),
            key=lambda record: record["M"],
        )
        for design in ("base h", "fixed H h/2")
        for s in (3, 5)
    }
    groups = {key: value for key, value in groups.items() if value}

    def counter_series(counter: str) -> list[dict[str, Any]]:
        return [
            _series(
                f"{design} WR s{s}",
                [float(record["M"]) for record in rows],
                [
                    _finite(
                        _object(record["work"], "precheck work").get(counter),
                        f"{record['name']}.{counter}",
                    )
                    for record in rows
                ],
            )
            for (design, s), rows in groups.items()
        ]

    status_text = "; ".join(
        f"{record['design']} WR s{record['s']} M={record['M']} "
        f"{record['classification']}"
        for record in sorted(
            wr_records,
            key=lambda record: (record["design"], record["s"], record["M"]),
        )
    )
    work_figure = _figure(
        "first-window-boundary-work",
        "First-window macro boundary work and cost",
        "verification",
        [
            _panel(
                "Coarse node calls",
                "Declared M microsteps",
                "calls",
                counter_series("coarse_node_calls"),
                xscale="log",
                yscale="log",
            ),
            _panel(
                "Stage evaluator calls",
                "Declared M microsteps",
                "calls",
                counter_series("full_stage_calls"),
                xscale="log",
                yscale="log",
            ),
            _panel(
                "Field evaluations",
                "Declared M microsteps",
                "evaluations",
                counter_series("field_evaluations"),
                xscale="log",
                yscale="log",
            ),
            _panel(
                "First-window wall time",
                "Declared M microsteps",
                "seconds",
                [
                    _series(
                        f"{design} WR s{s}",
                        [float(record["M"]) for record in rows],
                        [record["cost"] for record in rows],
                    )
                    for (design, s), rows in groups.items()
                ],
                xscale="log",
                yscale="log",
            ),
        ],
        question="How do first-window WR work and cost change across the declared macro ladder?",
        fixed="CPU-only one-macro prechecks; base h and same-H fixed h/2 remain separate",
        scan=status_text,
        reference=(
            "Raw precheck work spies and first-window timings; A0/T1 controls "
            "are excluded from the WR ladder"
        ),
        pass_scope=(
            "First-window only. Acceptedprefix rows are unscanned beyond their "
            "segment budget. Converged-hard-rejected rows fail a hard check; "
            "this is not a stability bound or 2T result."
        ),
        caption=(
            f"First-window only; counters are raw and never summed. Stage "
            f"evaluator means raw full_stage_calls after the counter erratum. "
            f"Status: {status_text}. Base-h M64/M128 are absent and remain "
            "not_run with no zero fillers; accepted-prefix rows stop at the "
            "segment budget with the remaining horizon unscanned."
        ),
        status="first-window precheck",
        device="CPU",
    )
    energy_groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for record in wr_records:
        for check in record["checks"]:
            name = check.get("name")
            if name == "kinetic_energy":
                metric = "ion kinetic"
            elif name == "field_energy_B_interval":
                metric = "field energy"
            else:
                continue
            threshold = _finite(
                check.get("threshold"),
                f"{record['name']}.{name}.threshold",
            )
            if threshold <= 0.0:
                raise InputError(f"{record['name']} energy threshold is nonpositive")
            ratio = abs(
                _finite(check.get("raw"), f"{record['name']}.{name}.raw")
            ) / threshold
            record = dict(record)
            record["ratio"] = ratio
            energy_groups.setdefault(
                (record["design"], record["s"], metric),
                [],
            ).append(record)
    energy_series = [
        _series(
            f"{design} WR s{s} {metric}",
            [record["effective_M"] for record in rows],
            [record["ratio"] for record in rows],
        )
        for (design, s, metric), rows in sorted(energy_groups.items())
    ]
    if not energy_series:
        raise InputError("boundary prechecks contain no explicit energy rejection checks")
    accepted = [
        record
        for record in wr_records
        if record["classification"] == "acceptedprefix"
    ]
    energy_figure = _figure(
        "first-window-energy-rejection",
        "First-window hard energy rejection ratios",
        "verification",
        [
            _panel(
                "Ion kinetic and field-energy hard rejection",
                "Effective base-h M = M*h/h_base",
                "|raw| / threshold",
                energy_series,
                xscale="log",
                yscale="log",
                horizontal_lines=[{"y": 1.0, "label": "hard threshold"}],
            )
        ],
        question="Which first-window kinetic/field energy checks cross the hard threshold?",
        fixed="Only explicit hard-check raw values; h/h2 mapped to effective base-h M",
        scan=(
            "base h and fixed H h/2, separately for s3 and s5; no zero fill for "
            "accepted-prefix rows"
        ),
        reference="Declared check thresholds, normalized here to ratio 1",
        pass_scope=(
            "First-window hard rejection only. No inferred stability bound and "
            "no claim beyond the unscanned budget stop."
        ),
        caption=(
            "Explicit ion kinetic and field-energy rejection ratios only. "
            f"Acceptedprefix cases ({', '.join(record['name'] for record in accepted)}) "
            "have no rejection sample in the source and are not zero-filled. "
            "Base-h M64/M128 remain not_run. The threshold-1 line is the "
            "declared hard check; no stability bound is inferred."
        ),
        status="first-window precheck",
        device="CPU",
    )
    return [work_figure, energy_figure]


def _build_data(
    manifest: dict[str, Any],
    paths: dict[str, Path],
    raw_root: Path,
    index: InputIndex,
) -> dict[str, Any]:
    for role in OPTIONAL_SOURCES:
        if role in paths:
            index.file(paths[role], role)
    cfl = _json_object(index, paths["cfl"], "cfl")
    cfl_rows = [_object(row, "cfl.rows[]") for row in _items(cfl.get("rows"), "cfl.rows")]
    if not cfl_rows:
        raise InputError("cfl.rows must not be empty")
    micro = _json_object(index, paths["microstep_reference"], "microstep_reference")
    micro_rows = [
        _object(row, "microstep_reference.rows[]")
        for row in _items(micro.get("rows"), "microstep_reference.rows")
    ]
    linear = _json_object(index, paths["linear_selection"], "linear_selection")
    linear_rows = [
        _object(row, "linear_selection.rows[]")
        for row in _items(linear.get("rows"), "linear_selection.rows")
    ]
    backend_rows = [
        _object(row, "backend_comparison[]")
        for row in _json_list(index, paths["backend_comparison"], "backend_comparison")
    ]
    parity = _json_object(index, paths["parity_v3"], "parity_v3")
    wr_cases = [
        _object(row, "parity_v3.wr.cases[]")
        for row in _items(_object(parity.get("wr"), "parity_v3.wr").get("cases"), "parity_v3.wr.cases")
    ]
    massless = _object(parity.get("massless"), "parity_v3.massless")

    figures = []
    figures.append(
        _figure(
            "local-cfl",
            "Local Jacobian CFL and growth audit",
            "production_linearization",
            [
                _panel(
                    "Maximum local real rate",
                    "Jacobian case index",
                    "max Re(lambda) (1/s)",
                    [
                        _series(
                            f"{row.get('case', index)}/{row.get('subsystem', 'unknown')}",
                            [float(index)],
                            [_finite(row.get("max_real_s_inv"), "max_real_s_inv")],
                        )
                        for index, row in enumerate(cfl_rows)
                    ],
                    yscale="symlog",
                ),
                _panel(
                    "Dimensionless local rates",
                    "Jacobian case index",
                    "dimensionless",
                    [
                        _series(
                            f"{row.get('case', index)}/{row.get('subsystem', 'unknown')}",
                            [float(index)],
                            [_finite(row.get("omega_h"), "omega_h")],
                        )
                        for index, row in enumerate(cfl_rows)
                    ],
                    horizontal_lines=[
                        {
                            "y": 2.8284271247461903,
                            "label": "RK4 imaginary-axis bound",
                        }
                    ],
                ),
            ],
            question="Do local initial Jacobians show an actual RK4 CFL violation?",
            fixed="Existing spatial/physical evaluator and finite-difference epsilon.",
            scan="Uniform and finite nonuniform initial controls; full and fixed-ion field subsystems.",
            reference="Exact local Jacobian and RK4 stability bound, not a nonlinear run.",
            pass_scope="Local linearization only; nonnormal growth is not called numerical instability.",
            caption="Local eigenvalues and dimensionless RK4 rates from the listed raw audit.",
            status="bounded local evidence",
            device="CPU",
        )
    )
    if "periodic_trial" in paths:
        trial = _json_object(index, paths["periodic_trial"], "periodic_trial")
        if trial.get("schema") != "fomhd-wr-benchmark-v1":
            raise InputError("periodic trial schema mismatch")
        if trial.get("status") != "completed" or trial.get("production_accepted") is not False:
            raise InputError("periodic trial must be completed with production_accepted=false")
        history = [
            _object(row, "periodic_trial.mode_history[]")
            for row in _items(trial.get("mode_history"), "mode_history")
        ]
        if len(history) < 2:
            raise InputError("periodic trial mode history is too short")

        def magnitude(entry: dict[str, Any], population: str, component: str) -> float:
            amplitudes = _object(entry.get("amplitudes"), "mode amplitudes")
            group = _object(amplitudes.get(population), "mode population")
            value = _object(
                _object(group.get("components"), "mode components").get(component),
                "mode component",
            )
            return math.hypot(
                _finite(value.get("real"), "mode real"),
                _finite(value.get("imag"), "mode imag"),
            )

        x = [_finite(row.get("macro"), "macro") for row in history]
        target_series = []
        sentinel_series = []
        for component in ("ion_mom_phi", "electron_perp", "B_phi"):
            target_initial = magnitude(history[0], "target", component)
            sentinel_initial = magnitude(history[0], "sentinel", component)
            target_series.append(
                _series(
                    f"target {component}",
                    x,
                    [magnitude(row, "target", component) / target_initial for row in history],
                )
            )
            sentinel_series.append(
                _series(
                    f"mode16 {component}",
                    x,
                    [
                        magnitude(row, "sentinel", component) / sentinel_initial
                        for row in history
                    ],
                )
            )
        comparisons = _object(trial.get("comparisons"), "periodic_trial.comparisons")
        comparison_names = sorted(comparisons)
        comparison_x = [float(index) for index in range(len(comparison_names))]
        comparison_mapping = ", ".join(
            f"{index} -> {name}"
            for index, name in enumerate(comparison_names)
        )
        config = _object(trial.get("config"), "periodic_trial.config")
        resolved = _object(trial.get("resolved_config"), "periodic_trial.resolved_config")
        figures.append(
            _figure(
                "periodic-2t-trial",
                "Periodic 33-point 2T mathematical trial",
                "verification",
                [
                    _panel(
                        "Target mode1 component magnitudes",
                        "Accepted macro",
                        "magnitude / initial",
                        target_series,
                    ),
                    _panel(
                        "High-mode16 sentinel magnitudes",
                        "Accepted macro",
                        "magnitude / initial",
                        sentinel_series,
                        yscale="log",
                    ),
                    _panel(
                        "Normalized same-window comparisons",
                        "Comparison index (caption maps each index to reference name)",
                        "normalized error",
                        [
                            _series(
                                "P",
                                comparison_x,
                                [
                                    _finite(
                                        _object(comparisons[name], name).get("p_normalized"),
                                        "p_normalized",
                                    )
                                    for name in comparison_names
                                ],
                            ),
                            _series(
                                "Z",
                                comparison_x,
                                [
                                    _finite(
                                        _object(comparisons[name], name).get("z_normalized"),
                                        "z_normalized",
                                    )
                                    for name in comparison_names
                                ],
                            ),
                        ],
                        yscale="log",
                    ),
                ],
                question="What do the completed 33-point 2T mathematical trial mode histories show?",
                fixed=f"s={config.get('s')} M={config.get('M')} h={config.get('h')} H={resolved.get('H')}",
                scan="128 accepted macros at 2 nominal periods.",
                reference="Same-h and half-h trial references with mode1 target and mode16 sentinel.",
                pass_scope="Mathematical trial only; production_accepted=false, no production pass or cumulative ledger.",
                caption=(
                    "Actual periodic trial mode histories and normalized comparison rows. "
                    f"Sorted comparison index mapping: {comparison_mapping}."
                ),
                status="mathematical trial",
                device=str(config.get("device", "not recorded")),
            )
        )
    figures.append(
        _figure(
            "microstep-reference",
            "Microstep h versus h/2 reference response",
            "independent_reference",
            [
                _panel(
                    "Relative response difference",
                    "Transverse mode",
                    "relative h/h2 difference",
                    [
                        _series(
                            "nonlinear reference",
                            [
                                _finite(row.get("mode"), "mode")
                                for row in micro_rows
                            ],
                            [
                                _finite(row.get("reference_h_h2_relative"), "reference")
                                for row in micro_rows
                            ],
                        )
                    ],
                    yscale="log",
                ),
                _panel(
                    "Phase difference",
                    "Transverse mode",
                    "rad",
                    [
                        _series(
                            "h versus h/2",
                            [
                                _finite(row.get("mode"), "mode")
                                for row in micro_rows
                            ],
                            [
                                abs(_finite(row.get("phase_difference_rad"), "phase"))
                                for row in micro_rows
                            ],
                        )
                    ],
                    yscale="log",
                ),
            ],
            question="Does the fine reference response change when h is halved?",
            fixed="Nonuniform initial state, nominal target period and physical mode normalization.",
            scan="Mode 1 target response and mode 16 stability-only response.",
            reference="Independent h/2 nonlinear reference response.",
            pass_scope="Reference agreement only; mode 16 is not target accuracy acceptance.",
            caption="Actual h/h2 response and phase-difference rows.",
            status="reference evidence",
            device="CPU",
        )
    )
    series_radius = []
    series_error = []
    series_phase = []
    for s in sorted({int(row["s"]) for row in linear_rows}):
        group = sorted(
            (row for row in linear_rows if int(row["s"]) == s),
            key=lambda row: int(row["M"]),
        )
        x = [_finite(row.get("M"), "M") for row in group]
        series_radius.append(
            _series(f"s={s}", x, [_finite(row.get("outer_picard_radius"), "radius") for row in group])
        )
        series_error.append(
            _series(f"s={s}", x, [_finite(row.get("error"), "error") for row in group])
        )
        series_phase.append(
            _series(f"s={s}", x, [abs(_finite(row.get("phase"), "phase")) for row in group])
        )
    figures.append(
        _figure(
            "analytic-selection",
            "Analytic Hall nodal-map selection control",
            "independent_reference",
            [
                _panel("Outer Picard radius", "M", "spectral radius", series_radius, xscale="log", yscale="log"),
                _panel("Target response error", "M", "normalized error", series_error, xscale="log", yscale="log"),
                _panel("Absolute phase error", "M", "rad", series_phase, xscale="log", yscale="log"),
            ],
            question="How do s=3 and s=5 nodal maps behave in the analytic Hall control?",
            fixed="Single uniform transverse Hall model and exact direct nodal-map solve.",
            scan="M=1..128 for s=3 and s=5.",
            reference="Matrix exponential and direct linear nodal-map solve.",
            pass_scope="Narrow analytic control only; not nonlinear WR or production acceptance.",
            caption="Picard radius, target error and phase from the analytic control.",
            status="analytic control",
            device="CPU",
        )
    )
    backend_x = [float(index) for index in range(len(backend_rows))]
    figures.append(
        _figure(
            "short-backend-residual",
            "Short periodic CPU/CUDA residual consistency",
            "verification",
            [
                _panel(
                    "Relative backend residual",
                    "Compared quantity index",
                    "relative error",
                    [
                        _series(
                            ", ".join(str(row.get("name")) for row in backend_rows),
                            backend_x,
                            [_finite(row.get("relative"), "relative") for row in backend_rows],
                        )
                    ],
                    yscale="log",
                ),
                _panel(
                    "Absolute backend residual",
                    "Compared quantity index",
                    "raw array error",
                    [
                        _series(
                            "CPU/CUDA",
                            backend_x,
                            [_finite(row.get("abs_error"), "abs_error") for row in backend_rows],
                        )
                    ],
                    yscale="log",
                ),
            ],
            question="Are short periodic CPU and CUDA callback arrays consistent?",
            fixed="The listed short periodic state and callback adapters.",
            scan="Accepted P/Z, final nodes and fine P/Z/G/C tensors.",
            reference="Matched CPU execution of the same callback path.",
            pass_scope="Generic synthetic short comparison; no physical precision claim if P backgrounds mix.",
            caption="Actual short periodic backend residual rows.",
            status="consistency evidence",
            device="CPU and CUDA",
        )
    )
    parity_series_work = []
    parity_series_cpu = []
    parity_series_cuda = []
    for case in ("nonlinear", "hall"):
        for s in (3, 5):
            group = sorted(
                (
                    row
                    for row in wr_cases
                    if row.get("case") == case and int(row.get("s", -1)) == s
                ),
                key=lambda row: int(row["M"]),
            )
            if not group:
                continue
            label = f"{case} s={s}"
            x = [_finite(row.get("M"), "M") for row in group]
            parity_series_work.append(
                _series(
                    label,
                    x,
                    [
                        _finite(
                            _object(row.get("work"), "work").get("field_callbacks"),
                            "field_callbacks",
                        )
                        for row in group
                    ],
                )
            )
            parity_series_cpu.append(
                _series(
                    label,
                    x,
                    [
                        _finite(
                            _object(row.get("timing"), "timing").get("cpu_seconds"),
                            "cpu_seconds",
                        )
                        for row in group
                    ],
                )
            )
            parity_series_cuda.append(
                _series(
                    label,
                    x,
                    [
                        _finite(
                            _object(row.get("timing"), "timing").get("cuda_seconds"),
                            "cuda_seconds",
                        )
                        for row in group
                    ],
                )
            )
    figures.append(
        _figure(
            "generic-wr-parity",
            "Generic WR work and serial backend cost",
            "verification",
            [
                _panel("Actual field callbacks", "M", "callbacks", parity_series_work, xscale="log"),
                _panel("CPU observation", "M", "seconds", parity_series_cpu, xscale="log", yscale="log"),
                _panel("CUDA observation", "M", "seconds", parity_series_cuda, xscale="log", yscale="log"),
            ],
            question="Do callback work and serial backend observations match the generic matrix?",
            fixed="Small synthetic nonlinear and Hall states with accepted callback APIs.",
            scan="s=3/5 and M=3/8.",
            reference="Matched CPU execution and independent Hall matrix exponential.",
            pass_scope="Generic correctness/counter evidence; serial synthetic scans and no time-parallel speedup claim.",
            caption="Actual generic WR counters and one-execution timing observations.",
            status="verification scope",
            device="CPU and CUDA",
        )
    )
    strict_cpu = _object(massless.get("strict_k1_failure_cpu"), "strict_k1_failure_cpu")
    strict_gpu = _object(massless.get("strict_k1_failure_cuda"), "strict_k1_failure_cuda")
    cpu_primitive = _object(strict_cpu.get("primitive"), "strict_cpu.primitive")
    cpu_legacy = _object(strict_cpu.get("legacy_full_cpu"), "strict_cpu.legacy")
    gpu_primitive = _object(strict_gpu.get("primitive"), "strict_gpu.primitive")
    periodic_case = next(
        (
            row
            for row in _items(massless.get("cases"), "massless.cases")
            if _object(row, "case").get("fixture") == "periodic_nonuniform"
        ),
        None,
    )
    comparisons = (
        [
            _object(row, "periodic.comparisons[]")
            for row in _items(_object(periodic_case, "periodic_case").get("comparisons"), "comparisons")
        ]
        if periodic_case is not None
        else []
    )
    figures.append(
        _figure(
            "massless-scope",
            "Massless field hard-gate and consistency scope",
            "verification",
            [
                _panel(
                    "Strict original K1 wall-electron flux",
                    "Implementation",
                    "1/(m^2*s)",
                    [
                        _series(
                            "raw failure",
                            [0.0, 1.0, 2.0],
                            [
                                _finite(cpu_primitive.get("raw"), "cpu raw"),
                                _finite(cpu_legacy.get("raw"), "legacy raw"),
                                _finite(gpu_primitive.get("raw"), "gpu raw"),
                            ],
                        )
                    ],
                    horizontal_lines=[
                        {
                            "y": _finite(cpu_primitive.get("threshold"), "threshold"),
                            "label": "original hard threshold",
                        }
                    ],
                ),
                _panel(
                    "Periodic positive-gate normalized residuals",
                    "Compared field quantity",
                    "dimensionless error",
                    [
                        _series(
                            "periodic compared quantities",
                            [float(index) for index in range(len(comparisons))],
                            [
                                _finite(row.get("normalized_error"), "normalized_error")
                                for row in comparisons
                            ],
                        )
                    ]
                    if comparisons
                    else [
                        _series(
                            "not available",
                            [0.0],
                            [0.0],
                        )
                    ],
                    yscale="log",
                ),
            ],
            question="Which massless outcomes are strict failures, consistency-only, or positive hard gates?",
            fixed="Original declared budgets and actual field primitive owner contexts.",
            scan="Strict K1 CPU/legacy/CUDA failure and periodic normalized residuals.",
            reference="Original hard thresholds and matched CPU/CUDA comparisons.",
            pass_scope="Strict K1 remains a failure; soft K1 is consistency-only; periodic is the positive hard gate.",
            caption="Actual massless field failure and periodic comparison values.",
            status="scoped evidence",
            device="CPU and CUDA",
        )
    )

    if "periodic_comparison" in paths:
        figures.extend(
            _periodic_comparison_figures(index, paths["periodic_comparison"])
        )
    if "matched_gpu_cost" in paths:
        figures.append(_matched_gpu_cost_figure(index, paths["matched_gpu_cost"]))
    precheck_roles = {"production_precheck", "boundary_precheck"}
    present_prechecks = precheck_roles & set(paths)
    if present_prechecks and present_prechecks != precheck_roles:
        raise InputError("production and boundary prechecks must be supplied together")
    if present_prechecks:
        figures.extend(
            _first_window_boundary_figures(
                index,
                paths["production_precheck"],
                paths["boundary_precheck"],
            )
        )

    known_issues = []
    if "reviewed_sections" in paths:
        known_issues.append(
            {
                "label": "Retained reviewed prior RZ sections",
                "path": str(paths["reviewed_sections"]),
            }
        )
    if "energy_attribution" in paths:
        known_issues.append(
            {
                "label": "Retained prior A0/T1 energy-step attribution, not current WR acceptance",
                "path": str(paths["energy_attribution"]),
            }
        )
    for row in _items(manifest.get("availability"), "availability"):
        if row["status"] in {"blocked", "not_run", "partial"}:
            known_issues.append(
                {
                    "label": f"{row['id']}: {row['status']}",
                    "path": str(paths["cfl"]),
                }
            )
    return {
        "title": manifest["title"],
        "introduction": manifest["introduction"],
        "conclusions": manifest["conclusions"],
        "known_issues": known_issues,
        "archive": {"label": "Immutable raw campaign root", "path": str(raw_root)},
        "notation": [
            {"symbol": "P,Z", "meaning": "Retained ion nodes and field state."},
            {"symbol": "h,H,M,s", "meaning": "Microstep, macro H, microsteps per H, coarse node count."},
            {"symbol": "rho", "meaning": "Spectral radius of a fixed-point map."},
        ],
        "availability": manifest["availability"],
        "figures": figures,
    }


def _entry_path(base: Path, entry: dict[str, Any], key: str) -> Path:
    value = entry.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"run entry {entry.get('id', '?')} lacks {key}")
    return _resolve(base, value, key)


def _periodic_inventory(
    manifest: dict[str, Any],
    base: Path,
    index: InputIndex,
) -> list[dict[str, Any]]:
    result = []
    for position, raw in enumerate(
        _items(manifest.get("periodic_runs", []), "manifest.periodic_runs")
    ):
        entry = _object(raw, f"periodic_runs[{position}]")
        ident = _text(entry, "id", f"periodic_runs[{position}]")
        label = _text(entry, "label", f"periodic_runs[{position}]")
        run_json = _entry_path(base, entry, "run_json")
        history = _entry_path(base, entry, "history_jsonl")
        snapshots = _entry_path(base, entry, "snapshots")
        checkpoint = (
            _entry_path(base, entry, "checkpoint")
            if entry.get("checkpoint") is not None
            else None
        )
        for path, role in (
            (run_json, f"{ident} run.json"),
            (history, f"{ident} history.jsonl"),
            (snapshots, f"{ident} snapshots"),
        ):
            index.file(path, role)
        if checkpoint is not None:
            index.file(checkpoint, f"{ident} checkpoint")
        run = _json_object(index, run_json, f"{ident} run.json")
        result.append(
            {
                "id": ident,
                "label": label,
                "status": str(run.get("status", "unrecorded")),
                "schema": str(run.get("schema", "")),
                "total_macros": run.get("total_macros"),
                "history_bytes": history.stat().st_size,
                "run_json": str(run_json),
                "history_jsonl": str(history),
                "snapshots": str(snapshots),
                "checkpoint": str(checkpoint) if checkpoint else None,
            }
        )
    return result


def _real_sample(
    values: np.ndarray,
    *,
    name: str,
    nphi: int,
    nc: int,
    components: int,
    sample_index: int,
) -> np.ndarray:
    expected = (nphi, components, nc) if components else (nphi, nc)
    if values.shape != expected:
        raise InputError(f"{name} shape {values.shape} != {expected}")
    sample = np.asarray(values[sample_index])
    if not np.isfinite(sample.real).all() or not np.isfinite(sample.imag).all():
        raise InputError(f"{name} contains nonfinite values")
    scale = max(float(np.max(np.abs(sample.real), initial=0.0)), 1.0)
    if float(np.max(np.abs(sample.imag), initial=0.0)) > 1.0e-10 * scale:
        raise InputError(f"{name} is not physical-real at phi sample {sample_index}")
    return sample.real.astype(np.float64, copy=True)


def _rz_snapshot(
    path: Path,
    *,
    role: str,
    index: InputIndex,
    nphi: int,
    nc: int,
    sample_index: int,
) -> dict[str, Any]:
    arrays = index.npz(path, role)
    for key in ("BN", "u_i", "n_i", "p_i", "step", "time_s", "accepted"):
        if key not in arrays:
            raise InputError(f"{path} lacks {key}")
    if not bool(np.asarray(arrays["accepted"]).item()):
        raise InputError(f"{path} is not an accepted snapshot")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "step": int(np.asarray(arrays["step"]).item()),
        "time_s": float(np.asarray(arrays["time_s"]).item()),
        "B": _real_sample(
            arrays["BN"], name="BN", nphi=nphi, nc=nc, components=3,
            sample_index=sample_index,
        ),
        "u": _real_sample(
            arrays["u_i"], name="u_i", nphi=nphi, nc=nc, components=3,
            sample_index=sample_index,
        ),
        "n": _real_sample(
            arrays["n_i"], name="n_i", nphi=nphi, nc=nc, components=0,
            sample_index=sample_index,
        ),
        "p": _real_sample(
            arrays["p_i"], name="p_i", nphi=nphi, nc=nc, components=0,
            sample_index=sample_index,
        ),
    }


def _rz_status(report: dict[str, Any]) -> str:
    status = report.get("status")
    reached = report.get("reached_macros")
    requested = report.get("global_requested_macros")
    if (
        isinstance(reached, bool)
        or not isinstance(reached, int)
        or reached < 0
        or isinstance(requested, bool)
        or not isinstance(requested, int)
        or requested <= 0
        or reached > requested
    ):
        raise InputError("RZ report has invalid accepted/requested macro counts")
    if status == "completed":
        if reached != requested:
            raise InputError("RZ completed report has not reached the requested horizon")
        return "completed"
    if status == "segment_complete":
        if reached >= requested:
            raise InputError("RZ segment_complete report reaches the whole requested horizon")
        return "segment"
    if status in {"budget_stopped", "partial"}:
        if reached >= requested:
            raise InputError("RZ partial report reaches the whole requested horizon")
        return "partial"
    if status == "failed":
        return "failure"
    raise InputError(f"unsupported RZ report status: {status}")


def _rz_panel_layout(
    component_names: list[str],
    has_middle: bool,
) -> list[tuple[str, str, bool]]:
    if component_names in (["B_phi"], ["u_phi"]):
        component = component_names[0]
        if has_middle:
            return [
                ("initial", component, True),
                ("middle", component, True),
                ("final", component, True),
                ("final", component, False),
            ]
        return [
            ("initial", component, False),
            ("initial", component, True),
            ("final", component, False),
            ("final", component, True),
        ]
    return [
        (stage, component, delta)
        for component in component_names
        for stage, delta in (("final", False), ("final", True))
    ]


def _rz_caption(run: dict[str, Any], components_text: str) -> str:
    return (
        f"Actual {run['backend']} {run['label']} {components_text} cross sections; "
        f"status={run['status_label']}, nodes={run['nodes']}, s={run['s']}, "
        f"M={run['M']}, h={run['h']}, H={run['H']}, Nc={run['nc']}; "
        f"reached {run['reached_macros']} of {run['requested_macros']} requested "
        f"macros; phi sample {run['sample_index']} of Nphi={run['nphi']}; "
        f"no smoothing or extrapolation outside the {run['nc']}-cell active mask."
    )


def _rz_status_scope(status: str) -> str:
    return {
        "completed": "completed-report descriptive evidence",
        "segment": "segment-complete prefix evidence; not the whole requested horizon",
        "partial": "partial-prefix descriptive evidence",
        "failure": "failure-report accepted-state evidence only",
    }[status]


def _read_rz_runs(
    manifest: dict[str, Any],
    base: Path,
    index: InputIndex,
    sections: Any,
) -> list[dict[str, Any]]:
    runs = []
    for position, raw in enumerate(
        _items(manifest.get("rz_runs", []), "manifest.rz_runs")
    ):
        entry = _object(raw, f"rz_runs[{position}]")
        ident = _text(entry, "id", f"rz_runs[{position}]")
        label = _text(entry, "label", f"rz_runs[{position}]")
        backend = _text(entry, "backend", f"rz_runs[{position}]")
        report_path = _entry_path(base, entry, "report")
        mapping_path = _entry_path(base, entry, "input_mapping")
        report = _json_object(index, report_path, f"{ident} report")
        if report.get("schema") != "fomhd-warp-wr-rz-production-v2":
            raise InputError(f"{ident} report schema mismatch")
        status_label = _rz_status(report)
        actual = _object(report.get("actual_grid"), f"{ident}.actual_grid")
        nphi = int(actual.get("Nphi"))
        nc = int(actual.get("Nc"))
        sample_index = entry.get("sample_index", 0)
        if isinstance(sample_index, bool) or not isinstance(sample_index, int):
            raise InputError(f"{ident}.sample_index must be an integer")
        if not 0 <= sample_index < nphi:
            raise InputError(f"{ident}.sample_index is outside Nphi")
        mapping_arrays = index.npz(mapping_path, f"{ident} mapping")
        required = ("R", "Z", "polygon")
        if any(key not in mapping_arrays for key in required):
            raise InputError(f"{ident} mapping lacks R/Z/polygon")
        R = np.asarray(mapping_arrays["R"], dtype=np.float64)
        Z = np.asarray(mapping_arrays["Z"], dtype=np.float64)
        polygon = np.asarray(mapping_arrays["polygon"], dtype=np.float64)
        if R.shape != (nc,) or Z.shape != (nc,):
            raise InputError(f"{ident} mapping active-cell count mismatch")
        mapping = sections.Mapping(
            R,
            Z,
            polygon,
            sections.axis_boundaries(R, "R"),
            sections.axis_boundaries(Z, "Z"),
        )
        snapshots = _object(entry.get("snapshots"), f"{ident}.snapshots")
        snapshot_paths = {}
        for key in (
            "control_initial",
            "control_final",
            "perturbed_initial",
            "perturbed_final",
        ):
            value = snapshots.get(key)
            if not isinstance(value, str) or not value.strip():
                raise InputError(f"{ident}.snapshots lacks {key}")
            snapshot_paths[key] = _resolve(base, value, key)
        middle_paths = {
            key: snapshots.get(key)
            for key in ("control_middle", "perturbed_middle")
        }
        if (middle_paths["control_middle"] is None) != (
            middle_paths["perturbed_middle"] is None
        ):
            raise InputError(f"{ident} middle snapshots must be supplied together")
        if middle_paths["control_middle"] is not None:
            for key, value in middle_paths.items():
                if not isinstance(value, str) or not value.strip():
                    raise InputError(f"{ident}.snapshots has invalid {key}")
                snapshot_paths[key] = _resolve(base, value, key)
        snapshot_data = {
            key: _rz_snapshot(
                path,
                role=f"{ident} {key}",
                index=index,
                nphi=nphi,
                nc=nc,
                sample_index=sample_index,
            )
            for key, path in snapshot_paths.items()
        }
        has_middle = "control_middle" in snapshot_data
        stages = ["initial", "middle", "final"] if has_middle else ["initial", "final"]
        for stage in stages:
            left = snapshot_data[f"control_{stage}"]
            right = snapshot_data[f"perturbed_{stage}"]
            if left["step"] != right["step"] or not math.isclose(
                left["time_s"], right["time_s"], rel_tol=1e-12, abs_tol=1e-24
            ):
                raise InputError(f"{ident} {stage} control/perturbed snapshots are not matched")
        stage_times = [snapshot_data[f"control_{stage}"]["time_s"] for stage in stages]
        if any(left >= right for left, right in zip(stage_times, stage_times[1:])):
            raise InputError(f"{ident} snapshots are not strictly ordered initial/middle/final")
        checkpoints = _object(entry.get("checkpoints", {}), f"{ident}.checkpoints")
        for key, value in checkpoints.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise InputError(f"{ident}.checkpoints is invalid")
            index.file(_resolve(base, value, key), f"{ident} checkpoint {key}")
        case = _object(report.get("case"), f"{ident}.case")
        plan = _object(report.get("source_plan"), f"{ident}.source_plan")
        runs.append(
            {
                "id": ident,
                "label": label,
                "backend": backend,
                "report": report,
                "report_status": str(report["status"]),
                "status_label": status_label,
                "report_path": str(report_path),
                "mapping": mapping,
                "mapping_path": str(mapping_path),
                "sample_index": sample_index,
                "nphi": nphi,
                "nc": nc,
                "nodes": int(case.get("nodes")),
                "s": int(case.get("s")),
                "M": int(case.get("M")),
                "h": float(case.get("h")),
                "H": float(plan.get("effective_macro_s")),
                "reached_macros": int(report.get("reached_macros")),
                "requested_macros": int(report.get("global_requested_macros")),
                "has_middle": has_middle,
                "stages": stages,
                "hard_failures": int(_object(report.get("check_counts"), "counts").get("hard_fail", 0)),
                "observe_failures": int(_object(report.get("check_counts"), "counts").get("observe_fail", 0)),
                "work_total": _object(report.get("work_total"), "work_total"),
                "snapshots": snapshot_data,
            }
        )
    return runs


def _rz_work_figure(run: dict[str, Any], suffix: str) -> dict[str, Any]:
    report = run["report"]
    work = run["work_total"]
    keys = (
        "complete_advance_calls",
        "full_ion_calls",
        "field_evaluations",
        "coarse_transport_calls",
        "fine_full_stage_calls",
    )
    macros = _items(report.get("macros"), "macros")
    if not macros:
        raise InputError("RZ work figure requires one recorded macro")
    macro = _object(macros[0], "macros[0]")
    macro_number = int(macro.get("macro"))
    macro_step = int(macro.get("step"))
    macro_time = _finite(macro.get("time_s"), "macros[0].time_s")
    macro_label = (
        f"First recorded macro {macro_number} "
        f"(step={macro_step}, t={macro_time:.9e}s)"
    )
    residual_series = []
    for population in ("control_coupling_metadata", "perturbed_coupling_metadata"):
        raw_metadata = macro.get(population)
        if isinstance(raw_metadata, list):
            raw_metadata = _items(raw_metadata, population)[0]
        metadata = _object(raw_metadata, population)
        history = [
            _object(row, f"{population}.residual_history[]")
            for row in _items(metadata.get("residual_history"), "residual_history")
        ]
        residual_series.append(
            _series(
                population.replace("_coupling_metadata", ""),
                [_finite(row.get("iteration"), "iteration") for row in history],
                [_finite(row.get("normalized_max"), "normalized_max") for row in history],
            )
        )
    counts = _object(report.get("check_counts"), "check_counts")
    actual_work = [
        _series(
            key,
            [float(index)],
            [_finite(work.get(key), key)],
        )
        for index, key in enumerate(keys)
    ]
    return _figure(
        f"rz-work-checks{suffix}",
        "RZ actual work and WR convergence scope",
        "verification",
        [
            _panel(
                "Report-total selected work counters",
                "Counter index (legend gives actual counter)",
                "calls",
                actual_work,
            ),
            _panel(
                f"{macro_label}: control/perturbed normalized WR residual",
                "Outer iteration",
                "normalized defect",
                residual_series,
                yscale="log",
            ),
        ],
        question="What report work and first-recorded-macro residual are present?",
        fixed=(
            f"nodes={run['nodes']} s={run['s']} M={run['M']} "
            f"h={run['h']} H={run['H']}"
        ),
        scan=(
            f"{run['reached_macros']} accepted macro(s) of "
            f"{run['requested_macros']} requested; residual history is "
            f"{macro_label}."
        ),
        reference="Actual full-stage, field, transport and convergence records.",
        pass_scope=(
            f"Report status={run['status_label']}; "
            f"hard_fail={counts.get('hard_fail')} "
            f"observe_fail={counts.get('observe_fail')}. "
            "Residual history is the first recorded macro only and is not "
            "evidence of whole-prefix convergence. "
            "Expensive/fine work is mixed and does not prove net GPU acceleration."
        ),
        caption=(
            f"Actual report work counters plus the {macro_label} residual history; "
            "this is not whole-prefix convergence evidence."
        ),
        status=run["status_label"],
        device=run["backend"],
    )


def _rz_field(run: dict[str, Any], pair: str, field: str) -> np.ndarray:
    snapshot = run["snapshots"][pair]
    if field.startswith("B_"):
        return snapshot["B"][("B_R", "B_phi", "B_Z").index(field)]
    if field.startswith("u_"):
        return snapshot["u"][("u_R", "u_phi", "u_Z").index(field)]
    if field == "n_i":
        return snapshot["n"]
    if field == "p_i":
        return snapshot["p"]
    raise InputError(f"unknown RZ field {field}")


def _render_rz_pages(
    runs: list[dict[str, Any]],
    output: Path,
    sections: Any,
    t0: Any,
    start_number: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    specifications = [
        (
            "rz-magnetic-toroidal-initial-final",
            "RZ magnetic toroidal cross sections",
            ["B_phi"],
        ),
        (
            "rz-magnetic-poloidal-final",
            "RZ magnetic poloidal cross sections",
            ["B_R", "B_Z"],
        ),
        (
            "rz-velocity-toroidal-initial-final",
            "RZ ion-velocity toroidal cross sections",
            ["u_phi"],
        ),
        (
            "rz-velocity-poloidal-final",
            "RZ ion-velocity poloidal cross sections",
            ["u_R", "u_Z"],
        ),
        (
            "rz-density-pressure-final",
            "RZ ion density and pressure cross sections",
            ["n_i", "p_i"],
        ),
    ]
    fields = {
        "B_R": ("Radial B", "T", True),
        "B_phi": ("Toroidal B", "T", True),
        "B_Z": ("Vertical B", "T", True),
        "u_R": ("Radial ion velocity", "m/s", True),
        "u_phi": ("Toroidal ion velocity", "m/s", True),
        "u_Z": ("Vertical ion velocity", "m/s", True),
        "n_i": ("Ion density", "m^-3", False),
        "p_i": ("Ion pressure", "Pa", False),
    }
    figure_specs = []
    page_records = []
    number = start_number
    for run in runs:
        suffix = "" if len(runs) == 1 else f"-{run['id']}"
        for base_id, title, component_names in specifications:
            toroidal = component_names in (["B_phi"], ["u_phi"])
            figure_id = (
                base_id.replace(
                    "-initial-final",
                    "-initial-middle-final",
                )
                if run["has_middle"] and toroidal
                else base_id
            ) + suffix
            components_text = ", ".join(component_names)
            panels = _rz_panel_layout(component_names, run["has_middle"])
            stages = list(dict.fromkeys(stage for stage, _, _ in panels))
            if len(panels) > 4:
                raise InputError(f"{figure_id} exceeds four panels")
            control_limits = {}
            delta_limits = {}
            for component in component_names:
                controls = []
                deltas = []
                for stage in stages:
                    control = _rz_field(
                        run, f"control_{stage}", component
                    )
                    perturbed = _rz_field(
                        run, f"perturbed_{stage}", component
                    )
                    controls.append(control)
                    deltas.append(perturbed - control)
                if fields[component][2]:
                    control_limits[component] = sections.symmetric_limits(*controls)
                else:
                    control_limits[component] = sections.sequential_limits(
                        np.concatenate(controls)
                    )
                delta_limits[component] = sections.symmetric_limits(*deltas)
            rows = 1 if len(panels) <= 2 else 2
            columns = 1 if len(panels) == 1 else 2
            fig, axes_grid = plt.subplots(
                rows,
                columns,
                figsize=(11.5, 4.8 if rows == 1 else 9.0),
            )
            axes = np.asarray(axes_grid).reshape(-1)
            descriptions = []
            try:
                for index, (stage, component, delta) in enumerate(panels):
                    axis = axes[index]
                    control = _rz_field(run, f"control_{stage}", component)
                    perturbed = _rz_field(run, f"perturbed_{stage}", component)
                    values = perturbed - control if delta else control
                    low, high = (
                        delta_limits[component]
                        if delta
                        else control_limits[component]
                    )
                    norm = (
                        TwoSlopeNorm(vmin=low, vcenter=0.0, vmax=high)
                        if delta or fields[component][2]
                        else Normalize(vmin=low, vmax=high)
                    )
                    label, unit, _ = fields[component]
                    time_s = run["snapshots"][f"control_{stage}"]["time_s"]
                    sections._plot_cells(
                        axis,
                        run["mapping"],
                        values,
                        norm=norm,
                        cmap="RdBu_r" if delta or fields[component][2] else "cividis",
                        title=(
                            f"{'Delta ' if delta else 'Control '}{stage} {component}"
                        ),
                        colorbar_label=unit,
                        t0=t0,
                    )
                    t0.panel_label(axis, chr(ord("a") + index))
                    descriptions.append(
                        {
                            "letter": chr(ord("a") + index),
                            "description": (
                                f"{label} {'perturbed-minus-control' if delta else 'control'}, "
                                f"stage={stage}, t={time_s:.9e}s, phi sample={run['sample_index']}, "
                                f"unit={unit}, limits=[{low:.6e},{high:.6e}]."
                            ),
                        }
                    )
                for axis in axes[len(panels):]:
                    axis.set_visible(False)
                sections._save_figure(
                    fig, output, figure_id, t0, plt
                )
            finally:
                plt.close(fig)
            caption = _rz_caption(run, components_text)
            record = {
                "number": number,
                "title": title,
                "caption": (
                    f"Question: What real {'/'.join(stages)} RZ {components_text} "
                    "differences are present? Controls: matched control and "
                    "perturbed times. "
                    f"Criterion: {_rz_status_scope(run['status_label'])}. {caption}"
                ),
                "source_kind": "nonlinear_evolution",
                "status": run["status_label"],
                "device": run["backend"],
                "panels": descriptions,
                "image": f"{figure_id}.png",
                "svg": f"{figure_id}.svg",
                "data": "data.json",
            }
            page = {
                "id": figure_id,
                "title": title,
                "introduction": "Direct active-cell control and perturbed-minus-control maps from matched snapshots.",
                "figures": [record],
                "notes": [
                    f"Source report: {run['report_path']}",
                    f"Report status: {run['report_status']} ({run['status_label']}).",
                    f"Active-cell mapping: {run['mapping_path']}",
                    f"Hard failures: {run['hard_failures']}; observed failures: {run['observe_failures']}.",
                ],
            }
            figure_specs.append({"id": figure_id})
            page_records.append(page)
            number += 1
    return figure_specs, page_records


def _availability_html(rows: list[dict[str, Any]]) -> str:
    body = "".join(
        f"<tr><td>{html.escape(str(row['id']))}</td>"
        f"<td>{html.escape(str(row['status']))}</td>"
        f"<td>{html.escape(str(row['detail']))}</td></tr>"
        for row in rows
    )
    return (
        '<section id="availability"><h2>Data availability</h2>'
        "<p>Missing or blocked rows are explicit; no placeholder values are plotted.</p>"
        "<table><thead><tr><th>Dataset</th><th>Status</th><th>Scope</th>"
        f"</tr></thead><tbody>{body}</tbody></table></section>"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def build_gallery(
    manifest_path: Path,
    output: Path,
    source_root: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    output = output.resolve()
    source_root = source_root.resolve()
    manifest, paths, raw_root, base = load_manifest(manifest_path)
    inputs = [manifest_path, *paths.values()]
    output = _safe_output(
        output,
        [path for path in inputs if path.resolve().is_file()],
        raw_root,
    )

    index = InputIndex()
    index.file(manifest_path, "manifest")
    canonical_paths = {name: source_root / name for name in CANONICAL_SOURCES}
    for name, path in canonical_paths.items():
        if not path.is_file():
            raise InputError(f"canonical source is absent: {path}")
    t2 = _module(
        "wr_gallery_t2", canonical_paths["tools/plots/build_a1_t2_report.py"]
    )
    a0, t0 = t2.load_sources(source_root)
    sections = _module(
        "wr_gallery_sections", canonical_paths["tools/plots/build_050_sections.py"]
    )
    periodic_inventory = _periodic_inventory(manifest, base, index)
    rz_runs = _read_rz_runs(manifest, base, index, sections)
    data = _build_data(manifest, paths, raw_root, index)
    data["periodic_run_inventory"] = periodic_inventory
    data["rz_run_evidence"] = [
        {
            "id": run["id"],
            "label": run["label"],
            "backend": run["backend"],
            "report_status": run["report_status"],
            "status": run["status_label"],
            "report": run["report_path"],
            "mapping": run["mapping_path"],
            "Nc": run["nc"],
            "Nphi": run["nphi"],
            "nodes": run["nodes"],
            "s": run["s"],
            "M": run["M"],
            "h": run["h"],
            "H": run["H"],
            "reached_macros": run["reached_macros"],
            "requested_macros": run["requested_macros"],
            "hard_failures": run["hard_failures"],
            "observe_failures": run["observe_failures"],
            "has_middle": run["has_middle"],
            "stages": run["stages"],
            "sample_index": run["sample_index"],
            "snapshots": {
                name: {
                    "path": snapshot["path"],
                    "sha256": snapshot["sha256"],
                    "step": snapshot["step"],
                    "time_s": snapshot["time_s"],
                }
                for name, snapshot in run["snapshots"].items()
            },
        }
        for run in rz_runs
    ]
    for position, run in enumerate(rz_runs):
        suffix = "" if len(rz_runs) == 1 else f"-{run['id']}"
        data["figures"].append(_rz_work_figure(run, suffix))
    data = t2.validate_data(data)
    output = _safe_output(
        output,
        [Path(record["path"]) for record in index.export()],
        raw_root,
    )

    source_hashes = {
        "build_wr_gallery.py": sha256_file(Path(__file__).resolve()),
        **{name: sha256_file(path) for name, path in canonical_paths.items()},
    }
    for record in index.export():
        source_hashes[f"input:{','.join(record['roles'])}:{Path(record['path']).name}"] = record[
            "sha256"
        ]
    before = {path: sha256_file(path) for path in index.records}
    output.mkdir(parents=False, exist_ok=False)
    data_path = output / "data.json"
    _write_json(data_path, data)
    with tempfile.TemporaryDirectory(prefix="wr-gallery-mpl-") as cache:
        old = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            t2.render_figures(data, output, t0)
            custom_specs, custom_pages = _render_rz_pages(
                rz_runs,
                output,
                sections,
                t0,
                len(data["figures"]) + 1,
            )
        finally:
            if old is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old
    manifest_rendered = t2.build_manifest(data, data_path, output)
    manifest_rendered["pages"].extend(custom_pages)
    manifest_out = output / "manifest.json"
    _write_json(manifest_out, manifest_rendered)
    html_path = output / "index.html"
    rendered = a0.render_report(manifest_rendered, manifest_out, html_path)
    rendered = t2.embed_svgs(
        rendered,
        [*data["figures"], *custom_specs],
        output,
    )
    availability = _availability_html(data["availability"])
    marker = '<section id="conclusions">'
    if marker not in rendered:
        raise RuntimeError("canonical report lacks conclusions marker")
    rendered = rendered.replace(marker, availability + marker, 1)
    html_path.write_text(rendered, encoding="utf-8")

    provenance = {
        "schema": "fomhd-wr-gallery-provenance-v1",
        "builder_sha256": source_hashes["build_wr_gallery.py"],
        "source_hashes": source_hashes,
        "inputs": index.export(),
        "manifest": str(manifest_path),
        "raw_root": str(raw_root),
        "rendered_figures": len(data["figures"]) + len(custom_specs),
        "self_contained_svg": True,
    }
    _write_json(output / "provenance.json", provenance)
    index.verify_unchanged()
    for path, digest in before.items():
        if sha256_file(path) != digest:
            raise InputError(f"input changed during render: {path}")
    t2.verify_output(data, output, canonical_paths["tools/plots/build_a0_report.py"])
    for specification in custom_specs:
        name = specification["id"]
        if not (output / f"{name}.png").is_file() or not (output / f"{name}.svg").is_file():
            raise InputError(f"custom RZ figure is incomplete: {name}")
        if f'href="{name}.svg"' not in html_path.read_text():
            raise InputError(f"HTML lacks custom RZ SVG link: {name}")
    sections.write_checksums(output)
    if (output / "index.html").read_text().casefold().find("https://") >= 0:
        raise InputError("gallery HTML contains a remote HTTPS asset")
    return {
        "status": "ok",
        "output": str(output),
        "figures": len(data["figures"]) + len(custom_specs),
        "panels": (
            sum(len(figure["panels"]) for figure in data["figures"])
            + sum(len(page["figures"][0]["panels"]) for page in custom_pages)
        ),
        "availability": len(data["availability"]),
        "inputs": len(index.records),
        "html_bytes": html_path.stat().st_size,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = build_gallery(args.manifest, args.output, args.source_root)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
