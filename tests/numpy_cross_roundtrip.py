#!/usr/bin/env python3
"""Cross-check the native codec against NumPy in both directions."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def expected_manifest() -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    scalar_f8 = np.asarray(3.5, dtype=">f8")
    scalar_u8 = np.asarray(np.iinfo(np.uint64).max, dtype="<u8")
    scalar_b1 = np.asarray(True, dtype="|b1")
    vector_f8 = np.asarray([3.0, -1.5, 0.0], dtype=">f8")
    matrix_c16 = np.asarray(
        [
            [complex(-1.5, -2.25), complex(-0.0, 4.0)],
            [complex(9.0, -0.0), complex(-8.0, 0.125)],
        ],
        dtype=">c16",
    )
    unicode_chinese = np.asarray("A0中文", dtype=">U8")
    zero_size = np.empty((2, 0, 3), dtype="<f8")

    return {
        "scalar_f8": ("<f8", (), scalar_f8.astype("<f8").tobytes()),
        "scalar_u8": ("<u8", (), scalar_u8.astype("<u8").tobytes()),
        "scalar_b1": ("|b1", (), scalar_b1.tobytes()),
        "vector_f8": ("<f8", (3,), vector_f8.astype("<f8").tobytes()),
        "matrix_c16": ("<c16", (2, 2), matrix_c16.astype("<c16").tobytes()),
        "unicode_chinese": (
            "<U8",
            (),
            "A0中文".encode("utf-32-le").ljust(32, b"\0"),
        ),
        "zero_size": ("<f8", (2, 0, 3), zero_size.tobytes()),
    }


def parse_probe(output: str) -> dict[str, tuple[str, tuple[int, ...], str]]:
    parsed: dict[str, tuple[str, tuple[int, ...], str]] = {}
    for line in output.splitlines():
        key, descr, shape, digest = line.split("\t")
        dimensions = tuple(int(item) for item in shape.split(",") if item)
        parsed[key] = (descr, dimensions, digest)
    return parsed


def verify_numpy_to_native(probe: Path, directory: Path) -> None:
    source = directory / "numpy_input.npz"
    arrays = {
        "scalar_f8": np.asarray(3.5, dtype=">f8"),
        "scalar_u8": np.asarray(np.iinfo(np.uint64).max, dtype="<u8"),
        "scalar_b1": np.asarray(True, dtype="|b1"),
        "vector_f8": np.asarray([3.0, -1.5, 0.0], dtype=">f8"),
        "matrix_c16": np.asarray(
            [
                [complex(-1.5, -2.25), complex(-0.0, 4.0)],
                [complex(9.0, -0.0), complex(-8.0, 0.125)],
            ],
            dtype=">c16",
        ),
        "unicode_chinese": np.asarray("A0中文", dtype=">U8"),
        "zero_size": np.empty((2, 0, 3), dtype="<f8"),
    }
    np.savez_compressed(source, **arrays)
    completed = subprocess.run(
        [str(probe), "read", str(source)],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = parse_probe(completed.stdout)
    expected = expected_manifest()
    assert actual.keys() == expected.keys(), "C++ reader returned unexpected keys"
    for key, (descr, shape, payload) in expected.items():
        actual_descr, actual_shape, actual_digest = actual[key]
        assert actual_descr == descr, f"{key}: canonical dtype mismatch"
        assert actual_shape == shape, f"{key}: shape mismatch"
        assert actual_digest == hashlib.sha256(payload).hexdigest(), (
            f"{key}: canonical payload mismatch"
        )


def verify_native_to_numpy(probe: Path, directory: Path) -> None:
    generated = directory / "native_output.npz"
    subprocess.run([str(probe), "write", str(generated)], check=True)
    with np.load(generated, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "complex",
            "doubles",
            "empty",
            "flag",
            "label",
            "offset",
            "step",
        }
        np.testing.assert_array_equal(
            archive["doubles"], np.asarray([[1.25, -2.5], [0.0, 8.75]], dtype="<f8")
        )
        assert archive["doubles"].dtype == np.dtype("<f8")
        np.testing.assert_array_equal(
            archive["complex"],
            np.asarray(
                [complex(-1.5, -2.25), complex(-0.0, 4.0), complex(9.0, -0.0)],
                dtype="<c16",
            ),
        )
        assert np.signbit(archive["complex"][1].real)
        assert archive["empty"].shape == (2, 0, 3)
        assert bool(archive["flag"]) is True
        assert str(archive["label"]) == "中文"
        assert float(archive["offset"]) == -7.5
        assert int(archive["step"]) == np.iinfo(np.uint64).max


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", type=Path, required=True)
    arguments = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="fomhd-numpy-roundtrip-") as text:
        directory = Path(text)
        verify_numpy_to_native(arguments.probe, directory)
        verify_native_to_numpy(arguments.probe, directory)
    print("NumPy to native and native to NumPy checks passed")


if __name__ == "__main__":
    main()
