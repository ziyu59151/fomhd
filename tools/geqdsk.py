"""Bounded G-EQDSK input audit; it does not create a two-fluid equilibrium."""
import argparse
import hashlib
import json
import pathlib
import re

import numpy as np

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?")


def read_geqdsk(path):
    path = pathlib.Path(path)
    if not path.is_file() or path.stat().st_size > 32 << 20:
        raise ValueError("G-EQDSK input must be a regular file at most 32 MiB")
    raw = path.read_bytes()
    try:
        header, body = raw.decode("ascii").split("\n", 1)
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("G-EQDSK input must be ASCII with a header and body") from error
    dimensions = re.search(r"\s([+-]?\d+)\s+(\d+)\s+(\d+)\s*$", header)
    if dimensions is None:
        raise ValueError("missing G-EQDSK dimensions")
    nr, nz = map(int, dimensions.groups()[1:])
    if nr < 3 or nz < 3 or nr * nz > 2_000_000:
        raise ValueError("unsupported G-EQDSK dimensions")
    numbers = list(NUMBER.finditer(body))
    end = 20 + 5 * nr + nr * nz
    if len(numbers) < end + 2:
        raise ValueError("truncated G-EQDSK arrays")
    counts = np.array(
        [
            float(numbers[end].group()),
            float(numbers[end + 1].group()),
        ]
    )
    remaining_pairs = (len(numbers) - end - 2) // 2
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 0)
        or np.any(counts != np.floor(counts))
        or np.sum(counts) > remaining_pairs
    ):
        raise ValueError("invalid boundary/limiter count")
    nb, nl = int(counts[0]), int(counts[1])
    core_end = end + 2 + 2 * (nb + nl)
    if nb < 3 or len(numbers) < core_end:
        raise ValueError("truncated boundary or limiter arrays")
    core_text = body[:numbers[core_end - 1].end()]
    if NUMBER.sub("", core_text).strip():
        raise ValueError("nonnumeric data inside G-EQDSK numeric core")
    values = np.array(
        [
            float(match.group().replace("D", "E").replace("d", "e"))
            for match in numbers[:core_end]
        ]
    )
    if not np.isfinite(values).all():
        raise ValueError("nonfinite G-EQDSK value")
    extension = body[numbers[core_end - 1].end():].encode("ascii")
    if extension.strip(b" \t\r\n"):
        header_tag = header.split(maxsplit=1)[0] if header.split() else ""
        if not header_tag.upper().startswith("EFIT"):
            raise ValueError("unsupported nonnumeric G-EQDSK extension")
        extension_format = header_tag
    else:
        extension = b""
        extension_format = None
    rdim, zdim, rcentr, rleft, zmid, raxis, zaxis, psiaxis, psibdry, bcentr = values[:10]
    if min(rdim, zdim, rleft) <= 0 or psiaxis == psibdry:
        raise ValueError("invalid G-EQDSK extent or flux normalization")
    result = {"R": np.linspace(rleft, rleft + rdim, nr),
              "Z": np.linspace(zmid - zdim / 2, zmid + zdim / 2, nz),
              "psi_axis": float(psiaxis), "psi_boundary": float(psibdry),
              "magnetic_axis": [float(raxis), float(zaxis)],
              "central_R_B": [float(rcentr), float(bcentr)],
              "header_current_A": float(values[10]), "header": header,
              "sha256": hashlib.sha256(raw).hexdigest(),
              "extension": extension, "extension_format": extension_format,
              "extension_bytes": len(extension),
              "extension_sha256": hashlib.sha256(extension).hexdigest() if extension else None}
    cursor = 20
    for name in ("F", "pressure", "FFprime", "pprime"):
        result[name] = values[cursor:cursor + nr].copy()
        cursor += nr
    result["psi"] = values[cursor:cursor + nr * nz].reshape(nz, nr).copy()
    cursor += nr * nz
    result["q"] = values[cursor:cursor + nr].copy()
    result["boundary"] = values[end + 2:end + 2 + 2 * nb].reshape(nb, 2).copy()
    result["limiter"] = values[end + 2 + 2 * nb:].reshape(nl, 2).copy()
    return result


def reconstruct(eq, *, psi_factor):
    """B=grad(psi_factor*psi) cross grad(phi)+F grad(phi), explicit convention."""
    if not np.isfinite(psi_factor) or psi_factor == 0:
        raise ValueError("a finite nonzero explicit flux conversion is required")
    dz, dr = np.gradient(eq["psi"] * psi_factor, eq["Z"], eq["R"], edge_order=2)
    coordinate = (eq["psi"] - eq["psi_axis"]) / (eq["psi_boundary"] - eq["psi_axis"])
    F = np.interp(coordinate, np.linspace(0, 1, len(eq["F"])), eq["F"])
    return {"B_R": -dz / eq["R"][None, :], "B_Z": dr / eq["R"][None, :],
            "B_phi": F / eq["R"][None, :], "psi_normalized": coordinate,
            "profile_extrapolation_mask": (coordinate < 0) | (coordinate > 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=pathlib.Path)
    p.add_argument("output", type=pathlib.Path)
    p.add_argument("--psi-factor", type=float, required=True,
                   help="Explicit conversion to the assumed Wb/rad convention; not inferred from the file.")
    p.add_argument("--source-url", required=True)
    a = p.parse_args()
    eq = read_geqdsk(a.input)
    fields = reconstruct(eq, psi_factor=a.psi_factor)
    a.output.mkdir(parents=True, exist_ok=False)
    arrays = {key: value for key, value in eq.items() if isinstance(value, np.ndarray)}
    arrays.update(fields)
    np.savez_compressed(a.output / "equilibrium_audit.npz", **arrays)
    report = {
        key: value for key, value in eq.items()
        if not isinstance(value, (np.ndarray, bytes))
    }
    boundary = eq["boundary"]
    radii = np.linalg.norm(boundary - np.array(eq["magnetic_axis"]), axis=1)
    report.update({"source_url": a.source_url, "psi_factor_assumption": a.psi_factor,
                   "grid_shape_Z_R": list(eq["psi"].shape), "boundary_points": len(boundary),
                   "boundary_radius_min_max_m": [float(radii.min()), float(radii.max())],
                   "pressure_min_max_file_units": [float(eq["pressure"].min()), float(eq["pressure"].max())],
                   "F_min_max_file_units": [float(eq["F"].min()), float(eq["F"].max())],
                   "q_min_max": [float(eq["q"].min()), float(eq["q"].max())],
                   "profile_clamping_outside_closed_flux_region": int(fields["profile_extrapolation_mask"].sum()),
                   "status": "PARSED_NOT_VALIDATED_AS_A0_EQUILIBRIUM",
                   "limitations": ["COCOS/sign/2pi convention remains an explicit assumption until source confirmation.",
                                   "No unique species density, pressure split, electric field or finite-mass flows follow from a G-EQDSK file.",
                                   "The reconstructed point fields are not projected into the solver's closed staircase operators.",
                                   "This numerical example has no verified experimental discharge identity."]})
    (a.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
