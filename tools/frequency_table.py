#!/usr/bin/env python3
"""Write indicative SI scales for deuterium; this is not a solver benchmark."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fomhd.stability import E_MASS, frequency_scales  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for density in (1e19, 1e20):
        for magnetic in (1.0, 3.0, 5.0):
            scales = frequency_scales(ne=density, ni=density, B=magnetic,
                                      mi=3.3435837768e-27, me=E_MASS)
            upper = math.hypot(scales["omega_pe"], scales["omega_ce"])/(2*math.pi)
            for harmonic in (10, 30):
                target_sampling = 10*harmonic*scales["fci_Hz"]
                row = dict(ne_m3=density, B_T=magnetic, ion="deuterium",
                           harmonic=harmonic, target_samples_per_period=10,
                           f_ci_Hz=scales["fci_Hz"], f_pe_Hz=scales["fpe_Hz"],
                           f_ce_Hz=scales["fce_Hz"], f_upper_hybrid_Hz=upper,
                           target_sampling_Hz=target_sampling,
                           target_sampling_dt_s=1/target_sampling,
                           plasma_32_samples_dt_s=1/(32*scales["fpe_Hz"]),
                           upper_hybrid_32_samples_dt_s=1/(32*upper))
                assert all(math.isfinite(v) and v > 0 for v in row.values()
                           if isinstance(v, (float, int)))
                rows.append(row)
    with (args.output/"frequencies.csv").open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    archive = args.output/"source"
    archive.mkdir()
    hashes = {}
    for name in ("tools/frequency_table.py", "fomhd/stability.py"):
        data = (ROOT/name).read_bytes()
        hashes[name] = hashlib.sha256(data).hexdigest()
        target = archive/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    record = dict(scope="Indicative cold frequency scales, not a complete spectrum or CFL proof; "
                  "sampling does not enforce finite-horizon amplitude or phase error.",
                  source_hashes=hashes, rows=rows)
    (args.output/"frequencies.json").write_text(json.dumps(record, indent=2, allow_nan=False)+"\n")
    print(args.output/"frequencies.csv")


if __name__ == "__main__":
    main()
