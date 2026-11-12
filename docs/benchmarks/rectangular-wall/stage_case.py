"""Stage a checked copy of a frozen wall fixture for a CPU smoke run."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tomllib


def sha(path):
    return sha256(path.read_bytes()).hexdigest()


def stage(source, destination, steps):
    manifest = json.loads((source / "source.json").read_text(encoding="utf-8"))
    files = {"case": "case.toml", "geometry": "geometry.npz", "initial": "initial.npz"}
    for name, filename in files.items():
        if sha(source / filename) != manifest["input_hashes"][name]:
            raise ValueError(f"Fixture hash mismatch for {filename}")
    raw = (source / "case.toml").read_text(encoding="utf-8")
    config = tomllib.loads(raw)
    old_steps = config["time"]["nsteps"]
    if not 1 <= steps <= old_steps:
        raise ValueError(f"Expected steps between 1 and {old_steps}")
    replacements = {
        '"backend" = "cuda"': '"backend" = "cpu"',
        f'"nsteps" = {old_steps}\n': f'"nsteps" = {steps}\n',
    }
    for original, replacement in replacements.items():
        if raw.count(original) != 1:
            raise ValueError(f"Unexpected fixture assignment {original.strip()}")
        raw = raw.replace(original, replacement)
    destination.mkdir(parents=True, exist_ok=False)
    case = destination / "case.toml"
    case.write_text(raw, encoding="utf-8")
    for name in ("geometry", "initial"):
        shutil.copyfile(source / files[name], destination / files[name])
    manifest["input_hashes"]["case"] = sha(case)
    manifest["external_file"] = str((source / manifest["external_file"]).resolve())
    (destination / "source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", type=Path, help="An input subdirectory")
    parser.add_argument("destination", type=Path, help="A new external directory")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()
    print(stage(args.fixture, args.destination, args.steps))


if __name__ == "__main__":
    main()
