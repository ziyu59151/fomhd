#!/usr/bin/env python3
"""Save a bounded, content-addressed source snapshot for the actual executable."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

p = argparse.ArgumentParser()
p.add_argument("--source", type=pathlib.Path, required=True)
p.add_argument("--binary", type=pathlib.Path, required=True)
p.add_argument("--build", type=pathlib.Path, required=True)
p.add_argument("--compiler", required=True)
a = p.parse_args()
root = a.source.resolve()
assert pathlib.Path(subprocess.check_output(["git", "-C", str(root), "rev-parse", "--show-toplevel"], text=True).strip()).resolve() == root
files = []
for folder in ("app", "include", "src", "tests", "cmake", "tools", "jobs", "cases", "docs", "third_party"):
    for path in sorted((root / folder).rglob("*")):
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts:
            files.append(path)
for name in ("CMakeLists.txt", "CMakePresets.json", "README.md", "AGENTS.md"):
    files.append(root / name)
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
hashes = {str(path.relative_to(root)): sha(path) for path in sorted(files)}
source_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
snapshot = a.build / "source_snapshots" / source_id
snapshot.mkdir(parents=True, exist_ok=True)
for relative, digest in hashes.items():
    dest = snapshot / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(root / relative, dest)
    assert sha(dest) == digest
manifest = {
    "schema_version": 1,
    "source_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "dirty": bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)),
    "snapshot_path": str(snapshot.resolve()), "source_files": hashes,
    "source_sha256": source_id, "binary_path": str(a.binary.resolve()),
    "binary_sha256": sha(a.binary), "compiler": a.compiler,
    "dependencies": {"tomlplusplus": "3.4.0", "nlohmann_json": "3.12.0", "miniz": "3.0.2"},
}
for package in ("fftw3", "mkl-dynamic-lp64-seq", "openssl"):
    manifest["dependencies"][package] = subprocess.check_output(["pkg-config", "--modversion", package], text=True).strip()
manifest["loaded_libraries"] = subprocess.check_output(["ldd", str(a.binary)], text=True)
manifest["design_baseline_sha256"] = sha(root / "docs/pseudocode/SOURCE_BASELINE.json")
target = a.build / "build_manifest.json"
temp = target.with_suffix(".json.tmp")
temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
temp.replace(target)
