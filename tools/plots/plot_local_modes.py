#!/usr/bin/env python3
"""Plot accepted local response spectra and reconstructed A0 eigenfunctions."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
NAMES = ["n_i", "mom_i_R", "mom_i_phi", "mom_i_Z", "p_i", "n_e",
         "mom_e_R", "mom_e_phi", "mom_e_Z", "p_perp_e", "p_parallel_e",
         "B_R", "B_phi", "B_Z"]
CASE_NAMES = ["warm_parallel_isotropic", "warm_perpendicular_isotropic",
              "warm_oblique_anisotropic", "cgl_firehose_stable_control",
              "cgl_firehose_unstable", "cgl_mirror_like_oblique",
              "cgl_mirror_like_stable_control"]
SHORT = ["Parallel", "Perp.", "Oblique", "Firehose\ncontrol", "Firehose",
         "Mirror-like", "Mirror-like\ncontrol"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def vector(mode, kind):
    entries = mode[kind]
    if [x["component"] for x in entries] != NAMES:
        raise ValueError("Unsupported polarization component order")
    result = np.array([complex(x["real"], x["imag"]) for x in entries])
    if not np.all(np.isfinite(result)):
        raise ValueError("Nonfinite polarization")
    return result


def validate(data):
    if data.get("passed") is not True:
        raise ValueError("An accepted dispersion report is required")
    cases = {c["case"]: c for c in data["cases"]}
    if set(cases) != set(CASE_NAMES) or len(data["cases"]) != 7:
        raise ValueError("Expected exactly the seven documented A0 cases")
    rows = []
    for name in CASE_NAMES:
        c = cases[name]
        if not c["passed"] or c["admissible_eigen_dimension"] != 11 or len(c["modes"]) != 11:
            raise ValueError("Incomplete or rejected admissible spectrum")
        scales = np.asarray(c["state_scales_SI"], dtype=float)
        if scales.shape != (14,) or not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ValueError("Invalid state scales")
        for index, m in enumerate(c["modes"]):
            eigenvalues = []
            for prefix in ("analytic", "production"):
                norm = m[prefix + "_normalized_eigenvalue"]
                eigen = complex(norm["real"], norm["imag"]) * c["reference_frequency_s^-1"]
                stored = complex(m[prefix + "_growth_s^-1"], m[prefix + "_angular_frequency_rad_s^-1"])
                if not np.isfinite(eigen) or not np.isclose(eigen, stored, rtol=1e-12, atol=0):
                    raise ValueError("SI eigenvalue does not match normalized value")
                if not np.allclose(vector(m, prefix + "_normalized_polarization") * scales,
                                   vector(m, prefix + "_SI_polarization"), rtol=1e-12, atol=0):
                    raise ValueError("SI polarization does not match state scales")
                eigenvalues.append(eigen)
            allowance = m["allowed_eigen_error_s^-1"]
            if not np.isfinite(allowance) or allowance <= 0 or abs(eigenvalues[1]-eigenvalues[0]) > allowance:
                raise ValueError("Stored matched mode violates its acceptance gate")
            rows.append([name, index, eigenvalues[0].real, eigenvalues[0].imag,
                         eigenvalues[1].real, eigenvalues[1].imag, allowance])
    return cases, rows


def save(fig, output, stem):
    fig.savefig(output / (stem + ".png"), dpi=180, facecolor="white")
    fig.savefig(output / (stem + ".pdf"), facecolor="white")
    plt.close(fig)


def spectra(cases, output):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    fig.suptitle("A0 local response: independent continuum theory vs production stages", fontsize=16)
    ax = axes[0, 0]
    for ci, name in enumerate(CASE_NAMES[:3]):
        c = cases[name]
        modes = sorted([m for m in c["modes"] if m["analytic_angular_frequency_rad_s^-1"] > 1],
                       key=lambda m: m["analytic_angular_frequency_rad_s^-1"])
        x = np.arange(1, len(modes)+1) + (ci-1)*0.15
        for prefix, marker in [("analytic", "o"), ("production", "x")]:
            y = [m[prefix + "_angular_frequency_rad_s^-1"] / (2*np.pi*1e6) for m in modes]
            ax.scatter(x, y, color=COLORS[ci], marker=marker, s=55,
                       facecolors="none" if marker == "o" else COLORS[ci],
                       label=SHORT[ci] if prefix == "production" else None)
    ax.set(yscale="log", xlabel="Frequency rank within each case", ylabel="f = Im(lambda) / (2 pi) [MHz]",
           title="(a) Warm plasma: positive nonzero frequencies", xticks=[1, 2, 3, 4])
    ax.legend(fontsize=9, loc="lower right")
    ax.text(0.03, 0.97, "Open circles: theory\nCrosses: production", transform=ax.transAxes,
            va="top", fontsize=9)
    ax = axes[0, 1]
    for ci, name in enumerate(CASE_NAMES[3:]):
        for prefix, marker in [("analytic", "o"), ("production", "x")]:
            modes = [m for m in cases[name]["modes"] if abs(m["analytic_angular_frequency_rad_s^-1"]) < 1e6]
            ax.scatter([m[prefix + "_growth_s^-1"]/1e3 for m in modes],
                       [m[prefix + "_angular_frequency_rad_s^-1"]/1e3 for m in modes],
                       color=COLORS[ci], marker=marker, s=52,
                       facecolors="none" if marker == "o" else COLORS[ci],
                       label=SHORT[ci+3].replace("\n", " ") if prefix == "production" else None)
    ax.axvline(0, color="0.65", lw=0.8)
    ax.set(xlabel="Re(lambda): growth rate [10^3 / s]", ylabel="Im(lambda) [10^3 rad/s]",
           title="(b) CGL branches: low-frequency eigenvalue plane")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0.02, 0.02, "Window: |Im(lambda)| < 10^6 /s; charge pair excluded", transform=ax.transAxes, fontsize=8)
    ax = axes[1, 0]
    for prefix, marker, offset in [("analytic", "o", -0.07), ("production", "x", 0.07)]:
        values = [cases[n][prefix + "_maximum_growth_s^-1"] for n in CASE_NAMES[3:]]
        ax.scatter(np.arange(4)+offset, values, marker=marker, s=70,
                   color=COLORS[0 if prefix == "analytic" else 1], label=prefix.capitalize(),
                   facecolors="none" if prefix == "analytic" else COLORS[1])
    ax.axhline(0.01, color="0.4", linestyle="--", label="Stable classification limit")
    ax.set_yscale("symlog", linthresh=1e-7)
    all_growth = [cases[n][p + "_maximum_growth_s^-1"] for n in CASE_NAMES[3:] for p in ("analytic", "production")]
    ax.set_ylim(min(0, min(all_growth)*2), max(all_growth)*2)
    ax.set(xticks=np.arange(4), xticklabels=SHORT[3:], ylabel="Maximum Re(lambda) [1/s]",
           title="(c) Growth and stable-control numerical noise")
    ax.legend(fontsize=8, loc="center left")
    ax = axes[1, 1]
    for key, denominator, label, marker, color in [
        ("matrix_discrepancy_normalized_max", "matrix_tolerance_normalized", "Matrix", "o", COLORS[0]),
        ("linearity_sensitivity_max", "linearity_sensitivity_tolerance", "Amplitude linearity", "s", COLORS[1]),
        ("eigen_discrepancy_normalized_max", None, "Eigenvalue", "^", COLORS[2])]:
        y = [cases[n][key] / (cases[n][denominator] if denominator else 1) for n in CASE_NAMES]
        if not all(np.isfinite(y)) or min(y) <= 0:
            raise ValueError("Expected positive finite recorded discrepancy ratios")
        ax.plot(np.arange(7), y, marker=marker, color=color, lw=1, label=label)
    ax.axhline(1, color="0.4", ls="--", label="Acceptance limit")
    ax.set(yscale="log", xticks=np.arange(7), xticklabels=SHORT,
           ylabel="Recorded discrepancy / allowed error", title="(d) Agreement relative to the original test limits")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(fontsize=8, loc="lower right")
    for ax in axes.flat:
        ax.grid(alpha=0.18)
    fig.supxlabel("Discrete cases at their recorded k and plasma parameters; this is not a continuous k scan.", fontsize=10)
    save(fig, output, "local_spectra")


def choose(c, unstable):
    if unstable:
        return max(range(11), key=lambda i: c["modes"][i]["analytic_growth_s^-1"])
    candidates = [i for i, m in enumerate(c["modes"]) if m["analytic_angular_frequency_rad_s^-1"] > 1]
    magnetic = {i: np.linalg.norm(vector(c["modes"][i], "analytic_normalized_polarization")[11:]) for i in candidates}
    threshold = 0.1 * max(magnetic.values())
    return min((i for i in candidates if magnetic[i] > threshold),
               key=lambda i: c["modes"][i]["analytic_angular_frequency_rad_s^-1"])


def mode_atlas(cases, output):
    choices = [(CASE_NAMES[0], "Parallel transverse wave", 0, False),
               (CASE_NAMES[1], "Perpendicular compression", 2, False),
               (CASE_NAMES[4], "CGL firehose", 0, True),
               (CASE_NAMES[5], "Oblique CGL mirror-like", 2, True)]
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.1), layout="constrained",
                             gridspec_kw={"height_ratios": [1.15, 1]})
    fig.suptitle("Recognizable local mode shapes from the measured A0 response", fontsize=17)
    coordinate = np.linspace(0, 2, 161)
    xx, zz = np.meshgrid(coordinate, coordinate)
    phase = np.linspace(0, 2*np.pi, 121)
    # This catches the Fourier phase sign without introducing a solver test.
    if not np.allclose(np.real(1j*np.exp(1j*phase)), -np.sin(phase), rtol=0, atol=1e-15):
        raise ValueError("Complex phase reconstruction check failed")
    records = []
    for column, (name, title, component, unstable) in enumerate(choices):
        c = cases[name]; index = choose(c, unstable); m = c["modes"][index]
        a = vector(m, "analytic_normalized_polarization")
        p = vector(m, "production_normalized_polarization")
        anchor = 11 + int(np.argmax(np.abs(a[11:])))
        phase_a = np.exp(-1j*np.angle(a[anchor])); a = a*phase_a
        phase_p = np.exp(-1j*np.angle(np.vdot(a, p))); p = p*phase_p
        scales = np.asarray(c["state_scales_SI"])
        ba, bp = (a*scales)[11:], (p*scales)[11:]
        bnorm = np.linalg.norm(ba)
        if not np.isfinite(bnorm) or bnorm <= 0:
            raise ValueError("A nonzero magnetic branch is required")
        ba, bp = ba/bnorm, bp/bnorm
        spatial_phase = 2*np.pi*(np.sin(c["theta_rad"])*xx + np.cos(c["theta_rad"])*zz)
        field = np.real(bp[component]*np.exp(1j*spatial_phase))
        ax = axes[0, column]
        im = ax.pcolormesh(xx, zz, field, cmap="RdBu_r", vmin=-1, vmax=1, shading="auto", rasterized=True)
        ax.set(aspect="equal", xlabel=r"$k_0 x / (2\pi)$", ylabel=r"$k_0 z / (2\pi)$",
               title=title + "\n" + r"$\mathrm{Re}\,\delta B_{" + ["x", "y", "z"][component] + r"}/\|\widehat{\delta B}_{\rm theory}\|$")
        ax.annotate(r"$B_0$", xy=(0.10, 0.93), xytext=(0.10, 0.72), xycoords="axes fraction",
                    color="black", ha="center", arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.7})
        origin = np.array([0.76, 0.73]); end = origin + 0.19*np.array([np.sin(c["theta_rad"]),np.cos(c["theta_rad"])])
        ax.annotate("k", xy=end, xytext=origin, xycoords="axes fraction", color="black",
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.5})
        ax = axes[1, column]
        analytic_profiles = np.real(ba[:, None]*np.exp(1j*phase))
        production_profiles = np.real(bp[:, None]*np.exp(1j*phase))
        for j, label in enumerate([r"$\delta B_x$", r"$\delta B_y$", r"$\delta B_z$"]):
            ax.plot(phase/(2*np.pi), analytic_profiles[j], color=COLORS[j], lw=1.8, label=label)
            ax.plot(phase[::8]/(2*np.pi), production_profiles[j, ::8], linestyle="none", marker="o",
                    markersize=3.7, markerfacecolor="none", color=COLORS[j])
        gamma = m["production_growth_s^-1"]; freq = m["production_ordinary_frequency_hz"]
        ax.set(xlabel=r"Wave phase $\mathbf{k}\cdot\mathbf{x}/(2\pi)$", ylabel="Common magnetic normalization",
               ylim=(-1.08, 1.08), title=f"f = {freq:.4g} Hz\ngamma = {gamma:.4g} /s")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=9, ncol=3, loc="lower center")
        records.append({"case":name,"json_mode_array_index":index,"selection":"largest analytic growth" if unstable else "lowest positive-frequency significant magnetic branch",
                        "analytic_phase_factor":[phase_a.real,phase_a.imag],"production_phase_factor":[phase_p.real,phase_p.imag],
                        "magnetic_normalization_T":float(bnorm),"phase_cycles":(phase/(2*np.pi)).tolist(),
                        "analytic_B_profiles":analytic_profiles.tolist(),"production_B_profiles":production_profiles.tolist(),
                        "production_growth_s-1":gamma,"production_frequency_hz":freq,
                        "polarization_difference_norm":float(np.linalg.norm(bp-ba)),
                        "x_phase_coordinate":coordinate.tolist(),"z_phase_coordinate":coordinate.tolist(),
                        "map_component":NAMES[11+component],"map_values":field.tolist()})
    fig.colorbar(im, ax=list(axes[0]), shrink=0.72, label="Normalized perturbation (arbitrary eigenmode amplitude)")
    fig.supxlabel("Lines: independent theory. Markers/maps: production-response eigenvectors. Local t = 0 reconstruction; not an evolved/global mode.", fontsize=10)
    save(fig, output, "local_mode_shapes")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    cases, rows = validate(data)
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size":10,"axes.titlesize":11,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    spectra(cases, args.output)
    records = mode_atlas(cases, args.output)
    with (args.output/"eigenvalues.csv").open("x", newline="") as stream:
        writer=csv.writer(stream); writer.writerow(["case","json_mode_index","theory_growth_s-1","theory_omega_rad_s-1","production_growth_s-1","production_omega_rad_s-1","allowed_eigen_error_s-1"]); writer.writerows(rows)
    (args.output/"plot-data.json").write_text(json.dumps({"modes":records},indent=2,allow_nan=False)+"\n")
    provenance={"inputs":{str(args.input.resolve()):sha(args.input)},"script_sha256":sha(__file__),
                "script_path":str(Path(__file__).resolve()),"command":sys.argv,"python":platform.python_version(),
                "numpy":np.__version__,"matplotlib":matplotlib.__version__,"scope":"Local response spectra and t=0 eigenvector reconstruction; no time evolution or global/experimental validation"}
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    (args.output/"README.md").write_text("# Local mode figures\n\nThe points compare independent continuum theory with the linear response of production A0 stages. Each case uses its recorded k and plasma parameters; no continuous dispersion scan is implied. Mode maps reconstruct measured eigenvectors at t=0, with arbitrary amplitude and a common theory/production phase. They are not time-integrated snapshots or global tokamak modes. x, y, z denote local R, phi, Z directions; B0 is along z. Stable-control positive growth below 0.01 /s is numerical noise, not an instability. No kinetic threshold is claimed. All matched eigenvalues are retained in eigenvalues.csv.\n")
    print(json.dumps({"output":str(args.output.resolve()),"figures":2,"cases":len(cases),"selected_modes":len(records)}))


if __name__ == "__main__":
    main()
