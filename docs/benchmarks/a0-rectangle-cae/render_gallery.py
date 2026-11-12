"""Render the completed mode campaign using the A0 benchmark gallery."""

from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tools.plots.build_a0_report import build_report

ROOT=Path(__file__).resolve().parent


def read(file):
    return json.loads((ROOT/file).read_text())


def picture(number,title,caption,image,data,kind='mixed_comparison',**metadata):
    values=dict(number=number,title=title,caption=caption,image=image,data=data,
                source_kind=kind,**metadata)
    if kind=='mixed_comparison':
        values['components']=['accepted A0 evolution','independent cold Hall reference']
    return values


def cold_picture(cold):
    t=np.array([s['t_s'] for s in cold['samples']])
    coeff=np.asarray([s['coeff'] for s in cold['samples']])
    expected=np.asarray([s['expected'] for s in cold['samples']])
    phase=np.unwrap(np.angle(coeff[:,0]+1j*coeff[:,1]))
    oracle=np.unwrap(np.angle(expected[:,0]+1j*expected[:,1]))
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.6),layout='constrained')
    axes[0].plot(t*1e9,phase,'o-',label='Accepted A0')
    axes[0].plot(t*1e9,oracle,'k--',label='Independent analytic')
    axes[0].set(xlabel='Time (ns)',ylabel='Modal phase (rad)',title='Axial cold Hall wave')
    axes[0].legend()
    axes[1].plot(t*1e9,[s['outside_reference_span'] for s in cold['samples']],'o-')
    axes[1].set(xlabel='Time (ns)',ylabel='Energy-scaled departure',title='Outside analytic modal span')
    fig.savefig(ROOT/'cold-hall-control.png',dpi=150)
    plt.close(fig)
    return float(phase[-1]-oracle[-1])


def finish():
    comparison=read('control-comparison.json')
    narrow,half,wide=[comparison['cases'][name] for name in ('g16','g16_half','g20_wide')]
    coarse=comparison['cases']['g12']
    if any(s['case_status']!='completed' for s in (narrow,coarse,half,wide)):
        raise ValueError('Terminal accepted cases are required for the completed gallery')
    cold=read('cold-hall-short-comparison.json')
    cold_phase=cold_picture(cold)
    g16=read('inputs/narrow-g16/source.json')
    cost=read('short-r15-g16-parity.json')
    halfdiff=comparison['half_step']
    ref=comparison['refinement']
    wall=comparison['wall_distance']
    collar=comparison['wall_collar']['g16']
    wide_quiet=read('wide-quiet-cpu-drift.json')
    hall_shift=read('hall-reference-shift.json')
    hard_counts={
        name:read(path)['counts']['hard_PASS']
        for name,path in (
            ('12','analysis-r15-g12/check-receipt.json'),
            ('16','analysis-r15-g16/check-receipt.json'),
            ('quiet','raw/quiet-g16/check-receipt.json'),
            ('half','analysis-r15-g16-half/check-receipt.json'),
            ('wide','analysis-r15-wide/check-receipt.json'),
        )
    }
    fmt=lambda q: f"{q['frequency_hz']/1e6:.6f}"
    report=fr"""# A0 rectangular-wall compressional Alfvén benchmark

## Physical case and separate evidence

The synthetic equilibrium fills an axisymmetric box with a curl-free
toroidal guide field $B_\phi= B_0R_0/R$, spatially uniform density
$2\times10^{{19}}\,\mathrm{{m}}^{{-3}}$, stationary deuterium and constant
isotropic pressures. $R_0=15\,\mathrm{{m}}$ and $B_0=2\,\mathrm{{T}}$.
The solver retains $n_\phi=0,\pm1$ with five angular quadrature samples.
The positive pressures give total axis beta $10^{{-5}}$. The exterior
wall has the repaired relative E/B representation. The internal marker
in the geometry file carries no plasma boundary in this vacuum-field test.

Bradshaw's Chapter 4 develops cylindrical Hall compression branches
from the homogeneous circular theta-pinch. Chapter 5, Figure 5.3,
compares mode numbers and frequencies for a circular toroidal
cross-section. The circular examples motivate branch identification by
mode structure and polarization. Their frequencies belong to their
own geometry and equilibrium.
[Whales2 chapters 4 and 5](https://wrap.warwick.ac.uk/161400/1/WRAP_Theses_Bradshaw_2020.pdf#page=99)

The reference is a separate cold Hall Ritz solve in this same physical
box. It uses analytic cylindrical curl and volume quadrature, matching
homogeneous perturbation normal flux, zero normal current, zero normal
velocity, and the associated weak wall reaction. Polynomial order 12,
14 and 16 gives 6.162656, 6.170819 and 6.176967 MHz for the narrow box.
The last two orders give 0.998944 parallel magnetic shape overlap and
the p16 shifted-wall normal current residual is $3.19\times10^{{-12}}$
of the peak current. The wider box reference is 4.938820 MHz at p16.
The weak wall subspace and finite pressure in the full solver are stated
here so the comparison can be repeated exactly.

Scaling the Hall block alone from zero to its physical value in the
same narrow-box Ritz subspace shifts this mode from
{hall_shift['steps'][-1]['frequency_hz']/1e6:.6f} to
{hall_shift['steps'][0]['frequency_hz']/1e6:.6f} MHz
({100*hall_shift['hall_relative_frequency_shift']:+.4f}%). Successive
whole-magnetic overlaps along the continuation exceed 0.9998. This
isolates the contribution of that block in the cold reference at
fixed normal traces. The accepted A0 trajectories all use the physical
Hall term. [Hall coefficient continuation](hall-reference-shift.json)

The initial perturbations use the matched frozen cold mode of the
production spatial operators. In each run the actual frequency and
structure are fitted from accepted RK4 snapshots. The selected
narrow-16 frozen mode is 5.648499 MHz. It is an input branch selection
and a numerical check on the evolution. The independent benchmark is
the separate Ritz frequency and field. The three evidence types have
separate labels in the gallery.

## Accepted runs

| Case | Step | Accepted time (ns) | Measured cycles | Fitted frequency (MHz) | Ritz frequency (MHz) | Parallel B overlap, full / ≥0.20 m from wall |
|---|---:|---:|---:|---:|---:|---:|
| Narrow 12×12 | 195 ps | {coarse['final_saved_time_s']*1e9:.3f} | {coarse['cycles']:.3f} | {fmt(coarse)} | {coarse['reference_frequency_hz']/1e6:.6f} | {coarse['parallel_shape_overlap_full']:.4f} / {coarse['parallel_shape_overlap_wall_excluded']:.4f} |
| Narrow 16×16 | 130 ps | {narrow['final_saved_time_s']*1e9:.3f} | {narrow['cycles']:.3f} | {fmt(narrow)} | {narrow['reference_frequency_hz']/1e6:.6f} | {narrow['parallel_shape_overlap_full']:.4f} / {narrow['parallel_shape_overlap_wall_excluded']:.4f} |
| Narrow 16×16 half step | 65 ps | {half['final_saved_time_s']*1e9:.3f} | {half['cycles']:.3f} | {fmt(half)} | {half['reference_frequency_hz']/1e6:.6f} | {half['parallel_shape_overlap_full']:.4f} / {half['parallel_shape_overlap_wall_excluded']:.4f} |
| Wider 20×20 | 130 ps | {wide['final_saved_time_s']*1e9:.3f} | {wide['cycles']:.3f} | {fmt(wide)} | {wide['reference_frequency_hz']/1e6:.6f} | {wide['parallel_shape_overlap_full']:.4f} / {wide['parallel_shape_overlap_wall_excluded']:.4f} |

The narrow 16×16 fit has phase RMS {narrow['phase_rms_rad']:.3g} rad,
relative amplitude drift {narrow['relative_amplitude_drift']:.3g}, and
parallel magnetic fraction {narrow['parallel_magnetic_fraction']:.6f}.
The sampled independent field has parallel fraction
{narrow['independent_parallel_magnetic_fraction']:.6f}. The relative phase
of the ion-velocity and magnetic projections differs by
{narrow['velocity_phase_relative_to_magnetic_vs_reference_rad']:+.3g} rad.
The component-resolved polarization data are in the mode summary. Frequency moves
{100*ref['relative_frequency_shift_12_to_16']:+.3f}% on fixed-box refinement.
The two accepted modes have common-support parallel shape overlap
{ref['common_support_parallel_shape_overlap']:.4f} and interior overlap
{ref['inner_parallel_shape_overlap_away_0p2m']:.4f}. At 16×16 the
signed frequency difference from the independent solution remains
{100*narrow['reference_relative_frequency_error']:+.3f}%. The full-box
shape has a larger difference than the interior. In the 0.20 m wall
collar, the parallel magnetic shape overlap is
{collar['parallel_shape_overlap']:.4f}. This region contains
{100*collar['accepted_parallel_energy_fraction']:.1f}% of the accepted
parallel magnetic energy and
{100*collar['reference_parallel_energy_fraction']:.1f}% of the sampled
reference. The full-box velocity shape overlap is
{narrow['all_u_overlap_full']:.4f}. The branch and interior structure
have a reproducible comparison, with quantitative frequency, near-wall
magnetic shape and velocity-shape accuracy still unresolved.

Halving the step on 16×16 gives {halfdiff['matched_times']} common accepted
times through {halfdiff['final_matched_time_s']*1e9:.3f} ns. The maximum
relative magnetic perturbation gap is {halfdiff['peak_relative_delta_B']:.3g},
the maximum relative velocity perturbation gap is
{halfdiff['peak_relative_delta_u']:.3g}, and the phase gap is
{halfdiff['peak_phase_gap_rad']:.3g} rad. The same-box discrepancy is
therefore much larger than the time-step sensitivity observed here.

The physical walls move from ±0.60 m to ±0.75 m around the box center
while the cell spacing remains 0.075 m. The full RK4 fitted frequency
changes {100*wall['relative_frequency_shift_observed']:+.3f}% compared
with {100*wall['relative_frequency_shift_independent']:+.3f}% in the
independent reference. This compares the tracked compression mode
across boxes at the same spacing and time step.

The quiet 16×16 run shares the physical input and times of the
perturbation. At the last common save its peak velocity drift is
{narrow['quiet']['velocity_peak_quiet_drift']:.3g} m/s and peak magnetic
drift is {narrow['quiet']['B_peak_quiet_drift']:.3g} T. The gallery
shows absolute physical fields before the matched subtraction. The
cell-centered electric field is reconstructed through the normal
stage and H transfer from the accepted state. The absolute density is
around $2\times10^{{19}}\,\mathrm{{m}}^{{-3}}$. The difference column
subtracts the two scalar magnitudes where the row shows a magnitude.
The wider box has a separate {wide_quiet['steps']}-step CPU quiet
control through {wide_quiet['time_s']*1e9:.2f} ns. Wider and coarser
absolute plots use the initial static background.

The five full runs recorded {hard_counts['12']:,}, {hard_counts['16']:,},
{hard_counts['quiet']:,}, {hard_counts['half']:,} and {hard_counts['wide']:,}
passing hard checks, respectively, with all final endpoints accepted.
Check receipts retain measured divergence, wall current, normal flux,
pressure and electric work, energy interval and step-estimate values.

## Cold Hall short control and cost

The older axisymmetric axial cold Hall oracle supplies a separate
control. The repaired source completes 120 steps of 100 ps through
12 ns. At the endpoint its phase differs by {cold_phase:+.4g} rad from
the analytic value and the energy-scaled error is
{100*cold['final_total_error']:.2f}%. The reference frequency is
{cold['reference_frequency_MHz']:.6f} MHz. The short control is too
short to fit an independent numerical frequency with the usual
multi-period criterion.

In the matched 16×16 pilot, archived and provisional optimized A0
used identical input bytes on two RTX 5090 allocations of the same
model. Full-step medians were
{cost['archived']['median_full_advance_s']:.3f} and
{cost['provisional_optimized']['median_full_advance_s']:.3f} s.
The peak sampled CuPy reservations were
{cost['archived']['cupy_pool_reserved_bytes']/2**20:.1f} and
{cost['provisional_optimized']['cupy_pool_reserved_bytes']/2**20:.1f} MiB.
The largest scaled primary-state difference after eight steps was
{max(v['dimensionless'] for v in cost['comparison'].values()):.3g}.
This pilot and the full nonlinear mode campaign preserve separate
source hashes. The long runs use the frozen numerical scaling source
identified in their provenance.

[Gallery](index.html) · [Mode and controls data](control-comparison.json) ·
[Source and reproduction commands](README.md) ·
[Independent branch review](independent-review/REVIEW.md) ·
[Run ledger](RUN_LEDGER.md) · [Archived campaign receipt](verified-campaign.json)
"""
    report += "\n" + (ROOT/'RITZ_SPATIAL_RESOLUTION.md').read_text()
    report += """

The near-wall current differences also depend on current orientation.
In this cold linear toroidal guide field, only (J_R,J_Z) contribute
to the Lorentz and Hall coupling with the background. The mapped
A0 relative RMS of this perpendicular current in the 0.20 m collar
improves from 13.38% at 12×12 to 9.34% at 16×16, with 1.17% and
0.295% farther inside. In the independent reference alone the
perpendicular collar difference is 11.03% from order 14 to 16 and
9.76% from order 16 to 18, after common parallel-field phase and
scale alignment. The order-18 5 cm side-strip current is 97.5%
toroidal in current-squared measure. It is parallel to the guide field.
The physical lobe spacing recommendation follows the magnetic
compression scale. The derivatives and the strong-wall momentum
trace have a separate evidence limit.

[Weak cold wall and current orientation](independent-review/WALL_CURRENT_WEAK_TRACE.md)
"""

    (ROOT/'MODE_REPORT.md').write_text(report)
    manifest=dict(
        title='A0 rectangular-wall compressional Alfvén benchmark',
        introduction='Independent cold Hall reference and accepted nonlinear RK4 trajectories in the grid-aligned RZ conductor box. The final evidence includes fixed-box refinement, half-step timing, matched quiet drift and a physical wall-distance control.',
        notation=[
            {'symbol':'R, Z','meaning':'Physical poloidal cross-section coordinates in m'},
            {'symbol':'n_phi, K','meaning':'Toroidal harmonic index and retained Fourier cutoff'},
            {'symbol':'δB_parallel','meaning':'Toroidal magnetic perturbation parallel to the vacuum guide field'},
        ],
        archive={'label':'Methods, metrics and run links','path':'MODE_REPORT.md'},
        known_issues=[
            {'label':'Independent branch screening and full-domain near-wall limits','path':'independent-review/REVIEW.md'},
        ],
        pages=[
            dict(id='reference',title='Cold Hall short control',
                 introduction='A previously defined exact axial standing wave checks the repaired spatial path before the toroidal mode comparison.',
                 figures=[picture(1,'Axial wave phase and span error',
                     'The 12 ns accepted trajectory is compared to the independently given wave at each saved time.',
                     'cold-hall-control.png','cold-hall-short-comparison.json',grid={'n_r':8,'n_z':8},n_phi=0)]),
            dict(id='mode',title='Tracked n_phi=1 mode',
                 introduction='The modal phase is fitted from accepted state outputs. The independent reference and the local frozen source are recorded separately.',
                 figures=[
                     picture(2,'16×16 accepted frequency and phase',
                       f'Fitted frequency {fmt(narrow)} MHz and independent value {narrow["reference_frequency_hz"]/1e6:.6f} MHz.',
                       'analysis-r15-g16/phase-amplitude.png','analysis-r15-g16/mode-data.npz',
                       grid={'n_r':16,'n_z':16},n_phi=1,device='RTX 5090'),
                     picture(3,'Parallel magnetic field structure',
                       f'Full-box overlap {narrow["parallel_shape_overlap_full"]:.3f} and overlap {narrow["parallel_shape_overlap_wall_excluded"]:.3f} at ≥0.20 m from the wall. Separate RMS normalization makes the comparison a shape diagnostic.',
                       'analysis-r15-g16/mode-structure.png','analysis-r15-g16/mode-data.npz',
                       grid={'n_r':16,'n_z':16},n_phi=1,device='RTX 5090'),
                     picture(4,'Magnetic polarization',
                         'The magnetic components use the same phase and one parallel-field normalization. Colorbars are separate by component.',
                         'analysis-r15-g16/polarization.png','analysis-r15-g16/mode-data.npz',
                         grid={'n_r':16,'n_z':16},n_phi=1,collapsed=True),
                 ]),
            dict(id='physical-fields',title='Absolute physical fields and matched differences',
                 introduction='The first two columns show the full field in the quiet and perturbed runs at the same physical time. The right column uses the difference at that time.',
                 figures=[picture(5,'16×16 physical RZ sections',
                     'Ion density, magnetic magnitude, ion speed and electric magnitude at φ=0. The full density color scale starts at zero. Black lines denote the conducting walls.',
                     'analysis-r15-g16/total-fields.png','analysis-r15-g16/matched-physical-fields.npz',
                     kind='nonlinear_evolution',grid={'n_r':16,'n_z':16},n_phi=1),
                     picture(6,'16×16 pressures before and after subtraction',
                         'The scalar ion pressure and both electron pressure components are shown in physical units.',
                         'analysis-r15-g16/total-pressure.png','analysis-r15-g16/matched-physical-fields.npz',
                         kind='nonlinear_evolution',grid={'n_r':16,'n_z':16},n_phi=1,collapsed=True),
                     picture(7,'Wider-box 20×20 physical RZ sections',
                         'The left column uses the static equilibrium. The middle column is the accepted total field at 748.8 ns. Black lines mark the wider conducting walls.',
                         'analysis-r15-wide/total-fields.png','analysis-r15-wide/accepted-total-and-background.npz',
                         kind='nonlinear_evolution',grid={'n_r':20,'n_z':20},n_phi=1,collapsed=True),
                     picture(8,'Narrow-box 12×12 physical RZ sections',
                         'The static equilibrium and accepted total fields use a common scale. The right column subtracts the static equilibrium.',
                         'analysis-r15-g12/total-fields.png','analysis-r15-g12/accepted-total-and-background.npz',
                         kind='nonlinear_evolution',grid={'n_r':12,'n_z':12},n_phi=1,collapsed=True)]),
            dict(id='controls',title='Separate mesh, time-step and wall controls',
                 introduction='The mesh series holds the physical walls fixed. The wall series holds the cell spacing and the time step fixed.',
                 figures=[picture(9,'Separate physical and numerical sensitivities',
                     'Fixed-domain grid refinement, matched half-step phase gap and same-spacing physical wall movement.',
                     'refinement-time-wall-controls.png','control-comparison.json'),
                     picture(10,'Wider physical wall structure',
                     'The wider box uses the same 0.075 m cells as the narrow 16×16 case.',
                     'analysis-r15-wide/mode-structure.png','analysis-r15-wide/mode-data.npz',
                     grid={'n_r':20,'n_z':20},n_phi=1),
                     picture(11,'Physical structures across the three meshes',
                       'Fixed-box grids are shown beside the wider physical box. Dotted lines identify the fixed 0.20 m wall collar, and all panels share one normalized scale.',
                       'refinement-structure.png','control-comparison.json')]),
            dict(id='physical-resolution',title='Physical spatial resolution',
                 introduction='The continuous reference supplies the directional wavelengths and dominant lobe widths. The fixed-box grid and time estimates are kept separate from the wider physical box.',
                 figures=[picture(12,'Independent mode scales and cuts',
                     'The cuts mark the connected amplitude half-maximum intervals in R and Z. Grid counts and the next fixed-box measurement are recorded in the methods.',
                     'ritz-spatial-scales.png','ritz-spatial-scales.json',
                     kind='independent_reference',n_phi=1)]),
        ],
        conclusions='A0 follows the independently identified compressional branch over four measured periods and resolves the dominant interior structure. The full-box near-wall shape and absolute frequency retain measurable discrepancies at this resolution.',
    )
    (ROOT/'gallery-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    build_report(ROOT/'gallery-manifest.json',ROOT/'index.html',replace=True)


if __name__=='__main__':
    finish()
