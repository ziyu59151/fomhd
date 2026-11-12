"""Follow the rectangular cold Ritz branch while scaling its Hall term."""

import argparse
from pathlib import Path
import json

import numpy as np

from box_reference import fields, load, quadrature, solve


def main(reference_path, output):
    reference = load(reference_path)
    box = reference['box']
    r, z, weight = quadrature(box)
    b_basis = fields(box, r, z)['b'].reshape(3 * r.size, -1)
    weighted_basis = np.repeat(np.sqrt(weight), 3)[:, None] * b_basis
    index = 602
    full_hall = weighted_basis @ reference['a'][:, index]
    previous = full_hall
    rows = [dict(hall_multiplier=1.0, mode_index=index,
                 frequency_hz=float(reference['omega'][index]/(2*np.pi)),
                 overlap_previous=1.0, overlap_full_hall=1.0)]
    for multiplier in (.75, .5, .25, 0.0):
        trial = solve(box, hall_multiplier=multiplier)
        indices = np.flatnonzero((trial['norm_omega'] > .4) & (trial['compression'] > .6))
        candidates = weighted_basis @ trial['a'][:, indices]
        unit = candidates / np.linalg.norm(candidates, axis=0)
        overlaps = abs(unit.conj().T @ previous) / np.linalg.norm(previous)
        best = int(np.argmax(overlaps))
        mode = indices[best]
        previous = candidates[:, best]
        rows.append(dict(hall_multiplier=multiplier, mode_index=int(mode),
                         frequency_hz=float(trial['omega'][mode]/(2*np.pi)),
                         overlap_previous=float(overlaps[best]),
                         overlap_full_hall=float(abs(np.vdot(full_hall, unit[:, best]))
                                                 / np.linalg.norm(full_hall))))
    result = dict(
        reference='Independent constrained cold Hall Ritz, fixed normal traces and same box',
        box=box.__dict__, steps=rows,
        hall_relative_frequency_shift=rows[0]['frequency_hz']/rows[-1]['frequency_hz'] - 1,
    )
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.reference, args.output)
