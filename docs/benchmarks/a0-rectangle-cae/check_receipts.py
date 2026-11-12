"""Summarize saved physical gates without changing or truncating raw logs."""
from pathlib import Path
import argparse
from collections import Counter
import json

WATCHED=(
    'initial_divB','ct_divergence','ct_update','divB_update','wall_current',
    'wall_ion_flux','wall_electron_flux','pressure_pairing','electric_pairing',
    'field_energy_B_interval','field_exchange_interval','total_energy_interval',
    'time_step_estimate',
)


def collect(case):
    result={'counts':Counter(), 'worst':{}}
    with (case/'run/checks.jsonl').open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            if b'"gate": "hard"' in line:
                status=line.split(b'"status": "',1)[1].split(b'"',1)[0].decode()
                result['counts']['hard_'+status]+=1
            if not any(line.startswith(('"name": "'+name+'"').encode(),1)
                       for name in WATCHED):
                continue
            record=json.loads(line)
            name=record['name']
            if record['raw'] is None:
                continue
            old=result['worst'].get(name)
            if old is None or abs(old['raw'])<abs(record['raw']):
                result['worst'][name]={k:record[k] for k in ('raw','unit','threshold','step','status','gate')}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    args.output.write_text(json.dumps(collect(args.case),indent=2)+'\n')
