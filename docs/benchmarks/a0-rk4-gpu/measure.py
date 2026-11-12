"""Time production A0 CLI and record spatial operator storage, owned CUDA only."""

from __future__ import annotations

import argparse
import cProfile
import json
import os
from pathlib import Path
import resource
import sys
import time
from functools import wraps

from fomhd import __main__ as cli
from fomhd import io, operators, time as stepping


def measure(case_path, output, metrics, profile):
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("A0 CUDA measurements require a user Slurm allocation")
    if metrics.exists():
        raise ValueError(f"Metrics already exist at {metrics}")
    started = time.perf_counter()
    with metrics.open("x", buffering=1) as records:
        def record(event, **fields):
            data = {
                "event": event,
                "pid": os.getpid(),
                "allocation": os.environ["SLURM_JOB_ID"],
                "elapsed_s": time.perf_counter() - started,
                "host_peak_rss_B": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                **fields,
            }
            records.write(json.dumps(data, allow_nan=False) + "\n")

        def device_info(backend):
            xp = backend.xp
            free, total = xp.cuda.Device().mem_info
            pool = xp.get_default_memory_pool()
            return {
                "device_total_B": total,
                "device_free_B": free,
                "cupy_pool_live_B": pool.used_bytes(),
                "cupy_pool_reserved_B": pool.total_bytes(),
            }

        previous_assemble = operators.build_operators

        @wraps(previous_assemble)
        def assemble(geometry, K, *args, **kwargs):
            tick = time.perf_counter()
            op = previous_assemble(geometry, K, *args, **kwargs)
            op.compute.synchronize()
            members = {}
            for name, value in vars(op).items():
                parts = value if isinstance(value, (tuple, list)) else [value]
                arrays = [part for part in parts if isinstance(part, op.compute.xp.ndarray)]
                if arrays:
                    members[name] = {
                        "arrays": [list(a.shape) for a in arrays],
                        "bytes": sum(a.nbytes for a in arrays),
                        "nonzero": sum(int(op.compute.xp.count_nonzero(a).item()) for a in arrays),
                    }
            if getattr(op.QB, "format", None) == "csr":
                csr = op.QB
                members["QB"] = {
                    "arrays": [list(csr.shape)],
                    "bytes": sum(x.nbytes for x in (csr.data, csr.indices, csr.indptr)),
                    "nonzero": csr.nnz,
                    "storage": "csr",
                }
            record("operator_assembly", elapsed_assembly_s=time.perf_counter() - tick,
                   NR=int(geometry.R.size), NZ=int(geometry.Z.size),
                   K=K, Nc=op.Nc, Ne=op.Ne, Nb=op.Nb, members=members, **device_info(op.compute))
            return op

        previous_advance = stepping.advance

        @wraps(previous_advance)
        def advance(committed, cumulative, context, *args, **kwargs):
            backend = context.compute
            backend.synchronize()
            tick = time.perf_counter()
            result = previous_advance(committed, cumulative, context, *args, **kwargs)
            backend.synchronize()
            seconds = time.perf_counter() - tick
            hard = [c for c in result.checks if c.available and c.hard]
            record("step", step=committed.state.step, time_s=committed.state.time,
                   elapsed_step_s=seconds, checks=len(result.checks), hard=len(hard),
                   hard_failures=[c.name for c in hard if not c.passed], **device_info(backend))
            return result

        operators.build_operators = assemble
        stepping.advance = advance
        previous_initial = cli._initialize_once

        @wraps(previous_initial)
        def initialize(state, context, *args, **kwargs):
            tick = time.perf_counter()
            result = previous_initial(state, context, *args, **kwargs)
            context.compute.synchronize()
            xp = context.compute.xp
            device = xp.cuda.Device()
            p = xp.cuda.runtime.getDeviceProperties(device.id)
            gpu = p["name"].decode() if isinstance(p["name"], bytes) else str(p["name"])
            record("initialization", elapsed_initial_s=time.perf_counter() - tick,
                   gpu=gpu, numpy_version=__import__("numpy").__version__,
                   cupy_version=xp.__version__, warp_version=__import__("warp").__version__,
                   **device_info(context.compute))
            return result

        cli._initialize_once = initialize
        record("start", source=str(io.__file__), command=["run", str(case_path), "--output", str(output)])
        argv = ["run", str(case_path), "--output", str(output)]
        if profile:
            profiler = cProfile.Profile()
            code = profiler.runcall(cli._run_command, argv)
            profiler.dump_stats(profile)
        else:
            code = cli._run_command(argv)
        record("exit", code=code)
        return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    sys.exit(measure(args.case.resolve(), args.output.resolve(),
                     args.metrics.resolve(), args.profile))
