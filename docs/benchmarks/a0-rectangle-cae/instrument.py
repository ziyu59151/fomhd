"""Measure the unmodified archived A0 solver without changing its integration."""
import json
import os
from pathlib import Path
import resource
import sys
import time

import fomhd
from fomhd import core, operators
from fomhd import __main__ as cli

target = Path(os.environ["FOMHD_RECEIPT"])
epoch = time.perf_counter()


def record(event, backend=None, **values):
    row = {
        "event": event,
        "pid": os.getpid(),
        "elapsed_s": time.perf_counter() - epoch,
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        **values,
    }
    if backend is not None and backend.name == "cuda":
        cp = backend.xp
        free, total = cp.cuda.runtime.memGetInfo()
        pool = cp.get_default_memory_pool()
        row.update(device_total_bytes=total, device_free_bytes=free,
                   cupy_pool_live_bytes=pool.used_bytes(),
                   cupy_pool_reserved_bytes=pool.total_bytes())
    with target.open("a") as f:
        f.write(json.dumps(row) + "\n")


assemble = operators.build_operators
make_context = cli._context
advance = cli.time_module.advance


def timed_assembly(geometry, k, options=None, backend=None):
    estimate = operators._estimate_operator_bytes(geometry, k)
    record("assembly_start", backend, estimate_bytes=estimate, k=k,
           device_name=(backend.xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
                        if backend and backend.name == "cuda" else None))
    started = time.perf_counter()
    try:
        result = assemble(geometry, k, options, backend)
    except Exception as error:
        record("assembly_rejected", backend, reason=str(error))
        raise
    result.compute.synchronize()
    retained = {}
    for name in ("I", "H", "QB", "T", "G", "C", "DB", "DJ", "Cdag", "Lphi", "Bc", "Bc_full"):
        value = getattr(result, name)
        values = value if isinstance(value, list) else [value]
        retained[name] = [
            {"shape": list(item.shape),
             "bytes": (item.data.nbytes + item.indices.nbytes + item.indptr.nbytes
                       if hasattr(item,"indptr") else item.nbytes)}
            for item in values if item is not None
        ]
    record("assembly_end", backend, assembly_s=time.perf_counter()-started,
           operator_arrays=retained, cells=result.Nc, edges=result.Ne, magnetic_dofs=result.Nb)
    return result


def timed_context(case):
    context = make_context(case)
    record("context", context.compute, complete_plan_bytes=core.estimated_peak_bytes(context),
           host_cap_bytes=context.p.memory_limit, device_cap_bytes=context.p.device_memory_limit)
    return context


def timed_advance(committed, cumulative, context):
    started = time.perf_counter()
    output = advance(committed, cumulative, context)
    step = committed.state.step
    if step <= 32 or step % 16 == 0:
        record("accepted_step", context.compute, step=step,
               time_s=committed.state.time, advance_s=time.perf_counter()-started)
    return output


if __name__ == "__main__":
    record("start", source=str(Path(fomhd.__file__).resolve()),
           allocation=os.environ.get("SLURM_JOB_ID"), command=sys.argv[1:])
    operators.build_operators = timed_assembly
    cli._context = timed_context
    cli.time_module.advance = timed_advance
    code = cli.main()
    record("exit", code=code)
    raise SystemExit(code)
