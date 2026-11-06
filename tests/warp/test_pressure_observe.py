"""Observe replay must preserve numerical thresholds and hard model checks."""
import copy
from tools.warp_tokamak_cae import physical_parameters
from tools.warp_pressure_observe import observe_only


def test_observe_policy_keeps_every_existing_threshold():
    p=physical_parameters(1,1e16,.1,1.,.4)
    for b in p.budgets.values():b.hard=True
    before=copy.deepcopy(p.budgets)
    observe_only(p)
    for key,b in p.budgets.items():
        old=before[key]
        assert (b.atol,b.rtol,b.reference,b.norm,b.unit)==(old.atol,old.rtol,old.reference,old.norm,old.unit)
    assert not p.budgets['electron_pressure_adjoint.electron.value'].hard
    assert p.budgets['wall_current.linf'].hard
