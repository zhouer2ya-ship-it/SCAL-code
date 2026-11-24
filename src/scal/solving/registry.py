"""Concrete solver factory used for every engine fit and persisted by module/class."""

from .HCBP_solver import HCBPSolver
from .LAR_solver import LARSSolver
from .LARS_solver import LarsCVSolver
from .BCS_solver import BCSSolver
from .SBL_solver import SBLSolver

SOLVERS = {
    cls.__name__: cls for cls in (HCBPSolver, LARSSolver, LarsCVSolver, BCSSolver, SBLSolver)
}


def make_solver(name, poly_matrix, *, random_state=42, **params):
    if name not in SOLVERS:
        raise ValueError(f"Unknown solver {name}; available: {sorted(SOLVERS)}")
    model = SOLVERS[name](poly_matrix, **params)
    model.random_state = int(random_state)
    model.solver_config_ = dict(
        params,
        random_state=int(random_state),
        implementation=f"{type(model).__module__}.{type(model).__name__}",
    )
    return model
