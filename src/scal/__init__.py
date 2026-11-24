"""SCAL algorithms for sparse polynomial-chaos surrogate modelling."""

from .config import RunConfig
from .datasets import DatasetRepository
from .evaluation import BenchmarkEvaluator
from .experiment import Experiment, ExperimentResult
from .external import ExternalSimulator, SimulationError
from .functions import FunctionConfig, FunctionExperiment, InputSpace
from .models import PolynomialSurrogate
from .reporting import save_report
from .solving.HCBP_solver import HCBPSolver
from .sampling.ahsc import AHSCSampler
from .sampling.theta import ThetaSampler
from .sampling.s_optimal import SOptimalSampler

__version__ = "1.0.0"
__all__ = [
    "RunConfig",
    "DatasetRepository",
    "Experiment",
    "ExperimentResult",
    "ExternalSimulator",
    "SimulationError",
    "FunctionConfig",
    "FunctionExperiment",
    "InputSpace",
    "BenchmarkEvaluator",
    "PolynomialSurrogate",
    "save_report",
    "HCBPSolver",
    "AHSCSampler",
    "ThetaSampler",
    "SOptimalSampler",
]
