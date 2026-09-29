from pathlib import Path

# The project's fixed results directory (a sibling of 'src' at the repository root).
RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"

from .error_bound_evaluation import ErrorBoundEvaluationFEM
from .psi_fem import compute_psi

__all__ = ["ErrorBoundEvaluationFEM", "RESULTS_DIR", "compute_psi"]
