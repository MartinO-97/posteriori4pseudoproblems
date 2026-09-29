from pathlib import Path
from numpy import ndarray

from . import RESULTS_DIR

# The LaTeX symbols of the error norm and the estimator for each norm_used
_NORM_SYMBOL = {"l2": "0", "h1": "1"}
_ETA_SYMBOL = {"l2": r"\eta_{L^2}", "h1": r"\eta_{H^1_0}"}


def _error_estimator_table(norm_used: str,
                           M: ndarray,
                           N: ndarray,
                           error: ndarray,
                           order: ndarray,
                           estimator: ndarray,
                           efficiency: ndarray,
                           caption: str,
                           label: str) -> list[str]:

    r""" The lines of the LaTeX table of M, N, error, its convergence
    order, the estimator and its efficiency, one row per run. """

    eff_inv = 1 / efficiency

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\begin{tabular}{|c|c|c|c|c|c|}",
        r"\hline",
        rf"$M$ & \(N\) & \(\norm{{u(T)-u^M_h}}_{{{_NORM_SYMBOL[norm_used]},\Omega}}\) & "
        rf"$\mathrm{{ord}}_N$ & ${_ETA_SYMBOL[norm_used]}$ & $\mathrm{{eff}}^{{-1}}$ \\",
        r"\hline",
    ]

    for m, n, err, ord_n, eta, eff in zip(M, N, error, order, estimator, eff_inv):
        lines.append(f"{int(m)} & {int(n)} & {err:.3e} & {ord_n:.2f} & {eta:.3e} & {eff:.2f} \\\\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    return lines


def _components_table(M: ndarray,
                      N: ndarray,
                      eta_init: ndarray,
                      eta_f: ndarray,
                      eta_R: ndarray,
                      eta_Psi: ndarray,
                      eta_delta_psi: ndarray,
                      caption: str,
                      label: str) -> list[str]:

    r""" The lines of the LaTeX table of M, N and the estimator's
    components eta_init, eta_f, eta_R, eta_Psi, eta_delta_psi, one row per
    run. """

    lines = [
        r"\begin{table}",
        r"\centering",
        r"\begin{tabular}{|c|c|c|c|c|c|c|}",
        r"\hline",
        r"$M$ & \(N\) & $\eta_{\mathrm{init}}$ & $\eta_f$ & $\eta_R$ & $\eta_\Psi$ & $\eta_{\delta\psi}$ \\",
        r"\hline",
    ]

    for row in range(M.shape[0]):
        cells = [f"{int(M[row])}", f"{int(N[row])}", f"{eta_init[row]:.3e}", f"{eta_f[row]:.3e}",
                 f"{eta_R[row]:.3e}", f"{eta_Psi[row]:.3e}", f"{eta_delta_psi[row]:.3e}"]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table}")

    return lines


def _write_to_file(norm_used: str,
                   M: ndarray,
                   N: ndarray,
                   error: ndarray,
                   order: ndarray,
                   estimator: ndarray,
                   efficiency: ndarray,
                   eta_init: ndarray,
                   eta_f: ndarray,
                   eta_R: ndarray,
                   eta_Psi: ndarray,
                   eta_delta_psi: ndarray,
                   filename: str,
                   path: str,
                   caption: str = "",
                   label: str = "",
                   components_caption: str = "",
                   components_label: str = "") -> None:

    r""" Writes a posteriori error bound evaluation results as two LaTeX
    tables to `path/filename`: M, N, the L^2 or H^1 error (`norm_used`),
    its convergence order, the estimator and its efficiency, one row per
    run, followed by a second table, in the same style, of the estimator's
    components eta_init, eta_f, eta_R, eta_Psi, eta_delta_psi, one row per
    run.

    Args:
        norm_used (str): The norm of the error and the estimator: "l2" or "h1".
        M (ndarray): Number of temporal subintervals, one entry per row.
        N (ndarray): Number of spatial subintervals, one entry per row.
        error (ndarray): The error ||u(T)-u^M_h||, one entry per row.
        order (ndarray): Convergence order of `error`, one entry per row.
        estimator (ndarray): Value of the a posteriori error estimator,
            one entry per row.
        efficiency (ndarray): `error` / `estimator`, one entry per row.
        eta_init (ndarray): Value of the eta_init component, one entry per row.
        eta_f (ndarray): Value of the eta_f component, one entry per row.
        eta_R (ndarray): Value of the eta_R component, one entry per row.
        eta_Psi (ndarray): Value of the eta_Psi component, one entry per row.
        eta_delta_psi (ndarray): Value of the eta_delta_psi component, one
            entry per row.
        filename (str): Name of the .txt file the tables are written to.
        path (str): Directory the file is written to; must be located
            inside the project's `results` folder.
        caption (str): Caption of the error/estimator table. Defaults to
            an empty string.
        label (str): Label of the error/estimator table. Defaults to an
            empty string.
        components_caption (str): Caption of the components table.
            Defaults to an empty string.
        components_label (str): Label of the components table. Defaults
            to an empty string.

    Raises:
        ValueError: If `path` is not located inside the project's
            `results` folder.
    """

    target_dir = Path(path).resolve()
    if not target_dir.is_relative_to(RESULTS_DIR):
        raise ValueError(f"path must be located inside the results folder {RESULTS_DIR}, got {target_dir}.")

    target_dir.mkdir(parents=True, exist_ok=True)

    lines = _error_estimator_table(norm_used, M, N, error, order, estimator, efficiency, caption, label)
    lines.append("")
    lines += _components_table(M, N, eta_init, eta_f, eta_R, eta_Psi, eta_delta_psi,
                               components_caption, components_label)

    (target_dir / filename).write_text("\n".join(lines) + "\n")
