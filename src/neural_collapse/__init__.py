"""
Neural Collapse analysis package.

Submodules
----------
nc_analysis : NC1–NC4 metric computation across checkpoints (ID only).
nc_ood      : NC5 (ID/OOD orthogonality) + NECO score for OOD detection.
"""

from .nc_analysis import (
    NCMetricsTracker,
    load_checkpoints_and_analyze,
    plot_nc_evolution,
    plot_nc_individual,
    save_metrics_yaml,
)

from .nc_ood import (
    NCOODTracker,
    NECOResult,
    load_checkpoints_and_analyze_ood,
    compute_neco_scores,
    evaluate_ood_detection,
    plot_nc5_convergence,
    plot_neco_distributions,
    plot_neco_pca_2d,
    plot_ood_summary,
    save_ood_metrics_yaml,
)

__all__ = [
    # nc_analysis (ID only)
    "NCMetricsTracker",
    "load_checkpoints_and_analyze",
    "plot_nc_evolution",
    "plot_nc_individual",
    "save_metrics_yaml",
    # nc_ood (OOD detection)
    "NCOODTracker",
    "NECOResult",
    "load_checkpoints_and_analyze_ood",
    "compute_neco_scores",
    "evaluate_ood_detection",
    "plot_nc5_convergence",
    "plot_neco_distributions",
    "plot_neco_pca_2d",
    "plot_ood_summary",
    "save_ood_metrics_yaml",
]
