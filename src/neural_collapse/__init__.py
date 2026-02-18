"""
Neural Collapse analysis package.

Submodules
----------
nc_analysis : NC1–NC4 metric computation across checkpoints (ID only).
nc_ood      : NC5 (ID/OOD orthogonality) + NC1/NC2 with OOD as extra class.
neco        : NECO score (eq. 6), baselines (MSP/MaxLogit/Energy), evaluation.
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
    load_checkpoints_and_analyze_ood,
    plot_nc5_convergence,
    plot_ood_summary,
    save_ood_metrics_yaml,
)

from .neco import (
    NECOResult,
    compute_neco_scores,
    compute_baseline_scores,
    evaluate_ood_detection,
    plot_neco_distributions,
    plot_neco_pca_2d,
    plot_pca_dim_sweep,
)

from .nc_earlier_layer import (
    LayerNCResult,
    LayerNCTracker,
    analyze_layers_single_checkpoint,
    analyze_layers_across_checkpoints,
    plot_nc_by_layer,
    plot_nc_layers_across_epochs,
    plot_nc_heatmap,
    save_layer_metrics_yaml,
)

__all__ = [
    # nc_analysis (ID only — NC1–NC4)
    "NCMetricsTracker",
    "load_checkpoints_and_analyze",
    "plot_nc_evolution",
    "plot_nc_individual",
    "save_metrics_yaml",
    # nc_ood (NC5 across epochs)
    "NCOODTracker",
    "load_checkpoints_and_analyze_ood",
    "plot_nc5_convergence",
    "plot_ood_summary",
    "save_ood_metrics_yaml",
    # neco (NECO score + baselines + eval)
    "NECOResult",
    "compute_neco_scores",
    "compute_baseline_scores",
    "evaluate_ood_detection",
    "plot_neco_distributions",
    "plot_neco_pca_2d",
    "plot_pca_dim_sweep",
    # nc_earlier_layer (layer-wise NC analysis)
    "LayerNCResult",
    "LayerNCTracker",
    "analyze_layers_single_checkpoint",
    "analyze_layers_across_checkpoints",
    "plot_nc_by_layer",
    "plot_nc_layers_across_epochs",
    "plot_nc_heatmap",
    "save_layer_metrics_yaml",
]
