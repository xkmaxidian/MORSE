"""Optional input and gene-map plots. No figures are included in the release."""
import numpy as np
import matplotlib.pyplot as plt


def plot_input(data, marker_size=3):
    """Check registration: measured locations over the supplied H&E image."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(data.image)
    xy = data.counts.obsm["spatial"] * data.image_scale
    ax.scatter(xy[:, 0], xy[:, 1], s=marker_size, facecolors="none",
               edgecolors="cyan", linewidths=0.35)
    ax.set(title=data.sample_id, xlabel="Image x (pixels)", ylabel="Image y (pixels)")
    fig.tight_layout()
    return fig


def plot_expression(prepared, result, gene, marker_size=5):
    """Observed expression vs dense model predictions on a shared colour scale.

    This is a display of reconstruction, not a held-out accuracy assessment.
    The left/right panels contain different location sets and are labelled.
    """
    if gene not in prepared.observed.var_names:
        raise KeyError(f"{gene} was not selected; choose a gene in selected_genes.csv.")
    column = prepared.observed.var_names.get_loc(gene)
    truth = np.asarray(prepared.observed.X)[:, column]
    predicted = np.asarray(result.X)[:, column]
    vmin = float(min(truth.min(), predicted.min()))
    vmax = float(max(truth.max(), predicted.max()))
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharex=True, sharey=True)
    panels = [(prepared.coordinates, truth, "Measured locations"),
              (result.obsm["spatial"], predicted, "MORSE predictions: measured + new")]
    for ax, (xy, values, title) in zip(axes, panels):
        scatter = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=marker_size,
                             cmap="viridis", vmin=vmin, vmax=vmax, linewidths=0)
        ax.set_aspect("equal")
        ax.set(title=title, xlabel="Registered x (pixels)")
    axes[0].set_ylabel("Registered y (pixels)")
    axes[0].invert_yaxis()
    fig.suptitle(gene)
    fig.colorbar(scatter, ax=axes.tolist(), label="Normalized log1p expression", shrink=0.7)
    return fig
