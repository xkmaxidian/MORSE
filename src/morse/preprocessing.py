"""Count filtering, total-count normalization and variance-based gene selection."""
from dataclasses import dataclass
import anndata as ad
import numpy as np
from scipy import sparse


@dataclass
class PreparedData:
    observed: ad.AnnData
    metadata: dict

    @property
    def expression(self):
        return np.asarray(self.observed.X, dtype=np.float32)

    @property
    def coordinates(self):
        return np.asarray(self.observed.obsm["spatial"], dtype=np.float32)


def prepare_expression(counts, settings):
    """Fit preprocessing using the supplied expression-bearing spots only.

    These demonstration workflows train on all supplied measured tissue
    spots. They do not implement train/hold-out benchmarks. For an external
    hold-out analysis, subset the measured spots BEFORE calling this function.
    HVGs here means largest variance after total-count normalization/log1p,
    matching the source MORSE variance selector; it is not Seurat v3 HVG.
    """
    minimum = int(settings["min_detected_spots"])
    top = int(settings["n_top_genes"])
    library_size = float(settings["library_size"])
    if minimum < 1 or top < 1 or library_size <= 0:
        raise ValueError("Invalid count preprocessing settings.")
    raw = sparse.csr_matrix(counts.X, dtype=np.float32)
    raw.sum_duplicates()
    if not np.isfinite(raw.data).all() or (raw.data < 0).any():
        raise ValueError("Expected finite, nonnegative raw counts.")
    if not np.allclose(raw.data, np.rint(raw.data), atol=1e-5, rtol=0):
        raise ValueError("Input is not integer-like counts; do not normalize a log-normalized matrix again.")
    totals = np.asarray(raw.sum(axis=1)).ravel()
    keep_spots = totals > 0
    raw, totals = raw[keep_spots], totals[keep_spots]
    if raw.shape[0] < 3:
        raise ValueError("Fewer than three nonempty measured spots.")
    detected = np.asarray((raw > 0).sum(axis=0)).ravel()
    eligible = np.flatnonzero(detected >= minimum)
    if not len(eligible):
        raise ValueError("No genes passed the measured-spot detection filter.")
    # Library totals are calculated across all input genes, BEFORE filtering.
    normalized = raw.multiply((library_size / totals)[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    eligible_matrix = normalized[:, eligible]
    mean = np.asarray(eligible_matrix.mean(axis=0)).ravel()
    second = np.asarray(eligible_matrix.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - mean * mean, 0)
    ranked = np.argsort(-variance, kind="stable")[:min(top, len(eligible))]
    selected = np.sort(eligible[ranked])
    observed = counts[keep_spots, selected].copy()
    observed.X = normalized[:, selected].toarray().astype(np.float32)
    observed.obs["total_counts_before_gene_filter"] = totals
    observed.var["detected_spots"] = detected[selected]
    observed.var["highly_variable"] = True
    observed.var["normalized_variance"] = variance[np.searchsorted(eligible, selected)]
    metadata = {
        "input_spots": int(counts.n_obs), "input_genes": int(counts.n_vars),
        "training_spots": int(observed.n_obs), "eligible_genes": int(len(eligible)),
        "selected_genes": int(observed.n_vars), "library_size": library_size,
        "min_detected_spots": minimum,
        "gene_selection": "largest variance of total-count-normalized log1p expression",
        "expression_units": "log1p(count / spot_library_total * library_size)",
    }
    return PreparedData(observed, metadata)
