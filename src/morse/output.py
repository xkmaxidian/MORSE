"""Save a user's run and reload its model for expression generation."""
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import platform
import anndata as ad
import numpy as np
import pandas as pd
import torch
from .config import MORSEConfig
from .model import MORSE


def make_prediction_adata(prepared, prediction, intrinsic, candidates, features):
    observed = prepared.observed
    n = observed.n_obs
    if prediction.shape != (n + len(candidates), observed.n_vars):
        raise ValueError("Prediction shape must follow observed rows then candidate rows.")
    ids = list(observed.obs_names) + [f"morse_new_{i:06d}" for i in range(len(candidates))]
    if len(set(ids)) != len(ids):
        raise ValueError("Generated location IDs overlap measured IDs.")
    obs = pd.DataFrame(index=pd.Index(ids, name="location_id"))
    obs["is_measured_location"] = np.arange(len(ids)) < n
    obs["location_type"] = np.where(obs["is_measured_location"], "measured", "generated")
    result = ad.AnnData(prediction, obs=obs, var=observed.var.copy())
    result.obsm["spatial"] = np.vstack((prepared.coordinates, candidates)).astype(np.float32)
    result.obsm["X_morse_intrinsic"] = intrinsic
    result.obsm["morphology_rgb"] = features
    result.uns["expression_values"] = "model predictions in normalized log1p units; not raw counts"
    result.uns["coordinate_units"] = "registered image pixels; intrinsic coordinates are standardized"
    return result


def save_checkpoint(path, model, state, gene_names):
    # Only tensors/primitives are saved, allowing weights_only=True loading.
    saved = {"implementation": state["implementation"], "config": state["config"],
             "n_genes": state["n_genes"], "gene_names": list(gene_names),
             "weights": {key: value.detach().cpu() for key, value in model.state_dict().items()},
             "scalers": {key: torch.from_numpy(value) for key, value in state["scalers"].items()},
             "observed_coordinates": torch.from_numpy(state["observed_coordinates"]),
             "observed_morphology": torch.from_numpy(state["observed_morphology"])}
    torch.save(saved, path)


def load_checkpoint(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = MORSEConfig(**{**saved["config"], "device": device})
    config.validate()
    model = MORSE(saved["n_genes"], config).to(device)
    model.load_state_dict(saved["weights"])
    model.eval()
    state = {"implementation": saved["implementation"], "config": asdict(config),
             "n_genes": saved["n_genes"],
             "scalers": {key: value.numpy() for key, value in saved["scalers"].items()},
             "observed_coordinates": saved["observed_coordinates"].numpy(),
             "observed_morphology": saved["observed_morphology"].numpy()}
    return model, state, saved["gene_names"]


def save_run(output_dir, config, data, prepared, model, state, history,
             result, generation_metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output folder is not empty; choose a new folder: {output_dir}")
    result.write_h5ad(output_dir / "generated_expression.h5ad")
    prepared.observed.write_h5ad(output_dir / "observed_expression.h5ad")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    prepared.observed.var.to_csv(output_dir / "selected_genes.csv", index_label="gene")
    coordinates = pd.DataFrame(result.obsm["spatial"], columns=["x", "y"], index=result.obs_names)
    coordinates["is_measured_location"] = result.obs["is_measured_location"].to_numpy()
    coordinates.to_csv(output_dir / "locations.csv", index_label="location_id")
    save_checkpoint(output_dir / "morse_model.pt", model, state, result.var_names)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    packages = ["torch", "numpy", "pandas", "scipy", "scikit-learn", "anndata", "h5py", "Pillow"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    manifest = {"sample_id": data.sample_id, "preprocessing": prepared.metadata,
                "generation": generation_metadata, "image_scale": data.image_scale,
                "python": platform.python_version(), "packages": versions,
                "model": asdict(model.config),
                "prediction_note": "All rows in generated_expression.h5ad are predictions, including measured locations."}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved run to {output_dir.resolve()}")
