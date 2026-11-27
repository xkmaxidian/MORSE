"""A single main-model workflow, shared by CLI and stepwise notebooks."""
from pathlib import Path
import numpy as np
from .config import MORSEConfig
from .data import load_dataset
from .preprocessing import prepare_expression
from .image_features import rgb_features
from .training import fit_morse, predict_expression
from .locations import candidate_locations
from .output import make_prediction_adata, save_run


def run_pipeline(config, project_root, output_dir=None):
    root = Path(project_root)
    output = root / config["output_dir"] if output_dir is None else Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    print("1/6 Read measured counts, registered coordinates and H&E", flush=True)
    data = load_dataset(config, root)
    print("2/6 Filter, normalize/log1p and select genes", flush=True)
    prepared = prepare_expression(data.counts, config["preprocessing"])
    print(f"    {prepared.observed.n_obs} training locations, {prepared.observed.n_vars} selected genes")
    print("3/6 Extract RGB patch means/SD", flush=True)
    features = rgb_features(data.image, prepared.coordinates * data.image_scale,
                            config["image"]["patch_size"])
    print("4/6 Fit the full MORSE model on measured locations", flush=True)
    model, state, history = fit_morse(prepared.expression, prepared.coordinates,
                                     features, MORSEConfig(**config["model"]))
    print("5/6 Propose supported new locations and generate on O union U", flush=True)
    candidates, metadata = candidate_locations(
        prepared.coordinates, prepared.observed.obs,
        mode=config["generation"]["mode"],
        density_factor=config["generation"]["density_factor"], seed=config["model"]["seed"])
    new_features = rgb_features(data.image, candidates * data.image_scale,
                                config["image"]["patch_size"])
    prediction, intrinsic = predict_expression(model, state, candidates, new_features)
    result = make_prediction_adata(prepared, prediction, intrinsic, candidates,
                                    np.vstack((features, new_features)))
    print("6/6 Save expression, model, selected genes, locations and configuration", flush=True)
    save_run(output, config, data, prepared, model, state, history, result, metadata)
    return result
