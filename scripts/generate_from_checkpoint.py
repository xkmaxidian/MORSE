"""Generate at user-supplied x/y locations using a previously saved MORSE run."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import anndata as ad
import numpy as np
import pandas as pd
from morse.config import read_config
from morse.data import load_dataset
from morse.image_features import rgb_features
from morse.output import load_checkpoint
from morse.training import predict_expression


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--locations", type=Path, required=True, help="CSV with x,y in registered coordinate units")
    parser.add_argument("--output", type=Path, required=True, help="New .h5ad file for query predictions only")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    config = read_config(args.run_dir / "config.json")
    data = load_dataset(config, ROOT)
    model, state, genes = load_checkpoint(args.run_dir / "morse_model.pt", args.device)
    query = pd.read_csv(args.locations)[["x", "y"]].to_numpy(np.float32)
    h = rgb_features(data.image, query * data.image_scale, config["image"]["patch_size"])
    all_prediction, all_intrinsic = predict_expression(model, state, query, h)
    n = len(state["observed_coordinates"])
    obs = pd.DataFrame(index=[f"query_{i:06d}" for i in range(len(query))])
    result = ad.AnnData(all_prediction[n:], obs=obs, var=pd.DataFrame(index=genes))
    result.obsm["spatial"] = query
    result.obsm["X_morse_intrinsic"] = all_intrinsic[n:]
    result.uns["expression_units"] = "normalized log1p; predictions, not measured counts"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.write_h5ad(args.output)
    print(f"Saved {len(query)} query predictions to {args.output}")


if __name__ == "__main__":
    main()
