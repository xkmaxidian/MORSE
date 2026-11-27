"""Read measured counts, registered coordinates and H&E without preprocessing."""
from dataclasses import dataclass
import csv
import gzip
import json
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse


@dataclass
class SpatialInput:
    counts: ad.AnnData
    image: np.ndarray
    image_scale: float
    sample_id: str


def _image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _strings(values):
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values]


def read_pdac(input_paths, project_root):
    """Read GSM3036911 counts; retain the 428 registered tissue locations.

    In this historical registration file, the column called 'x' stores
    image ROW and 'y' stores image COLUMN. We follow the original PDAC
    notebook: spatial = [pixel_x, pixel_y] = [source y, source x].
    Reference domain annotations are retained for display, never training.
    """
    root = Path(project_root)
    locations = pd.read_csv(root / input_paths["locations"], sep="\t")
    if locations.shape[1] != 5:
        raise ValueError("Expected the supplied five-column PDAC registration file.")
    locations.columns = ["source_index", "spot_id", "pixel_y", "pixel_x", "reference_domain"]
    locations["spot_id"] = locations["spot_id"].astype(str)
    locations = locations.set_index("spot_id", verify_integrity=True)
    wanted = set(locations.index)
    rows, names = [], []
    with gzip.open(root / input_paths["counts"], "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        gene_names = next(reader)[1:]
        for row in reader:
            if not row or row[0] not in wanted:
                continue
            values = np.asarray(row[1:], dtype=np.float32)
            if len(values) != len(gene_names):
                raise ValueError(f"Incorrect count-column number for {row[0]}")
            names.append(row[0])
            rows.append(sparse.csr_matrix(values.reshape(1, -1)))
    if len(set(names)) != len(names):
        raise ValueError("Duplicated spot IDs in the raw count file.")
    missing = wanted - set(names)
    if missing:
        raise ValueError(f"Registered spots absent from counts: {sorted(missing)[:5]}")
    matrix = sparse.vstack(rows, format="csr")
    order = pd.Index(names).get_indexer(locations.index)
    observed = locations[["pixel_x", "pixel_y", "reference_domain"]].copy()
    # Native array coordinates are included as metadata only.
    native = observed.index.str.split("x", expand=True)
    observed["array_x"] = native.get_level_values(0).astype(int)
    observed["array_y"] = native.get_level_values(1).astype(int)
    var = pd.DataFrame({"gene_symbol": gene_names}, index=gene_names)
    counts = ad.AnnData(matrix[order], obs=observed, var=var)
    counts.var_names_make_unique()
    counts.obsm["spatial"] = observed[["pixel_x", "pixel_y"]].to_numpy(np.float32)
    counts.uns["sample_id"] = "GSM3036911_PDACA_ST1"
    return SpatialInput(counts, _image(root / input_paths["image"]), 1.0,
                        counts.uns["sample_id"])


def read_visium(input_paths, project_root):
    """Read the standard 10x HDF5 and Space Ranger spatial files.

    The HDF5 matrix is stored as genes x barcodes in CSC form. AnnData
    receives barcodes x genes in CSR form. Barcode matching, not row
    position, aligns the matrix and spatial file.
    """
    root = Path(project_root)
    with h5py.File(root / input_paths["matrix"], "r") as handle:
        group = handle["matrix"]
        shape = tuple(group["shape"][:])
        matrix = sparse.csc_matrix((group["data"][:], group["indices"][:],
                                    group["indptr"][:]), shape=shape).T.tocsr()
        barcodes = _strings(group["barcodes"][:])
        features = group["features"]
        var = pd.DataFrame({"gene_id": _strings(features["id"][:]),
                            "gene_symbol": _strings(features["name"][:])})
        if "feature_type" in features:
            var["feature_type"] = _strings(features["feature_type"][:])
    if "feature_type" in var:
        keep = var["feature_type"].eq("Gene Expression").to_numpy()
        matrix, var = matrix[:, keep], var.loc[keep].copy()
    var.index = pd.Index(var["gene_symbol"], name="gene")
    columns = ["barcode", "in_tissue", "array_row", "array_col", "pixel_row", "pixel_col"]
    locations = pd.read_csv(root / input_paths["positions"], header=None, names=columns)
    locations = locations.set_index("barcode", verify_integrity=True)
    missing = pd.Index(barcodes).difference(locations.index)
    if len(missing):
        raise ValueError(f"Barcodes missing from the position file: {missing[:5].tolist()}")
    observed = locations.loc[barcodes].copy()
    keep = observed["in_tissue"].eq(1).to_numpy()
    observed = observed.loc[keep].copy()
    counts = ad.AnnData(matrix[keep], obs=observed, var=var)
    counts.var_names_make_unique()
    counts.obsm["spatial"] = observed[["pixel_col", "pixel_row"]].to_numpy(np.float32)
    scales = json.loads((root / input_paths["scalefactors"]).read_text(encoding="utf-8"))
    counts.uns["sample_id"] = "V1_Breast_Cancer_Block_A_Section_1"
    counts.uns["spatial_scalefactors"] = scales
    return SpatialInput(counts, _image(root / input_paths["image"]),
                        float(scales["tissue_hires_scalef"]), counts.uns["sample_id"])


def load_dataset(config, project_root):
    paths = config["input"]
    for name, relative in paths.items():
        if not (Path(project_root) / relative).is_file():
            raise FileNotFoundError(f"Missing {name}: {Path(project_root) / relative}")
    reader = read_pdac if config["dataset"] == "pdac" else read_visium
    return reader(paths, project_root)
