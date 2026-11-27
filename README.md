# MORSE

**Morphology-Informed Organization and Reconstruction of Spatial Expression**

This repository contains the full MORSE workflow for two human tissue datasets:
**PDAC A-1** and **Breast Cancer Visium, Block A Section 1**. Counts, registered
coordinates, H&E images, configurations and step-by-step notebooks are included.
Each workflow trains on the supplied measured tissue spots and generates
expression at additional supported locations.

| Dataset | Technology / sample | Measured tissue spots | Step-by-step notebook | Command-line entry |
| --- | --- | ---: | --- | --- |
| PDAC A-1 | Original spatial transcriptomics; GSM3036911 | 428 | [01_pdac.ipynb](notebooks/01_pdac.ipynb) | `scripts/run_pdac.py` |
| Breast cancer | 10x Visium; V1_Breast_Cancer_Block_A_Section_1 | 3,798 | [02_visium_breast.ipynb](notebooks/02_visium_breast.ipynb) | `scripts/run_visium_breast.py` |

## 1. Installation

Use Python 3.10. From the extracted `MORSE` folder:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell alternative: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

`requirements.txt` installs CPU PyTorch and the pinned scientific libraries.
A GPU is optional. CUDA installation and the recorded environment are described
in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md). No MNMST, pretrained image network,
PyTorch Geometric or external model weights are needed.

## 2. Run a dataset

```bash
python scripts/run_pdac.py
python scripts/run_visium_breast.py
```

Paths in the JSON configurations are relative to this repository; there is no
server-specific path to edit. The defaults are the full model, 3,000 selected
genes, 5,000 epochs, seed 1234 and CPU execution. To use an available CUDA device:

```bash
python scripts/run_pdac.py --device cuda:0 --output-dir outputs/pdac_cuda
```

Change the configuration files to adjust model or preprocessing settings.
`--epochs`, `--seed`, `--device` and `--output-dir` override the corresponding
settings. A nonempty output folder is never overwritten; choose a new folder
for another run. There are no automatic parameter searches or benchmark loops.

## 3. Follow the notebooks

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name morse --display-name "Python (MORSE)"
jupyter lab
```

Open either notebook and select the MORSE kernel. Both notebooks expose these
steps separately and call the same functions used by the command-line entries:

1. Read the count matrix, align spot IDs and load the registered H&E image.
2. Filter genes, normalize by library size, apply log1p and select variable genes.
3. Extract RGB means and standard deviations from 15 × 15 image patches.
4. Fit the encoder, decoder, topology predictor and bounded coordinate deformation.
5. Propose new locations and extract their H&E features.
6. Construct the inference graph over measured and new locations and decode expression.
7. Save the run; optionally display H&E registration and gene-expression maps.

## 4. Read the source

```text
MORSE/
├── configs/                 # One complete JSON configuration per dataset
├── data/
│   ├── pdac/                # Count TSV, registered locations, H&E image
│   └── visium_breast/       # 10x count HDF5 and Space Ranger spatial files
├── notebooks/              # Two stepwise main-workflow tutorials
├── scripts/                # Dataset entry points and checkpoint-based generation
├── src/morse/
│   ├── config.py           # Model settings and configuration loading
│   ├── data.py             # PDAC and 10x readers; barcode/coordinate alignment
│   ├── preprocessing.py    # Counts → normalized expression and selected genes
│   ├── image_features.py   # H&E patches → six RGB descriptors
│   ├── model.py            # E, D, topology f, deformation phi
│   ├── losses.py           # The five training objectives
│   ├── training.py         # Observed-only graph, optimization and O ∪ U inference
│   ├── locations.py        # Candidate proposals and coordinate-support filtering
│   ├── output.py           # H5AD/CSV/checkpoint serialization
│   ├── visualization.py    # Optional registration and gene maps
│   ├── pipeline.py         # The complete workflow in execution order
│   └── cli.py              # Command-line configuration overrides
├── docs/                   # Data, method, parameter, environment and source notes
└── requirements.txt
```

Start with `pipeline.py` to follow the workflow, `model.py` for the architecture,
and `losses.py` for the objective. Full parameter definitions are in
[docs/CONFIGURATION.md](docs/CONFIGURATION.md); equations and source correspondence
are in [docs/METHOD.md](docs/METHOD.md).

## 5. Files created when you run the code

Outputs are written under `outputs/pdac/` or `outputs/visium_breast/`:

| File | Contents |
| --- | --- |
| `observed_expression.h5ad` | Measured training expression after normalization/log1p and gene selection |
| `generated_expression.h5ad` | Model predictions at measured locations **and** new locations |
| `locations.csv` | Registered x/y coordinates and measured/generated indicator |
| `selected_genes.csv` | Gene names, identifiers where available, detection counts and selection metadata |
| `training_history.csv` | Total loss and each component per epoch |
| `morse_model.pt` | Model parameters, fitted scalers, observed coordinates/features and gene order |
| `config.json`, `run_manifest.json` | Effective settings, preprocessing, achieved density and library versions |

All values in `generated_expression.h5ad.X` are predictions in normalized log1p
expression units, including the rows at measured coordinates. Measured values
are saved separately. `.obs.is_measured_location` distinguishes the row types;
`.obsm['spatial']` contains registered physical coordinates and
`.obsm['X_morse_intrinsic']` contains standardized intrinsic coordinates.
Predictions are not converted into integer counts or clipped to make maps look better.

## 6. Generate at your own locations

After training, supply a CSV containing `x,y` columns in the dataset's registered
coordinate system:

```bash
python scripts/generate_from_checkpoint.py --run-dir outputs/pdac --locations my_locations.csv --output outputs/pdac_queries.h5ad
```

These coordinates must match the supplied image registration. The script needs
their coordinates and H&E features, not their expression. It saves query rows only.
The inference graph is constructed over the original measured locations plus
the supplied query set, so changing that set can change local graph neighborhoods.

## Data and implementation notes

Dataset accessions, image scales and file provenance are in
[docs/DATA.md](docs/DATA.md). The packaged PDAC image is a registered working
image from the original project, not the full-resolution GEO slide. Candidate
filtering uses coordinate support; it is not an image-derived tissue-hole mask.
Filtering can therefore reduce the achieved density below the requested fourfold
value. Both details are recorded explicitly rather than hidden in a notebook.

The full-model architecture is derived from the existing MORSE source. This
organized version also makes manuscript-formula corrections explicit; the exact
changes and original source fingerprint are listed in [docs/SOURCE.md](docs/SOURCE.md).
It is not a byte-for-byte archive of every historical experimental script.

This distribution contains input data and source only: no fitted weights,
computed results, ablation suites, competing-method benchmarks or reviewer tests.
Packaging checks were static; training, inference and notebook execution were
not performed for this release. The notebooks have no stored outputs.

## Citation and license

Please cite the MORSE manuscript and the original dataset providers described
in `docs/DATA.md`. The upstream MIT notice is retained in `LICENSE`; it does not
relicense the third-party datasets or histology images.
