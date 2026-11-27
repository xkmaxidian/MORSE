# Environment

Python 3.10 is the reference interpreter. Scientific versions recorded from
the source server are pinned in `requirements-common.txt`; CPU PyTorch is pinned
separately in `requirements.txt`. This is a dependency record, not a claim that
a fresh environment was installed and exercised during packaging.

| Component | Recorded version |
| --- | --- |
| Python | 3.10.21 |
| PyTorch | 2.13.0+cpu |
| NumPy | 2.2.6 |
| pandas | 2.3.3 |
| SciPy | 1.15.3 |
| scikit-learn | 1.7.2 |
| AnnData | 0.11.4 |
| h5py | 3.16.0 |
| Pillow | 12.3.0 |
| Matplotlib | 3.10.9 |

## CPU

Follow the README virtual-environment commands, or run from project root:

```bash
conda env create -f environment.yml
conda activate morse
python -m pip install -e . --no-deps
```

## CUDA

In a clean Python 3.10 environment, install the PyTorch wheel compatible with
your installed NVIDIA driver/CUDA runtime using the official PyTorch selector.
Then install the scientific libraries without reinstalling CPU PyTorch:

```bash
python -m pip install -r requirements-common.txt
python -m pip install -e . --no-deps
python -m pip install -r requirements-notebook.txt
python scripts/run_pdac.py --device cuda:0 --output-dir outputs/pdac_cuda
```

Explicit CUDA requests fail with a clear error if CUDA is unavailable. The
workflow does not silently switch devices. CPU is the portable default; no
runtime or GPU-memory estimate is claimed without executing the workflow.

## Notebook kernel

```bash
python -m ipykernel install --user --name morse --display-name "Python (MORSE)"
jupyter lab
```

The two notebooks are deliberately unexecuted. Select the environment containing
the dependencies above and run cells from top to bottom. All dataset paths are
resolved relative to the project root, located automatically from the notebook
working directory. Each saved run records its actual library versions.
