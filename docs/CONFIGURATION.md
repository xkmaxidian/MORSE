# Configuration reference

Both JSON files contain the same full-model settings; only inputs, candidate
geometry and the output directory differ. There are no implicit dataset-specific
parameters in the command-line wrappers.

| Section / key | Default | Meaning |
| --- | --- | --- |
| `input` | Dataset-relative paths | Measured counts, registration and image assets |
| `preprocessing.min_detected_spots` | 5 | Gene detected in at least this many supplied measured spots |
| `preprocessing.library_size` | 10,000 | Fixed total-count normalization target, before log1p |
| `preprocessing.n_top_genes` | 3,000 | Largest normalized-expression variances among eligible genes |
| `image.patch_size` | 15 | Odd patch width/height in the loaded working/hires image |
| `generation.mode` | PDAC: `grid`; Visium: `lattice` | Grid candidates or native hexagonal half-step candidates |
| `generation.density_factor` | 4 | Requested total density, measured plus new locations |
| `model.epochs` | 5,000 | Full-batch optimization steps |
| `model.learning_rate` | 0.001 | Adam learning rate |
| `model.hidden_dim` | 256 | E/D hidden width |
| `model.topology_hidden_dim` | 128 | f/phi hidden width |
| `model.neighbors` | 6 | Directed nearest neighbors in standardized coordinate space |
| `model.displacement_bound` | 1.0 | Per-coordinate bound on deformation in standardized units |
| `model.morphology_mask_ratio` | 0.2 | Probability of masking an entire six-feature row during training |
| `model.tau_gene`, `model.tau_morphology` | 1.0, 1.0 | Bandwidths in exp(-squared distance / tau²) |
| `model.lambda_*` | All 1.0 | Outer weights on the five objectives |
| `model.l1_weight` | 0.1 | L1 coefficient in the two expression losses |
| `model.wasserstein_weight` | 0.1 | SWD coefficient within the latent-coordinate objective |
| `model.swd_projections` | 100 | Random 1D projections per sliced-Wasserstein estimate |
| `model.epsilon` | 1e-8 | Topology-weight normalization denominator offset |
| `model.seed` | 1234 | Initialization, masking, projections and candidate subsampling |
| `model.device` | `cpu` | PyTorch device; `cuda:0` is an optional override |
| `output_dir` | `outputs/<dataset>` | Saved run, created only when the workflow is executed |

Expression, physical coordinates and image features are independently z-scored
using the measured training spots. Population SD is floored at 1e-6. These fitted
scalers are reused at generation time. The training graph contains measured spots
only; the inference graph is rebuilt on measured and candidate spots together.
Gradient norms are clipped to 5; Adam uses PyTorch defaults other than the stated
learning rate. No scheduler, validation-based early stopping or minibatch sampler
is active. Random seeds do not guarantee identical floating-point results on all
hardware or PyTorch versions.

The manuscript does not give numerical SWD projection/coefficient settings.
`wasserstein_weight = 0.1` and `swd_projections = 100` are explicit defaults for
this organized implementation, not retrospectively verified historical run values.
See `SOURCE.md` for the differences from the historical core.

## Candidate support

PDAC: grid spacing is d/sqrt(factor), hence d/2 for factor 4, where d is the median
measured nearest-neighbor distance. Visium: fit its native array row/column basis
to registered physical coordinates and apply three half-step translations.
Both modes deduplicate candidates, retain the measured convex hull, exclude
near-duplicates at r <= 0.12d and reject distances r > 1.35d to the nearest
measured spot. If too many survive, spatial farthest-point selection caps new
locations at round((factor - 1)N). No rejected candidate is reintroduced to force
an exact density. Visium's three-offset mode currently implements factor 4 only.

This rule constrains coordinate support; it does not explicitly segment tissue
holes or concave boundaries. The achieved count and density are written into
`run_manifest.json`. Query locations supplied by the user are not automatically
re-sampled or tissue-masked.
