# Source lineage and organization

## Original model source

The main architecture derives from the full MORSE branch in the existing project:

```text
STAGE-label/STAGE/morse_ablation.py
SHA-256: d0ce4e06ff7d726298297ccf72f2257d46d5e5421ed77ecc52105879d19f05b2
```

That file mixes the full model, ablation switches, anchors, splitting and metrics.
This package separates the main model into readable modules and includes only
the two complete generation workflows requested for public distribution.
It does not use the older `model_graph.py` / MNMST STAGE workflow as MORSE.

## Retained structure

- E, D, f and phi correspond to `_MORSE.E`, `_MORSE.D`, `_MORSE.f` and `_MORSE.phi`.
- MLP layer order, widths, two-dimensional latent coordinates and zero-initialized
  final deformation layer are retained.
- Shared molecular/morphology affinity targets, whole-row feature masking,
  five-objective training, bounded deformation and Adam are retained.
- Candidate support rules retain the project's PDAC convex-hull/distance procedure.
  Visium uses its native row/column lattice basis to avoid treating a hexagonal
  array as an axis-aligned square grid.

## Explicit changes

The following are substantive implementation changes, not merely renamed files:

| Item | Historical core | Organized manuscript implementation |
| --- | --- | --- |
| Latent objective | Element-wise L1 only | Spot-wise L1 vector norm plus sliced Wasserstein term stated in Methods |
| Affinity | exp(-distance² / (2 tau²)) | exp(-distance² / tau²), matching manuscript equations |
| Expression-loss reduction | Mean over all entries | Sum over genes, mean over spots, matching the stated vector norms |
| Neighbor-weight denominator | Sum clamped at 1e-6 | Sum plus explicit epsilon, matching the stated normalized weights |
| Input organization | Preconstructed matrices and external morphology | Explicit raw-count readers and six RGB patch descriptors |
| Generation | Mixed with held-out benchmark routine | Separate joint graph on observed plus new locations |
| Experimental branches | Ablations, residual anchors, baseline inputs and metrics | Removed from the distributed main workflow |

The original core did not set SWD hyperparameters. This version explicitly sets
weight 0.1 and 100 projections. They are configuration defaults, not recovered
evidence of a historical setting. The reorganized code is therefore **not a
byte-identical archive or a verified reproduction of all previously reported
numerical results**. Changes to the above terms can affect fitted results.

## Traceability

Input file paths and fingerprints are in `data/manifest.json`.
`MANIFEST.sha256` fingerprints every distributed file other than the checksum
file itself. The original MIT notice is preserved in `LICENSE`.
Historical ablation scripts and execution outputs are intentionally not included.

## Checks performed during packaging

Source and notebook code were parsed statically; JSON, local import targets,
input paths and package contents were checked. No model import, training,
inference, environment installation or notebook execution was performed for this
release. Runtime correctness and numerical equivalence have not been established
by these packaging checks.
