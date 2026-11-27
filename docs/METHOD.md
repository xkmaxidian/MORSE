# Method and source map

Let O denote measured expression-bearing locations and U new candidate locations.
Input expression has shape N × G, registered coordinates N × 2 and image features
N × 6. The six features are per-channel RGB means and population SD, concatenated
as `[mean_R, mean_G, mean_B, sd_R, sd_G, sd_B]`. Intensity is in [0,1] before
measured-spot z-scoring. Image borders are edge-padded for a complete odd patch.

## Network

Every MLP has Linear → GELU → LayerNorm → Linear:

| Module | Dimensions | Role |
| --- | --- | --- |
| E | G → 256 → 2 | Expression-to-coordinate encoder |
| D | 2 → 256 → G | Shared expression decoder |
| f | 14 → 128 → 1, then sigmoid | Edge affinity from h_i, h_j, y_i-y_j |
| phi | 12 → 128 → 2, then bounded tanh displacement | Intrinsic-coordinate deformation |

The last phi layer starts at zero. Its input concatenates the local six-feature
vector with the topology-weighted six-feature neighbor aggregate. Intrinsic
coordinates are `u_i = y_i + delta * tanh(phi([h_i, aggregated_h_i]))`.
Each directed KNN row normalizes predicted edge weights by their sum plus epsilon.

## Training on O

`training.fit_morse` standardizes expression, coordinates and morphology using O.
It builds the physical KNN graph on O and calculates fixed molecular and image
targets `A_G = exp(-||g_i-g_j||²/tau_G²)` and
`A_H = exp(-||h_i-h_j||²/tau_H²)`. Their shared target is `(A_G + A_H)/2`.
The geometric target for an edge is `||y_i-y_j|| * (2 - A_target)`.

The five terms in `losses.morse_objective` are:

1. Latent-coordinate alignment: mean spot-wise L1 norm between E(g) and y,
   plus weighted Monte Carlo sliced 2-Wasserstein distance.
2. Expression reconstruction: D(E(g)) compared with g.
3. Topology supervision: edge-weight MSE against the fixed shared target.
4. Morphology-guided expression reconstruction: D(u) compared with g.
5. Geometry: SmoothL1 between intrinsic edge distance and the geometric target.

For both expression losses, squared L2 and weighted L1 vector norms are summed
over genes and then averaged over spots. Topology/geometry losses average over
edges. One Bernoulli mask is sampled per spot per iteration and shared across its
six morphology features; molecular/image targets are built before this masking.
All networks are updated jointly with full-batch Adam.

## Generation on O union U

The new locations require coordinates and their H&E descriptors, not expression.
Reuse O-fitted scalers and build a KNN graph on O ∪ U. Predict its edge affinities
with f, aggregate morphology, obtain u with phi, and decode using D(u). Molecular
targets are not recalculated for U. Reverse the fitted expression z-score to
return normalized log1p expression. Both O and U rows are predictions; the
original measured expression is stored in a separate output file.

## Where to inspect each step

| Step | Implementation |
| --- | --- |
| Input row alignment and image scale | `data.read_pdac`, `data.read_visium` |
| Counts, gene filter and variance ranking | `preprocessing.prepare_expression` |
| Image patch descriptors | `image_features.rgb_features` |
| Four networks and coordinate deformation | `model.MORSE` |
| Five losses | `losses.morse_objective` |
| Graph construction and fitting | `training.graph_edges`, `training.fit_morse` |
| New coordinates and support filters | `locations.candidate_locations` |
| Joint O ∪ U inference | `training.predict_expression` |
| Output row identity and units | `output.make_prediction_adata` |
| End-to-end order | `pipeline.run_pipeline` |

These two workflows are main-model generation examples, not repeated-split
accuracy experiments. A reconstructed map is not an independent measurement,
and newly generated spots are not additional biological replicates.
