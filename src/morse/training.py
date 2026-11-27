"""Observed-only fitting and expression generation on measured plus new spots."""
from dataclasses import asdict
import random
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from .model import MORSE
from .losses import affinity, morse_objective


def fit_scaler(values):
    """Population mean/SD; avoid division by zero for constant features."""
    mean = values.mean(axis=0, keepdims=True)
    sd = np.maximum(values.std(axis=0, keepdims=True), 1e-6)
    return mean, sd


def graph_edges(coordinates, k):
    """Directed KNN edges, excluding self indices even at tied coordinates."""
    n = len(coordinates)
    if n < 2:
        raise ValueError("At least two measured locations are required.")
    k = min(k, n - 1)
    search = NearestNeighbors(n_neighbors=k + 1).fit(coordinates)
    indices = search.kneighbors(coordinates, return_distance=False)
    neighbors = np.stack([row[row != i][:k] for i, row in enumerate(indices)])
    return np.repeat(np.arange(n), k), neighbors.ravel()


def fit_morse(expression, coordinates, morphology, config):
    """Fit the full model using expression-bearing measured locations O.

    Parameters are expression [N,G], registered coordinates [N,2], RGB
    descriptors [N,6] and MORSEConfig. Returns model, fitted state and
    epoch-wise loss history. New-location expression is not an input.
    """
    config.validate()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu.")
    device = torch.device(config.device)
    expression = np.asarray(expression, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    morphology = np.asarray(morphology, dtype=np.float32)
    n = len(expression)
    if expression.ndim != 2 or coordinates.shape != (n, 2) or morphology.shape != (n, 6):
        raise ValueError("Expected expression [N,G], coordinates [N,2], morphology [N,6].")
    if not all(np.isfinite(a).all() for a in (expression, coordinates, morphology)):
        raise ValueError("Training inputs must be finite.")

    # Fit all scalers on measured O, and retain them for unseen coordinates.
    scalers, standardized = {}, []
    for name, values in (("expression", expression), ("coordinates", coordinates),
                         ("morphology", morphology)):
        mean, sd = fit_scaler(values)
        scalers[name + "_mean"] = mean
        scalers[name + "_sd"] = sd
        standardized.append(((values - mean) / sd).astype(np.float32))
    genes, xy, h = [torch.as_tensor(a, device=device) for a in standardized]
    source_np, target_np = graph_edges(standardized[1], config.neighbors)
    source = torch.as_tensor(source_np, dtype=torch.long, device=device)
    target = torch.as_tensor(target_np, dtype=torch.long, device=device)

    # Molecular targets exist only on O. They remain fixed through training.
    gene_affinity = affinity(genes, source, target, config.tau_gene)
    image_affinity = affinity(h, source, target, config.tau_morphology)
    topology_target = (0.5 * (gene_affinity + image_affinity)).detach()
    target_distance = torch.norm(xy[source] - xy[target], dim=1) * (2 - topology_target)
    model = MORSE(expression.shape[1], config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mask_generator = torch.Generator(device=device).manual_seed(config.seed + 17)
    projection_generator = torch.Generator(device=device).manual_seed(config.seed + 29)

    history = []
    model.train()
    for epoch in range(1, config.epochs + 1):
        optimizer.zero_grad()
        total, terms, swd = morse_objective(
            model, genes, xy, h, source, target, topology_target, target_distance,
            config, mask_generator, projection_generator,
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Training loss became non-finite.")
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        row = {"epoch": epoch, "total_loss": float(total.detach().cpu()),
               "sliced_wasserstein": float(swd.detach().cpu())}
        row.update({name + "_loss": float(value.detach().cpu()) for name, value in terms.items()})
        history.append(row)
        if epoch == 1 or epoch % max(1, config.epochs // 20) == 0 or epoch == config.epochs:
            print(f"epoch {epoch}/{config.epochs}: loss={row['total_loss']:.6g}", flush=True)
    state = {
        "implementation": "MORSE-publication-main-v0.1.0",
        "config": asdict(config), "scalers": scalers,
        "observed_coordinates": coordinates, "observed_morphology": morphology,
        "n_genes": expression.shape[1],
    }
    return model, state, history


def predict_expression(model, state, query_coordinates, query_morphology):
    """Decode on O union U using only coordinates and image features at U.

    Returns (expression, intrinsic_coordinates) for all rows in O then U.
    Neither query expression nor molecular affinities on U are accepted.
    """
    query_coordinates = np.asarray(query_coordinates, dtype=np.float32)
    query_morphology = np.asarray(query_morphology, dtype=np.float32)
    if (query_coordinates.ndim != 2 or query_coordinates.shape[1] != 2
            or query_morphology.shape != (len(query_coordinates), 6)):
        raise ValueError("Expected query coordinates [M,2] and morphology [M,6].")
    if not np.isfinite(query_coordinates).all() or not np.isfinite(query_morphology).all():
        raise ValueError("Query inputs must be finite.")
    coordinates = np.concatenate((state["observed_coordinates"], query_coordinates), axis=0)
    features = np.concatenate((state["observed_morphology"], query_morphology), axis=0)
    scalers = state["scalers"]
    xy = ((coordinates - scalers["coordinates_mean"]) / scalers["coordinates_sd"]).astype(np.float32)
    h = ((features - scalers["morphology_mean"]) / scalers["morphology_sd"]).astype(np.float32)
    source, target = graph_edges(xy, model.config.neighbors)
    device = next(model.parameters()).device
    xy = torch.as_tensor(xy, device=device)
    h = torch.as_tensor(h, device=device)
    source = torch.as_tensor(source, dtype=torch.long, device=device)
    target = torch.as_tensor(target, dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        weights = model.edge_weights(h, xy, source, target)
        intrinsic = model.intrinsic_coordinates(xy, h, source, target, weights)
        prediction = model.D(intrinsic).cpu().numpy()
    prediction = prediction * scalers["expression_sd"] + scalers["expression_mean"]
    return prediction.astype(np.float32), intrinsic.cpu().numpy()
