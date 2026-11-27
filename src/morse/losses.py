"""Five MORSE objectives, with spot-wise vector-norm reductions."""
import torch
from torch.nn import functional as F


def affinity(values, source, target, bandwidth):
    """Manuscript definition: exp(-squared Euclidean distance / tau squared)."""
    distance_squared = (values[source] - values[target]).square().sum(dim=1)
    return torch.exp(-distance_squared / (bandwidth ** 2))


def sliced_wasserstein(latent, coordinates, n_projections, generator):
    """Monte Carlo sliced 2-Wasserstein distance between two equal-size sets."""
    directions = torch.randn((latent.shape[1], n_projections), device=latent.device,
                             dtype=latent.dtype, generator=generator)
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(1e-12)
    latent_projection = (latent @ directions).sort(dim=0).values
    physical_projection = (coordinates @ directions).sort(dim=0).values
    return (latent_projection - physical_projection).square().mean().clamp_min(1e-12).sqrt()


def expression_loss(predicted, measured, l1_weight):
    """Sum squared L2 and weighted L1 over genes, then average over spots."""
    difference = predicted - measured
    return (difference.square().sum(dim=1) + l1_weight * difference.abs().sum(dim=1)).mean()


def morse_objective(model, genes, coordinates, features, source, target,
                    topology_target, target_distance, config,
                    mask_generator, projection_generator):
    # One mask value per spot, shared across all six RGB descriptors.
    probability = torch.full((len(features), 1), 1 - config.morphology_mask_ratio,
                             dtype=features.dtype, device=features.device)
    keep = torch.bernoulli(probability, generator=mask_generator)
    masked = features * keep
    latent = model.E(genes)
    reconstructed = model.D(latent)
    weights = model.edge_weights(masked, coordinates, source, target)
    intrinsic = model.intrinsic_coordinates(coordinates, masked, source, target, weights)
    generated = model.D(intrinsic)
    swd = sliced_wasserstein(latent, coordinates, config.swd_projections, projection_generator)
    terms = {
        "latent": (latent - coordinates).abs().sum(dim=1).mean() + config.wasserstein_weight * swd,
        "reconstruction": expression_loss(reconstructed, genes, config.l1_weight),
        "topology": F.mse_loss(weights, topology_target),
        "morphology": expression_loss(generated, genes, config.l1_weight),
        "geometry": F.smooth_l1_loss(torch.norm(intrinsic[source] - intrinsic[target], dim=1),
                                     target_distance),
    }
    coefficients = {
        "latent": config.lambda_latent, "reconstruction": config.lambda_reconstruction,
        "topology": config.lambda_topology, "morphology": config.lambda_morphology,
        "geometry": config.lambda_geometry,
    }
    total = sum(coefficients[name] * value for name, value in terms.items())
    return total, terms, swd
