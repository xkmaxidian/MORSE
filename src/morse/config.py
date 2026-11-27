"""Explicit model settings shared by the two dataset workflows."""
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class MORSEConfig:
    epochs: int = 5000
    learning_rate: float = 1e-3
    hidden_dim: int = 256
    topology_hidden_dim: int = 128
    neighbors: int = 6
    displacement_bound: float = 1.0
    morphology_mask_ratio: float = 0.2
    tau_gene: float = 1.0
    tau_morphology: float = 1.0
    lambda_latent: float = 1.0
    lambda_reconstruction: float = 1.0
    lambda_topology: float = 1.0
    lambda_morphology: float = 1.0
    lambda_geometry: float = 1.0
    l1_weight: float = 0.1
    wasserstein_weight: float = 0.1
    swd_projections: int = 100
    epsilon: float = 1e-8
    seed: int = 1234
    device: str = "cpu"

    def validate(self):
        positive_ints = (self.epochs, self.hidden_dim, self.topology_hidden_dim,
                         self.neighbors, self.swd_projections)
        if any(not isinstance(v, int) or v < 1 for v in positive_ints):
            raise ValueError("Epochs, network widths, K and SWD projections must be positive integers.")
        if min(self.learning_rate, self.tau_gene, self.tau_morphology, self.epsilon) <= 0:
            raise ValueError("Learning rate, bandwidths and epsilon must be positive.")
        if self.displacement_bound < 0 or not 0 <= self.morphology_mask_ratio < 1:
            raise ValueError("Invalid displacement bound or feature-mask probability.")
        weights = (self.lambda_latent, self.lambda_reconstruction, self.lambda_topology,
                   self.lambda_morphology, self.lambda_geometry, self.l1_weight,
                   self.wasserstein_weight)
        if min(weights) < 0:
            raise ValueError("Loss weights must be nonnegative.")


def read_config(path):
    """Read a dataset JSON configuration; paths remain relative to project root."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config["dataset"] not in {"pdac", "visium_breast"}:
        raise ValueError("dataset must be pdac or visium_breast")
    MORSEConfig(**config["model"]).validate()
    return config
