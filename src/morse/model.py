"""The full MORSE architecture: E, D, topology f and bounded deformation phi.

E/D/f/phi and the MLP layer order are retained from the original MORSE core.
Experimental variants and the optional residual-anchor decoder are excluded.
"""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden, zero_output=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden), nn.GELU(), nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_output),
        )
        if zero_output:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, values):
        return self.net(values)


class MORSE(nn.Module):
    """E: G->2; D: 2->G; f: 14->1; phi: 12->2 for six RGB features."""
    def __init__(self, n_genes, config):
        super().__init__()
        self.config = config
        self.E = MLP(n_genes, 2, config.hidden_dim)
        self.D = MLP(2, n_genes, config.hidden_dim)
        self.f = MLP(14, 1, config.topology_hidden_dim)
        self.phi = MLP(12, 2, config.topology_hidden_dim, zero_output=True)

    def edge_weights(self, features, coordinates, source, target):
        pairs = torch.cat((features[source], features[target],
                           coordinates[source] - coordinates[target]), dim=1)
        return torch.sigmoid(self.f(pairs).squeeze(1))

    def intrinsic_coordinates(self, coordinates, features, source, target, weights):
        numerator = torch.zeros_like(features)
        denominator = torch.zeros(len(features), device=features.device, dtype=features.dtype)
        numerator.index_add_(0, source, weights[:, None] * features[target])
        denominator.index_add_(0, source, weights)
        aggregated = numerator / (denominator[:, None] + self.config.epsilon)
        context = torch.cat((features, aggregated), dim=1)
        displacement = self.config.displacement_bound * torch.tanh(self.phi(context))
        return coordinates + displacement
