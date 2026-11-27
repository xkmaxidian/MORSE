"""Candidate proposals and coordinate-support filtering before image extraction."""
import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors


def median_spacing(coordinates):
    if len(coordinates) < 3:
        raise ValueError("At least three measured locations are needed.")
    distances = NearestNeighbors(n_neighbors=2).fit(coordinates).kneighbors(coordinates)[0][:, 1]
    positive = distances[distances > 0]
    if not len(positive):
        raise ValueError("Measured coordinates have no positive spacing.")
    return float(np.median(positive))


def evenly_spaced_subset(points, count, seed):
    """Cap a candidate pool by deterministic-seeded farthest-point selection."""
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(points)))]
    distance = ((points - points[chosen[0]]) ** 2).sum(axis=1)
    for _ in range(1, count):
        index = int(np.argmax(distance))
        chosen.append(index)
        distance = np.minimum(distance, ((points - points[index]) ** 2).sum(axis=1))
    return points[np.asarray(chosen)]


def candidate_locations(coordinates, obs, mode="grid", density_factor=4.0, seed=1234):
    """Propose new coordinates; density_factor counts measured AND new rows.

    No expression or domain label is used here. Coordinate filtering does
    not guarantee removal of every image-derived hole or concave boundary.
    """
    xy = np.asarray(coordinates, dtype=np.float64)
    if density_factor <= 1:
        raise ValueError("Density factor must exceed one.")
    d = median_spacing(xy)
    if mode == "lattice":
        if density_factor != 4:
            raise ValueError("Three half-step translations implement fourfold densification.")
        if not {"array_col", "array_row"}.issubset(obs.columns):
            raise KeyError("Visium lattice generation needs array_col and array_row.")
        design = np.c_[obs["array_col"].to_numpy(float), obs["array_row"].to_numpy(float), np.ones(len(xy))]
        if np.linalg.matrix_rank(design) < 3:
            raise ValueError("Array coordinates do not span a two-dimensional lattice.")
        transform = np.linalg.lstsq(design, xy, rcond=None)[0]
        # Native Visium in-row step = two columns; diagonal = one column/row.
        b1 = 2 * transform[0]
        b2 = transform[0] + transform[1]
        candidates = np.concatenate((xy + 0.5 * b1, xy + 0.5 * b2,
                                     xy + 0.5 * (b1 + b2)), axis=0)
    elif mode == "grid":
        step = d / np.sqrt(density_factor)
        lower, upper = xy.min(axis=0) - step, xy.max(axis=0) + step
        gx, gy = np.meshgrid(np.arange(lower[0], upper[0] + step, step),
                             np.arange(lower[1], upper[1] + step, step))
        candidates = np.c_[gx.ravel(), gy.ravel()]
    else:
        raise ValueError("Candidate mode must be grid or lattice.")

    tolerance = max(d * 1e-6, 1e-7)
    _, unique = np.unique(np.rint(candidates / tolerance).astype(np.int64), axis=0,
                          return_index=True)
    candidates = candidates[np.sort(unique)]
    inside = Delaunay(xy).find_simplex(candidates) >= 0
    distance = NearestNeighbors(n_neighbors=1).fit(xy).kneighbors(candidates)[0][:, 0]
    candidates = candidates[inside & (distance > 0.12 * d) & (distance <= 1.35 * d)]
    requested = max(1, int(round((density_factor - 1) * len(xy))))
    candidates = evenly_spaced_subset(candidates, requested, seed)
    if not len(candidates):
        raise ValueError("No valid new locations after coordinate-support filtering.")
    metadata = {
        "mode": mode, "median_nearest_neighbor_spacing": d,
        "grid_spacing": d / np.sqrt(density_factor) if mode == "grid" else None,
        "requested_density_factor": density_factor,
        "actual_density_factor": (len(xy) + len(candidates)) / len(xy),
        "n_measured": len(xy), "n_new": len(candidates), "n_total": len(xy) + len(candidates),
        "filter": "Measured convex hull; 0.12d < nearest measured distance <= 1.35d; deduplicated",
        "limitation": "Coordinate filtering is not an image-derived tissue-hole mask.",
    }
    return candidates.astype(np.float32), metadata
