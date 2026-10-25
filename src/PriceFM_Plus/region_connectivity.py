"""Fixed 40-value region inputs derived from archived 2026 border flows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SOURCE = (Path.cwd() / "Data" / "Config" /
          "flow_adjacency_2026.json")


def load_connectivity():
    graph = json.loads(SOURCE.read_text(encoding="utf-8"))
    regions = tuple(graph["regions"])
    if len(regions) != 40 or len(set(regions)) != 40:
        raise ValueError("Connectivity source must contain 40 distinct regions")
    matrix = np.asarray([graph["adjacency"][region] for region in regions],
                        dtype=np.float32)
    if (matrix.shape != (40, 40) or
            not np.isin(matrix, (0., 1.)).all() or
            not np.array_equal(matrix, matrix.T) or
            not np.all(np.diag(matrix) == 1)):
        raise ValueError("Connectivity source is not a symmetric binary 40x40 graph")
    return regions, matrix


def region_vector(region):
    """Return this region's 40-value row in the manifest's fixed column order."""
    regions, matrix = load_connectivity()
    try:
        return matrix[regions.index(region)].copy()
    except ValueError as exc:
        raise KeyError(f"Unknown active region: {region}") from exc
