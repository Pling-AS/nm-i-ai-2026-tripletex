"""Aggregate observations and compute Dirichlet posteriors.

Supports save/load to disk for resubmission without re-querying.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from features import CellArchetype
from utils import NUM_CLASSES, TERRAIN_TO_CLASS


class ObservationStore:
    """Stores per-cell terrain observations across all seeds and queries.

    Each simulation query yields a viewport of terrain codes.  We convert
    these to prediction class indices and accumulate counts.
    """

    def __init__(self, seeds_count: int, height: int, width: int) -> None:
        self.seeds_count = seeds_count
        self.height = height
        self.width = width

        # Per-cell observation counts: (seed, y, x) -> counts[6]
        self._counts = np.zeros(
            (seeds_count, height, width, NUM_CLASSES), dtype=np.int32
        )
        # How many times each cell has been observed
        self._obs_count = np.zeros((seeds_count, height, width), dtype=np.int32)

        # Archetype-level pooled counts: archetype -> counts[6]
        self._archetype_counts: dict[CellArchetype, NDArray[np.int32]] = defaultdict(
            lambda: np.zeros(NUM_CLASSES, dtype=np.int32)
        )
        self._archetype_obs_count: dict[CellArchetype, int] = defaultdict(int)

    def add_observation(
        self,
        seed_index: int,
        viewport: dict[str, int],
        grid: list[list[int]],
        archetypes: NDArray[np.object_],
    ) -> None:
        """Record one simulation observation.

        Parameters
        ----------
        seed_index : which seed was observed.
        viewport : dict with keys x, y, w, h.
        grid : viewport_h x viewport_w grid of internal terrain codes.
        archetypes : full (H, W) archetype array for this seed.
        """
        vx, vy = viewport["x"], viewport["y"]
        vh, vw = len(grid), len(grid[0]) if grid else 0

        for ry in range(vh):
            for rx in range(vw):
                ay = vy + ry  # absolute y
                ax = vx + rx  # absolute x
                if ay >= self.height or ax >= self.width:
                    continue

                terrain_code = grid[ry][rx]
                class_idx = TERRAIN_TO_CLASS.get(terrain_code, 0)

                self._counts[seed_index, ay, ax, class_idx] += 1
                self._obs_count[seed_index, ay, ax] += 1

                # Pool into archetype
                archetype = archetypes[ay, ax]
                self._archetype_counts[archetype][class_idx] += 1
                self._archetype_obs_count[archetype] += 1

    def get_cell_counts(self, seed_index: int, y: int, x: int) -> NDArray[np.int32]:
        """Return (6,) observation counts for a specific cell."""
        return self._counts[seed_index, y, x]

    def get_cell_obs_count(self, seed_index: int, y: int, x: int) -> int:
        """How many times has this cell been observed?"""
        return int(self._obs_count[seed_index, y, x])

    def get_archetype_counts(self, archetype: CellArchetype) -> NDArray[np.int32]:
        """Return pooled (6,) observation counts for an archetype (across all seeds)."""
        return self._archetype_counts[archetype]

    def get_archetype_obs_count(self, archetype: CellArchetype) -> int:
        """How many total observations exist for this archetype?"""
        return self._archetype_obs_count[archetype]

    def get_seed_counts(self, seed_index: int) -> NDArray[np.int32]:
        """Return (H, W, 6) counts array for one seed."""
        return self._counts[seed_index]

    def get_seed_obs_counts(self, seed_index: int) -> NDArray[np.int32]:
        """Return (H, W) observation count array for one seed."""
        return self._obs_count[seed_index]

    def compute_posterior_entropy(
        self,
        seed_index: int,
        y: int,
        x: int,
        alpha: float = 0.5,
    ) -> float:
        """Compute entropy of the Dirichlet posterior mean for a cell.

        Higher entropy = more uncertain = higher priority for repeat queries.
        """
        counts = self._counts[seed_index, y, x].astype(np.float64)
        n_obs = self._obs_count[seed_index, y, x]

        if n_obs == 0:
            # Maximum uncertainty — uniform over 6 classes
            return np.log(NUM_CLASSES)

        # Dirichlet posterior mean: (count + alpha) / (N + K*alpha)
        posterior = (counts + alpha) / (n_obs + NUM_CLASSES * alpha)
        # Entropy
        # Avoid log(0) by filtering zeros
        nonzero = posterior > 0
        entropy = -np.sum(posterior[nonzero] * np.log(posterior[nonzero]))
        return float(entropy)

    def compute_entropy_grid(
        self,
        seed_index: int,
        alpha: float = 0.5,
    ) -> NDArray[np.floating]:
        """Compute posterior entropy for every cell in a seed. Returns (H, W)."""
        counts = self._counts[seed_index].astype(np.float64)
        n_obs = self._obs_count[seed_index].astype(np.float64)

        # Dirichlet posterior mean
        posterior = (counts + alpha) / (n_obs[..., np.newaxis] + NUM_CLASSES * alpha)
        # Where n_obs == 0, posterior is uniform → entropy = log(6)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_post = np.where(posterior > 0, np.log(posterior), 0.0)
        entropy = -np.sum(posterior * log_post, axis=-1)
        return entropy

    # ------------------------------------------------------------------
    # Persistence — save/load observation data to disk
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save observation data (counts + archetype pools) to .npz file."""
        path = Path(path)

        # Serialize archetype counts/obs_counts to JSON-compatible format
        arch_data = {}
        for arch, counts in self._archetype_counts.items():
            key = str(arch)
            arch_data[key] = {
                "counts": counts.tolist(),
                "obs_count": self._archetype_obs_count[arch],
            }

        # Save numpy arrays + archetype JSON
        np.savez_compressed(
            path.with_suffix(".npz"),
            counts=self._counts,
            obs_count=self._obs_count,
        )

        # Save archetype data separately as JSON (NamedTuple keys aren't numpy-friendly)
        json_path = path.with_suffix(".json")
        meta = {
            "seeds_count": self.seeds_count,
            "height": self.height,
            "width": self.width,
            "archetype_pools": arch_data,
        }
        json_path.write_text(json.dumps(meta, indent=2))
        print(f"[obs_store] Saved to {path.with_suffix('.npz')} + {json_path}")

    @staticmethod
    def _parse_archetype_key(key: str) -> CellArchetype:
        """Parse CellArchetype string representation safely."""
        # Clean up legacy fields if present
        if "has_adjacent_ruin" in key:
            key = key.replace(", has_adjacent_ruin=True", "").replace(
                ", has_adjacent_ruin=False", ""
            )

        # Regex to extract fields
        # pattern: initial_terrain=(\d+), is_coastal=(True|False), dist_settlement_bucket=(\d+), has_adjacent_settlement=(True|False)
        m = re.search(
            r"initial_terrain=(\d+).*is_coastal=(True|False).*dist_settlement_bucket=(\d+).*has_adjacent_settlement=(True|False)",
            key,
        )
        if not m:
            # Fallback for empty/malformed keys (should not happen in valid saves)
            raise ValueError(f"Could not parse archetype key: {key}")

        return CellArchetype(
            initial_terrain=int(m.group(1)),
            is_coastal=m.group(2) == "True",
            dist_settlement_bucket=int(m.group(3)),
            has_adjacent_settlement=m.group(4) == "True",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ObservationStore":
        """Load observation data from disk."""
        path = Path(path)
        npz_path = path.with_suffix(".npz")
        json_path = path.with_suffix(".json")

        # Load metadata + archetype pools
        meta = json.loads(json_path.read_text())
        store = cls(meta["seeds_count"], meta["height"], meta["width"])

        # Load numpy arrays
        data = np.load(npz_path)
        store._counts = data["counts"]
        store._obs_count = data["obs_count"]

        for key_str, pool in meta["archetype_pools"].items():
            arch = cls._parse_archetype_key(key_str)
            store._archetype_counts[arch] += np.array(pool["counts"], dtype=np.int32)
            store._archetype_obs_count[arch] += pool["obs_count"]

        n_cells_observed = int((store._obs_count > 0).sum())
        n_archetypes = len(store._archetype_counts)
        print(
            f"[obs_store] Loaded: {n_cells_observed} cells observed, "
            f"{n_archetypes} archetype pools"
        )
        return store
