"""Pairwise agreement between scVI embeddings, within and across arms.

Usage:
    uv run --group validation python -I dev/scvi_validation/compare.py <out_dir>

VAE latents are not identifiable, so two seeds of the same code already
disagree. Every number here is therefore read against the seed-to-seed band
of scvi-tools itself (`scvi_torch` vs `scvi_torch`):

- Procrustes disparity (scipy, after centring and scaling; 0 is identical)
- kNN overlap: mean Jaccard of each cell's 15 nearest neighbours
- ARI between Leiden labellings from metrics.R (`clusters/<name>.tsv`)

Writes `<out_dir>/pairs.tsv` and prints a summary per arm pair.
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

K = 15


def knn(x: np.ndarray) -> np.ndarray:
    """Indices of the K nearest neighbours of each row, self excluded."""
    nn = NearestNeighbors(n_neighbors=K + 1).fit(x)
    return nn.kneighbors(x, return_distance=False)[:, 1:]


def knn_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Jaccard of two neighbour lists."""
    inter = np.array([len(set(r) & set(s)) for r, s in zip(a, b, strict=True)])
    return float(np.mean(inter / (2 * K - inter)))


def main() -> None:
    out = Path(sys.argv[1])
    emb = out / "embeddings"
    names = sorted(p.stem for p in emb.glob("*_s*.tsv"))
    latents = {n: pd.read_csv(emb / f"{n}.tsv", sep="\t").to_numpy() for n in names}
    neighbours = {n: knn(x) for n, x in latents.items()}
    clusters = {
        n: pd.read_csv(emb / "clusters" / f"{n}.tsv", sep="\t")["cluster"].to_numpy()
        for n in names
        if (emb / "clusters" / f"{n}.tsv").exists()
    }

    rows = []
    for a, b in itertools.combinations(names, 2):
        arm_a, arm_b = a.rsplit("_s", 1)[0], b.rsplit("_s", 1)[0]
        _, _, disparity = procrustes(latents[a], latents[b])
        rows.append(
            {
                "pair": " vs ".join(sorted([arm_a, arm_b])),
                "a": a,
                "b": b,
                "procrustes": disparity,
                "knn_jaccard": knn_overlap(neighbours[a], neighbours[b]),
                "ari": adjusted_rand_score(clusters[a], clusters[b])
                if a in clusters and b in clusters
                else np.nan,
            }
        )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(out / "pairs.tsv", sep="\t", index=False)
    summary = pairs.groupby("pair")[["procrustes", "knn_jaccard", "ari"]].agg(
        ["mean", "min", "max"]
    )
    with pd.option_context("display.width", 200, "display.precision", 3):
        print(summary)


if __name__ == "__main__":
    main()
