"""2D embedding parity with the bixverse R package.

R loads seeded synthetic data into two directories, runs its HVG, PCA and
neighbours on one and from those every 2D embedding at its defaults
(`dev/sc_embed_fixtures.R`), leaving the other untouched. Python copies the
untouched one, applies the same cell filter, takes R's PCA and kNN and runs
the same embeddings. Skipped when Rscript, bixverse, manifoldsR or the
`embed` extra is missing.

Tolerances. Both sides are deterministic for a seed and the manifolds-rs 0.5.0
wheel wraps the crate at 0.8.0, as manifoldsR does. UMAP, densMAP, PHATE and
ForceAtlas2 agree (measured max abs difference 5.1e-13, on a scale of 947) and are
compared at `atol=1e-6`.

t-SNE and den-SNE do not (uncorrelated axes, measured; manifoldsR builds from a git
branch, not the release, and the cause is not established), so they are compared on
structure:

- `pres`, the fraction of each cell's 15 nearest PCA neighbours kept among
  its 15 nearest in 2D, must agree within `PRES_TOL`.
- `shared`, the fraction of each cell's 15 nearest 2D neighbours common to
  both embeddings, must reach `MIN_SHARED`, the agreement of one implementation
  with itself under another seed (measured 0.337 and 0.303, against 0.355 and 0.317).
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import bixverse as bv

embed = pytest.importorskip("bixverse.embed", exc_type=ImportError)

SCRIPT = Path(__file__).parents[1] / "dev" / "sc_embed_fixtures.R"

K = 15
PRES_TOL = 0.03
EXACT = {"umap", "densmap", "phate", "fa2"}
MIN_SHARED = {"tsne": 0.25, "densne": 0.25}
FUNCTIONS = {
    "umap": embed.umap,
    "densmap": embed.densmap,
    "tsne": embed.tsne,
    "densne": embed.densne,
    "phate": embed.phate,
    "fa2": embed.forceatlas2,
}


def _r_available() -> bool:
    if shutil.which("Rscript") is None:
        return False
    probe = subprocess.run(
        [
            "Rscript",
            "-e",
            "quit(status = !all(sapply(c('bixverse', 'manifoldsR'), "
            "requireNamespace, quietly = TRUE)))",
        ],
        capture_output=True,
    )
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(
    not _r_available(), reason="needs Rscript, bixverse and manifoldsR"
)


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    """What R made of the embeddings."""
    root = tmp_path_factory.mktemp("r_embed")
    res = subprocess.run(
        ["Rscript", str(SCRIPT), str(root), str(root / "reference.json")],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    return json.loads((root / "reference.json").read_text())


@pytest.fixture
def sc(ref, tmp_path):
    """A fresh copy of R's untouched directory, R's cell filter, PCA and kNN."""
    directory = tmp_path / "py_embed"
    shutil.copytree(ref["base_dir"], directory)
    sc = bv.SingleCells(directory)
    sc.set_cells_to_keep(ref["keep"])
    sc.cache.pca_factors = np.array(ref["pca"], dtype=np.float32)
    sc.stamp("pca")
    sc.cache.knn = bv.KnnGraph(
        np.array(ref["knn"]["indices"], dtype=np.uint32),
        np.array(ref["knn"]["dist"], dtype=np.float32),
        ref["knn"]["metric"],
    )
    sc.stamp("knn", parents=("pca",))
    return sc


def _knn_2d(x: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :K]


def _shared(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean([len(set(x) & set(y)) / K for x, y in zip(a, b, strict=True)]))


##############
# Embeddings #
##############


@pytest.mark.parametrize("slot", list(FUNCTIONS))
def test_embedding_matches_r(sc, ref, slot):
    FUNCTIONS[slot](sc)
    got = sc.cache.embeddings[slot]
    want = np.array(ref["embeddings"][slot])
    assert got.shape == want.shape
    if slot in EXACT:
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-6)
        return
    pca = _knn_2d(sc.cache.pca_factors)
    g, w = _knn_2d(got), _knn_2d(want)
    assert abs(_shared(g, pca) - _shared(w, pca)) <= PRES_TOL
    assert _shared(g, w) >= MIN_SHARED[slot]


def test_parents_follow_r(sc):
    embed.umap(sc)
    embed.tsne(sc)
    ids = {label: s.id for label, s in sc.cache.artefacts().items() if s is not None}
    assert sc.cache.stamps["umap"].parents == (ids["pca"], ids["knn"])
    assert sc.cache.stamps["tsne"].parents == (ids["pca"],)


############
# Defaults #
############


def test_defaults_match_r(ref):
    d = ref["defaults"]
    nn = d["nn"]

    def check_nn(est):
        assert est.metric == nn["dist_metric"]
        for f, v in vars(est.nn_params).items():
            assert v == nn[f], f

    u = embed.umap_defaults()
    check_nn(u)
    top = d["umap_sc"]
    assert (u.n_components, u.n_neighbors, u.min_dist, u.spread, u.ann, u.seed) == (
        top["n_dim"],
        top["k"],
        top["min_dist"],
        top["spread"],
        top["knn_method"],
        top["seed"],
    )
    um = d["umap"]
    assert (u.learning_rate, u.optimiser, u.init, u.randomised) == (
        um["lr"],
        um["optimiser"],
        um["init"],
        um["randomised"],
    )
    # R's NULL is 500 under adam_parallel
    assert um["n_epochs"] is None
    assert u.n_epochs == 500
    for f in ("local_connectivity", "bandwidth", "mix_weight"):
        assert getattr(u.graph_params, f) == um[f], f
    for f in ("neg_sample_rate", "gamma"):
        assert getattr(u.optim_params, f) == um[f], f

    dm = embed.densmap_defaults()
    assert dm.lambda_ == d["densmap"]["lambda"]
    assert vars(dm.dens_params) == {f: d["densmap"][f] for f in ("frac", "var_shift")}
    assert {k: v for k, v in dm.get_params().items() if k in u.get_params()} == (
        u.get_params()
    )

    t = embed.tsne_defaults()
    check_nn(t)
    top, ts = d["tsne_sc"], d["tsne"]
    assert (t.n_components, t.perplexity, t.approx, t.ann, t.seed) == (
        top["n_dim"],
        top["perplexity"],
        top["approx_type"],
        top["knn_method"],
        top["seed"],
    )
    assert (t.learning_rate, t.n_epochs, t.init, t.randomised_init) == (
        ts["lr"],
        ts["n_epochs"],
        ts["init"],
        ts["randomised"],
    )
    for f, v in vars(t.optim_params).items():
        assert v == ts[f], f

    dn = embed.densne_defaults()
    assert dn.lambda_ == d["densne"]["lambda"]
    assert vars(dn.dens_params) == {f: d["densne"][f] for f in ("frac", "var_shift")}

    p = embed.phate_defaults()
    check_nn(p)
    top, ph = d["phate_sc"], d["phate"]
    assert (p.n_components, p.k, p.ann, p.seed) == (
        top["n_dim"],
        top["k"],
        top["knn_method"],
        top["seed"],
    )
    assert (p.decay, p.t, p.gamma, p.mds) == (
        ph["decay"],
        ph["t_custom"],
        ph["gamma"],
        ph["mds_method"],
    )
    # `thresh` is not in R's bundle, so it must stay at the crate default
    for f, v in vars(p.diffusion_params).items():
        assert v == ph.get(f), f

    fa = embed.forceatlas2_defaults()
    check_nn(fa)
    top, f2 = d["forceatlas2_sc"], d["fa2"]
    assert (fa.n_neighbors, fa.ann, fa.seed) == (
        top["k"],
        top["knn_method"],
        top["seed"],
    )
    assert (
        fa.n_epochs,
        fa.scaling_ratio,
        fa.gravity,
        fa.lin_log,
        fa.init,
        fa.randomised_init,
    ) == (
        f2["n_epochs"],
        f2["scaling_ratio"],
        f2["gravity"],
        f2["lin_log"],
        f2["init"],
        f2["randomised"],
    )
    for f in ("local_connectivity", "bandwidth"):
        assert getattr(fa.graph_params, f) == f2[f], f
    for f, v in vars(fa.optim_params).items():
        assert v == f2[f], f
