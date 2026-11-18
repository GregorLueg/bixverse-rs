# Installing bixverse (Python) and its ecosystem

## The package itself

PyPI ships pre-built wheels. No Rust toolchain, no compile:

```bash
uv pip install bixverse      # into the active environment
uv add bixverse              # or as a dependency of a uv project
```

Python 3.11 or newer. The wheel is built against the stable ABI
(`abi3-py311`), so one wheel serves every Python version from 3.11 up.

The required dependencies are `numpy`, `polars`, `duckdb`, `beartype` and
`tqdm`. pandas, AnnData, scanpy and h5py are not dependencies. The h5ad and 10x
h5 readers are Rust, and the wheel carries HDF5 compiled in (`hdf5-static`), so
no system HDF5 is needed. On macOS the wheel links Apple Accelerate for dense
GEMM; Accelerate ships with macOS.

PyPI holds wheels only, no sdist. Which platforms have a wheel is set by a
shared release workflow outside this repository (unverified).

## Optional extras

The core suite works without any extra. Each extra unlocks one module or one
method, and a missing one raises an `ImportError` that names the extra:

| Extra | Pulls in | Needed for |
|---|---|---|
| `embed` | `manifolds-rs>=0.6.0` | `import bixverse.embed`: UMAP, densMAP, t-SNE, den-SNE, PHATE, ForceAtlas2. The import itself raises without it |
| `plots` | `matplotlib>=3.8` | the `bv.pl` plots. Imported on first use of a plot; the `bv.pl.extract_*` frames need nothing |
| `sparse` | `scipy>=1.9` | `.scipy_csr()` on `Counts`, `CsrMatrix` and loader `Batch`es. Nothing else |
| `gpu` | `bixverse-gpu` | `import bixverse_gpu`, the GPU twins |
| `torch` | `torch>=2.4` | `bixverse.nn` with `backend="torch"` (the default) |
| `mlx` | `mlx>=0.25`, Apple Silicon only | `bixverse.nn` with `backend="mlx"` |
| `jax` | `jax>=0.5`, `optax>=0.2` | `bixverse.nn` with `backend="jax"`. CUDA needs `jax[cuda12]` on top |

Extras combine. The quickstart needs `"bixverse[embed,plots,sparse]"`.

The `mlx` extra carries a platform marker. On anything but macOS arm64 it
installs nothing and does not fail, so `--all-extras` stays installable
everywhere.

## From source

Needed for a development checkout, an unreleased commit or a platform without
a wheel. Requirements:

- Rust via rustup. The crates use edition 2024, so rustc 1.85 or newer.
- maturin `>=1.15,<2.0`, pulled in by the `dev` group.
- CMake and a C compiler. `hdf5-static` builds the bundled HDF5 from source
  through the `hdf5-metno-src` crate, which drives CMake.

```bash
git clone https://github.com/GregorLueg/bixverse-rs
cd bixverse-rs/python
uv sync --group dev
uv run maturin develop --release
```

`python/` is its own cargo workspace and reaches the crates through path
dependencies (`../crates/...`), so the build needs the whole repository, not
the `python/` folder alone. Always build with `--release`; a debug build runs
the numerics unoptimised. The first build compiles the full Rust tree and HDF5
and takes several minutes. That is normal, not a hang.

A direct git install goes through the same maturin build. As a uv project
dependency, pin the branch and let `uv.lock` record the commit:

```bash
uv pip install "bixverse @ git+https://github.com/GregorLueg/bixverse-rs#subdirectory=python"
```

```toml
[tool.uv.sources]
bixverse = { git = "https://github.com/GregorLueg/bixverse-rs", branch = "main", subdirectory = "python" }
```

Python extras cannot switch Rust features, so every wheel carries all of them:
multi-modal, Bonsai, the edgeR chain (`dge`), `hdf5-static` and `parquet`. One
HDF5 feature is not compiled in: `hdf5-filters`. h5ad files written with the
LZF or Blosc codecs therefore do not load; gzip-compressed files do.

## The GPU wheel

`bixverse-gpu` is a separate wheel on wgpu, imported as `bixverse_gpu`.
Install it through the extra; add `embed` for the CPU embeddings it lacks:

```bash
uv pip install "bixverse[gpu,embed]"
```

It shares the version line and release tag with `bixverse`, so keep the two
on the same version. Its modules mirror the CPU ones: `pp`, `graph`, `embed`,
`integrate`, `de`, `doublets`, `metacells`, `nmf`, `pathways` and `trees`.
They take the same objects, write the same cache slots and obs columns and
return the same result types, so a pipeline swaps one import for the other:

```python
import bixverse as bv
import bixverse_gpu as bvg

bv.pp.find_hvg(sc, hvg_no=2000)
bvg.pp.calculate_pca(sc, 30)
bvg.graph.find_neighbours(sc)
```

Two exceptions: GPU Harmony and fastMNN write `harmony_gpu` and `mnn_gpu`
instead of `harmony` and `mnn`. GPU-specific knobs live in
`bixverse_gpu.params`; everything else takes the `bixverse.params` bundles.
There is no GPU den-SNE or ForceAtlas2. `bvg.gpu_available()` probes for an
adapter; without one every twin raises `bixverse.BixverseError`.

From source, inside the checkout:

```bash
cd bixverse-rs/python/gpu
uv run --no-sync maturin develop --release
```

## bixverse.nn backends

`import bixverse.nn` needs no framework. The framework is imported when
`train_scvi(..., backend=...)` runs, and a missing one raises
`ImportError: the <backend> backend needs pip install bixverse[<backend>]`.

| `backend` | Extra | `device` |
|---|---|---|
| `"torch"` | `torch` | `"auto"` picks cuda, then mps, then cpu |
| `"mlx"` | `mlx` | ignored; MLX runs on the Apple GPU |
| `"jax"` | `jax` | a platform name (`"cpu"`, `"gpu"`), `"auto"` takes JAX's default |

On Apple Silicon, MLX is the fastest backend; without a GPU, JAX on CPU is
almost twice as fast as torch on CPU (`docs/scvi.md`). JAX on the Apple GPU (`jax-metal`) is
not supported. On Linux, a plain torch install resolves to the multi-GB CUDA
wheels; CI installs CPU wheels with uv's `--torch-backend cpu`.

## Related packages

| Package | What it is | Reach for it when |
|---|---|---|
| [manifolds-rs](https://pypi.org/project/manifolds-rs/) | the 2D embeddings behind `bv.embed` and `bvg.embed` | always with `embed`. `bv.embed.*_defaults()` return its estimators |
| [bixverse-rs](https://crates.io/crates/bixverse-rs) | the Rust crates underneath | Rust code. `bv.__core_version__` names the crate version a wheel vendored |

## Checking an install

Versions and optional modules, no computation:

```python
import importlib.util

import bixverse as bv

print(bv.__version__, bv.__core_version__)
for mod in ("manifolds_rs", "matplotlib", "scipy", "bixverse_gpu", "torch", "mlx", "jax"):
    print(f"{mod:14} {importlib.util.find_spec(mod) is not None}")
```

A smoke test on synthetic data, no network. If it prints a `SingleCells` with
`hvg=True, pca=True` and `knn=True`, the extension compiled and linked:

```python
import tempfile
from pathlib import Path

import bixverse as bv
from bixverse.params import SyntheticDataParams

sim = bv.io.generate_single_cell_test_data(
    SyntheticDataParams(n_cells=3000, n_genes=2000)
)
tmp = tempfile.TemporaryDirectory()
sc = bv.io.load_csr(Path(tmp.name) / "demo", sim.counts, sim.obs, sim.var)
bv.pp.find_hvg(sc, hvg_no=500)
bv.pp.calculate_pca(sc, no_pcs=20)
bv.graph.find_neighbours(sc)
print(sc)
```

## Keeping this skill current

The skill ships in the wheel and is versioned with the package. Install it
through the project's environment, so it matches the bixverse version the
project uses:

```bash
uv run bixverse skill install --project              # ./.claude/skills
uv run bixverse skill install --project --agent codex  # ./.agents/skills
uv run bixverse skill install                        # ~/.claude/skills
uv run bixverse skill status                         # installs and versions
```

The Python equivalent is
`bv.install_agent_skill(agent="claude", scope="project", overwrite=True)`.

`SKILL.md` follows the Agent Skills open standard, so Claude Code, Codex and
other agents that scan `.agents/skills` discover it without further setup.
Each install is stamped with the package version (`.bixverse-version`);
`import bixverse` warns when an installed copy is older than the package, and
`--force` (`overwrite=True`) updates it. A symlink to a development checkout is
never flagged, and `--force` replaces the symlink without touching the
checkout.

## Common failures

| Symptom | Fix |
|---|---|
| `ImportError: bixverse.embed needs manifolds-rs` | install the `embed` extra |
| `ImportError: The bixverse.pl plots need matplotlib` | install the `plots` extra |
| `ModuleNotFoundError: scipy` from `.scipy_csr()` | install the `sparse` extra |
| `ImportError: the mlx backend needs ...` on Linux or Intel macOS | MLX is Apple Silicon only; use `backend="torch"` or `"jax"` |
| `BixverseError` from every `bvg.*` call | no GPU adapter; check `bvg.gpu_available()` |
| source build fails in `hdf5-metno-src` | install CMake and a C compiler (unverified error text) |
| source build fails on edition 2024 | `rustup update`; rustc 1.85 or newer |
| h5ad does not load, file uses LZF or Blosc | rewrite it with gzip compression, or export the counts as MTX |
| warning that the store uses the v3 format | `bv.io.upgrade_sc_store(sc)`; v3 still works, v4 is smaller and faster |
