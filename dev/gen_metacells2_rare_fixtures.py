"""Generate MetaCells2 rare gene module parity fixtures.

Calls upstream `find_rare_gene_modules` on planted rare populations whose
counts are rebuilt from a seeded LCG on both sides, so no input data crosses
as text. Only the reference's outputs are emitted.

Run from the repo root:
  uv run --no-project --python 3.12 --with metacells==0.9.5 \
      --with "pandas<2.3" python -I dev/gen_metacells2_rare_fixtures.py \
      > crates/bixverse-sc/src/mc_generation/metacells2/parity_rare_fixtures.rs
  rustfmt --edition 2024 crates/bixverse-sc/src/mc_generation/metacells2/parity_rare_fixtures.rs
"""

import contextlib
import io
import logging

import anndata as ad
import numpy as np
import scipy.sparse as sp

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import metacells as mc

logging.disable(logging.CRITICAL)

# (name, lcg seed, cells, background genes, blocks, related, sporadic, sporadic mask, kwargs)
# A block is (gene start, gene end, cell start, cell end): the genes are
# expressed only in those cells. A related gene is (gene, cell start, cell end):
# strong in those cells and sparse background everywhere. Sporadic genes are
# uncorrelated, on when the masked LCG bits are all zero.
CASES = [
    (
        "PLANTED",
        2024,
        2000,
        200,
        [
            (200, 208, 0, 30),
            (208, 216, 27, 67),
            (216, 220, 300, 307),
            (220, 228, 500, 540),
            (228, 231, 700, 740),
        ],
        [(231, 500, 540)],
        (232, 260),
        0x7F,
        dict(max_gene_cell_fraction=0.02, target_metacell_size=48, target_pile_size=400),
    ),
    (
        "DEFAULT",
        77,
        20000,
        250,
        [
            (250, 258, 100, 114),
            (258, 266, 5000, 5020),
        ],
        [],
        (266, 300),
        0x7FF,
        dict(),
    ),
]


class Lcg:
    """Numerical Recipes LCG. Mirrored exactly on the Rust side."""

    def __init__(self, seed):
        self.x = seed

    def next(self):
        self.x = (1664525 * self.x + 1013904223) % (2**32)
        return self.x


def make_counts(seed, n_cells, n_background, blocks, related, sporadic, sporadic_mask):
    n_genes = sporadic[1]
    rng = Lcg(seed)
    counts = np.zeros((n_cells, n_genes), dtype=np.int64)
    block_of = {}
    for b, (g0, g1, _, _) in enumerate(blocks):
        for g in range(g0, g1):
            block_of[g] = b
    related_of = {g: (c0, c1) for g, c0, c1 in related}
    for i in range(n_cells):
        for g in range(n_genes):
            x = rng.next()
            hi = x >> 16
            mid = (x >> 8) & 0xFF
            if g < n_background:
                v = hi % 3 + 1 if mid % 4 == 0 else 0
            elif g in block_of:
                _, _, c0, c1 = blocks[block_of[g]]
                v = hi % 9 if c0 <= i < c1 and mid % 5 != 0 else 0
            elif g in related_of:
                c0, c1 = related_of[g]
                if c0 <= i < c1:
                    v = hi % 10 + 1 if mid % 2 == 0 else 0
                else:
                    v = 1 if mid % 64 == 0 else 0
            else:
                v = hi % 12 if (x >> 8) & sporadic_mask == 0 else 0
            counts[i, g] = v
    return counts


def rust_list(xs):
    return "&[" + ", ".join(str(int(x)) for x in xs) + "]"

def main():
    out = [
        "//! MetaCells2 rare gene module parity fixtures, generated from metacells 0.9.5.",
        "//!",
        "//! DO NOT EDIT. Regenerate with `dev/gen_metacells2_rare_fixtures.py`.",
        "//!",
        "//! Counts: Numerical Recipes LCG, see `make_counts` there and in `parity.rs`.",
        "",
    ]

    for name, seed, n_cells, n_background, blocks, related, sporadic, mask, kwargs in CASES:
        counts = make_counts(seed, n_cells, n_background, blocks, related, sporadic, mask)
        adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
        mc.tl.find_rare_gene_modules(adata, reproducible=True, **kwargs)
        gene_module = np.asarray(adata.var["rare_gene_module"].values)
        cell_module = np.asarray(adata.obs["cells_rare_gene_module"].values)
        rare_cells = np.where(cell_module >= 0)[0]
        out += [
            f"/// Case `{name}`: {n_cells} cells, upstream kwargs {kwargs}.",
            f"pub const {name}_SEED: u64 = {seed};",
            f"pub const {name}_N_CELLS: usize = {n_cells};",
            f"pub const {name}_N_BACKGROUND: usize = {n_background};",
            f"pub const {name}_BLOCKS: &[(usize, usize, usize, usize)] = &["
            + ", ".join(f"({a}, {b}, {c}, {d})" for a, b, c, d in blocks)
            + "];",
            f"pub const {name}_RELATED: &[(usize, usize, usize)] = &["
            + ", ".join(f"({a}, {b}, {c})" for a, b, c in related)
            + "];",
            f"pub const {name}_SPORADIC: (usize, usize) = ({sporadic[0]}, {sporadic[1]});",
            f"pub const {name}_SPORADIC_MASK: u64 = {mask};",
            "/// `rare_gene_module` per gene.",
            f"pub const {name}_GENE_MODULE: &[i32] = {rust_list(gene_module)};",
            "/// Cells with `cells_rare_gene_module >= 0` and their module.",
            f"pub const {name}_RARE_CELLS: &[usize] = {rust_list(rare_cells)};",
            f"pub const {name}_RARE_CELL_MODULE: &[i32] = {rust_list(cell_module[rare_cells])};",
            "",
        ]

    print("\n".join(out))


if __name__ == "__main__":
    main()
