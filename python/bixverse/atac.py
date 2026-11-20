"""Single cell ATAC: peak or tile counts, QC, TF-IDF, LSI, links to genes.

Two kinds of experiment hold ATAC:

- a `SingleCellsMultiModal` with RNA: the ATAC counts live in a second store
  under `atac/`, written for every cell of the RNA store in the same order,
  so the RNA cell filter applies to both. `bv.mm.generate_wnn_graph` fuses
  the two.
- a `SingleCellsAtac`, ATAC only: the peaks or tiles are the experiment's
  own store. Load one with `bv.io.load_atac_fragments` or
  `bv.io.load_atac_tenx`.

Every function here takes either. From the LSI on, the regular single cell
functions take `modality="atac"` and `embd_to_use="lsi"`; on a
`SingleCellsAtac` the default modality means the ATAC store as well.

```python
# multiome
mm = bv.SingleCellsMultiModal(sc.dir)
bv.atac.add_atac_tenx(mm, "filtered_feature_bc_matrix.h5")
bv.atac.fragment_qc(mm, "atac_fragments.tsv.gz", "genes.gtf.gz")
bv.atac.run_tfidf(mm)
bv.atac.find_top_features(mm, min_cutoff="q5")
bv.atac.calculate_lsi(mm, n_comp=30)
bv.mm.generate_wnn_graph(mm, modality_2="atac", embd_to_use_2="lsi")

# ATAC only
counts = bv.atac.barcode_fragment_counts("atac_fragments.tsv.gz")
x = bv.io.load_atac_fragments(
    "exp", "atac_fragments.tsv.gz", peaks="peaks.bed", min_fragments=1000
)
```

**Peaks or fragments.** `add_atac_tenx` takes the peak matrix Cell Ranger ARC
wrote into the h5. `add_atac_fragments` counts the fragment file itself into
a BED of peaks or fixed-width tiles; its default `mode="insertion"` counts
each Tn5 insertion as Cell Ranger does, and reproduces the h5 matrix on
Cell Ranger's peaks.

**Calling peaks.** `call_peaks` calls peaks on the fragments per cluster
(`group_by`), a native port of MACS3 `callpeak` (exact coordinates and
summits against MACS3 3.1.0); Cell Ranger's peak set under-calls rare
populations. `remove_blacklisted` drops peaks on blacklist regions before
the result goes into `add_atac_fragments(peaks=...)`. `annotate_peaks`
names each feature's closest gene.

**Barcodes and files.** Fragment barcodes are the cell ids unless
`barcode_map` says otherwise (a `-1` suffix, a sample prefix).
`merge_fragments` merges per-sample fragment files into one, barcodes
prefixed per sample, as `cellranger-arc aggr` does. Every fragment pass
errors when no cell id is a barcode of the file or the fragments and the
features share no chromosome name, and warns on partial mismatches.

**Multiplets.** `amulet` flags multiplets from loci where more than two of a
cell's fragments overlap (AMULET); it needs no clustering and no RNA. Pass
repeat and blacklist BEDs as `restrict_beds`.

**Gene activity.** `gene_activity` counts fragments over gene body plus
promoter (Signac's `GeneActivity`) into its own `SingleCells`, the query side
of an RNA to ATAC label transfer; `get_gene_activity` reopens it with the
experiment's cell filter. It is not RNA: methods that assume an mRNA count
model (residuals, doublets, Bonsai, gene set proportions) refuse it, and
those biased by gene length or by fragments shared between neighbouring genes
(AUCell, ScType, `lm_activity`, NMF, Hotspot gene correlations, SCENIC,
CellPhoneDB) warn. Peaks and tiles are refused by every method that works on
genes.

**Links and profiles.** `peak_to_gene` correlates peaks with nearby genes
over groups of cells (ArchR's peak to gene links). `fragment_profiles` gives
the TSS insertion profile and the fragment length histogram per group, for
`bv.pl.tss_profile_plot` and `bv.pl.fragment_length_plot`.

**Mapping and integration.** `project_lsi` puts a query onto a reference
LSI (Seurat's `lsiproject`); `run_rlsi` integrates batches through
reciprocal LSI anchors. `bv.integrate.harmony(x, "batch", embd_to_use="lsi",
modality="atac")` corrects the LSI instead and writes `"harmony"` to the ATAC
cache.

**Gotchas.** `run_tfidf` computes the IDF over the kept cells: run it again,
then the LSI, after changing them; it also clears the ATAC cache.
`calculate_lsi` drops component 1 by default, as the Signac vignettes do;
`get_lsi_depth_cor` and `bv.pl.lsi_depth_plot` show whether that is right
for the data.
"""

from bixverse._core.atac import (
    add_atac_fragments,
    add_atac_tenx,
    amulet,
    annotate_peaks,
    barcode_fragment_counts,
    calculate_lsi,
    call_peaks,
    find_top_features,
    fragment_profiles,
    fragment_qc,
    gene_activity,
    get_gene_activity,
    get_lsi_depth_cor,
    merge_fragments,
    peak_to_gene,
    project_lsi,
    read_cellranger_barcodes,
    remove_blacklisted,
    run_rlsi,
    run_tfidf,
)
from bixverse._core.atac_exp import AtacInfo

__all__ = [
    "AtacInfo",
    "add_atac_fragments",
    "add_atac_tenx",
    "amulet",
    "annotate_peaks",
    "barcode_fragment_counts",
    "calculate_lsi",
    "call_peaks",
    "find_top_features",
    "fragment_profiles",
    "fragment_qc",
    "gene_activity",
    "get_gene_activity",
    "get_lsi_depth_cor",
    "merge_fragments",
    "peak_to_gene",
    "project_lsi",
    "read_cellranger_barcodes",
    "remove_blacklisted",
    "run_rlsi",
    "run_tfidf",
]
