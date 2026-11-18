"""Single cell ATAC: peak or tile counts next to the RNA, QC, TF-IDF and LSI.

Adds a chromatin modality to a `SingleCellsMultiModal`. The counts live in a
second store under `atac/`, written for every cell of the RNA store in the
same order, so the RNA cell filter applies to both. From the LSI on, the
regular single cell functions take `modality="atac"` and
`embd_to_use="lsi"`; `bv.mm.generate_wnn_graph` fuses it with the RNA.

```python
mm = bv.SingleCellsMultiModal(sc.dir)
bv.atac.add_atac_tenx(mm, "filtered_feature_bc_matrix.h5")
bv.atac.atac_fragment_qc(mm, "atac_fragments.tsv.gz", "genes.gtf.gz")
bv.atac.run_tfidf(mm)
bv.atac.find_top_features(mm, min_cutoff="q5")
bv.atac.calculate_lsi(mm, n_comp=30)
bv.mm.generate_wnn_graph(
    mm, modality_2="atac", embd_to_use_2="lsi"
)
```

**Peaks or fragments.** `add_atac_tenx` takes the peak matrix Cell Ranger ARC
wrote into the h5. `add_atac_fragments` counts the fragment file itself into
a BED of peaks or fixed-width tiles; its default `mode="insertion"` counts
each Tn5 insertion as Cell Ranger does, and reproduces the h5 matrix on
Cell Ranger's peaks.

**Gene activity.** `gene_activity` counts fragments over gene body plus
promoter (Signac's `GeneActivity`) into its own `SingleCells`, the query side
of an RNA to ATAC label transfer. It is not RNA: methods that assume an mRNA
count model (residuals, doublets, Bonsai, gene set proportions) refuse it, and
those biased by gene length or by fragments shared between neighbouring genes
(AUCell, ScType, NMF, Hotspot gene correlations, SCENIC, CellPhoneDB) warn.

**Gotchas.** `run_tfidf` computes the IDF over the kept cells: run it again,
then the LSI, after changing them. `calculate_lsi` drops component 1 by
default, as the Signac vignettes do; `get_lsi_depth_cor` shows whether that
is right for the data.
"""

from bixverse._core.atac import (
    add_atac_fragments,
    add_atac_tenx,
    atac_fragment_qc,
    calculate_lsi,
    find_top_features,
    gene_activity,
    get_lsi_depth_cor,
    run_tfidf,
)
from bixverse._core.multimodal import AtacInfo

__all__ = [
    "AtacInfo",
    "add_atac_fragments",
    "add_atac_tenx",
    "atac_fragment_qc",
    "calculate_lsi",
    "find_top_features",
    "gene_activity",
    "get_lsi_depth_cor",
    "run_tfidf",
]
