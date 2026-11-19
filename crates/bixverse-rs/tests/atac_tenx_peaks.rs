//! The `Peaks` modality of a synthetic multiome 10x h5 into a peak store.

#![cfg(all(feature = "atac", feature = "h5"))]

use bixverse_rs::single_cell::atac::io::tenx_peaks::tenx_peaks_to_store;
use bixverse_rs::single_cell::atac::processing::lsi::{TF_IDF_SCALE, set_tf_idf_on_store};
use bixverse_rs::single_cell::sc_data::h5::tenx::{TenxFeatures, TenxVersion, write_tenx_h5};
use bixverse_rs::single_cell::sc_data::prelude::*;
use bixverse_rs::single_cell::sc_data::store::reader::ParallelSparseReader;
use bixverse_rs::single_cell::sc_data::store::v5::{FeatureKind, NormKind, cell_id_hash};
use hdf5::types::VarLenUnicode;

/// Write a multiome-shaped h5: two genes, three peaks, four cells, with
/// `features/interval` as Cell Ranger ARC writes it.
///
/// ### Params
///
/// * `path` - Where to write
fn write_multiome_h5(path: &std::path::Path) {
    // cells x features CSR; features G1, G2, P1, P2, P3
    let indptr = [0u32, 3, 5, 8, 9];
    let indices = [0u32, 2, 4, 1, 3, 2, 3, 4, 0];
    let data = [3u32, 1, 2, 1, 4, 1, 1, 1, 1];
    let counts = CompressedSparseData2::<u32>::new_csr(&data, &indices, &indptr, None, (4, 5));
    let barcodes: Vec<String> = ["AAA-1", "BBB-1", "CCC-1", "DDD-1"]
        .map(String::from)
        .to_vec();
    let ids: Vec<String> = ["G1", "G2", "chr1:100-200", "chr1:300-400", "chr2:50-80"]
        .map(String::from)
        .to_vec();
    let types: Vec<String> = [
        "Gene Expression",
        "Gene Expression",
        "Peaks",
        "Peaks",
        "Peaks",
    ]
    .map(String::from)
    .to_vec();
    write_tenx_h5(
        path,
        TenxVersion::V3,
        &counts,
        &barcodes,
        &TenxFeatures {
            ids: &ids,
            names: &ids,
            feature_types: Some(&types),
        },
    )
    .expect("10x h5 written");

    let intervals: Vec<VarLenUnicode> = [
        "chr1:10-20",
        "chr1:30-40",
        "chr1:100-200",
        "chr1:300-400",
        "chr2:50-80",
    ]
    .iter()
    .map(|s| s.parse().expect("ascii"))
    .collect();
    let file = hdf5::File::open_rw(path).expect("reopen h5");
    file.group("matrix/features")
        .expect("features group")
        .new_dataset_builder()
        .with_data(&intervals)
        .create("interval")
        .expect("interval dataset");
}

#[test]
fn test_tenx_peaks_to_store_follows_the_requested_cells() {
    let dir = tempfile::tempdir().unwrap();
    let h5 = dir.path().join("multiome.h5");
    write_multiome_h5(&h5);
    let cells = dir.path().join("cells.bin");
    let genes = dir.path().join("genes.bin");
    let (cells, genes) = (cells.to_str().unwrap(), genes.to_str().unwrap());

    let barcodes: Vec<String> = ["CCC-1", "AAA-1", "DDD-1"].map(String::from).to_vec();
    let res = tenx_peaks_to_store(&h5, cells, genes, &barcodes, None, false).expect("ingest");

    assert_eq!(
        res.peak_names,
        vec!["chr1:100-200", "chr1:300-400", "chr2:50-80"]
    );
    assert_eq!(res.lib_size, vec![3, 3, 0]);
    assert_eq!(
        res.intervals
            .iter()
            .map(|i| (i.chrom.as_str(), i.start, i.end))
            .collect::<Vec<_>>(),
        vec![("chr1", 100, 200), ("chr1", 300, 400), ("chr2", 50, 80)]
    );

    let reader = ParallelSparseReader::new(genes).unwrap();
    assert_eq!(reader.feature_kind(), FeatureKind::Peak);
    assert_eq!(reader.cell_id_hash(), Some(cell_id_hash(&barcodes)));
    let gi = reader.genome_intervals().unwrap().expect("interval index");
    assert_eq!(gi.len(), 3);

    // peak x cell in store order: CCC has P1, P2, P3; AAA has P1 and 2 x P3
    let csc = reader
        .read_genes_sparse::<u32>(&[0, 1, 2], None, &DataLayerReturn::Raw)
        .unwrap();
    let col = |g: usize| -> Vec<(u32, u32)> {
        let (s, e) = (csc.indptr[g] as usize, csc.indptr[g + 1] as usize);
        (s..e).map(|k| (csc.indices[k], csc.data[k])).collect()
    };
    assert_eq!(col(0), vec![(0, 1), (1, 1)]);
    assert_eq!(col(1), vec![(0, 1)]);
    assert_eq!(col(2), vec![(0, 1), (1, 2)]);

    // TF-IDF over the three cells: P3 totals 3, so idf = 1
    let idf = set_tf_idf_on_store(cells, genes, None, TF_IDF_SCALE).unwrap();
    assert_eq!(idf, vec![3.0 / 2.0, 3.0, 1.0]);
    let reader = ParallelSparseReader::new(genes).unwrap();
    assert_eq!(reader.norm_kind(), NormKind::TfIdf);
    let csc = reader
        .read_genes_sparse::<u32>(&[2], None, &DataLayerReturn::Norm)
        .unwrap();
    let norm = csc.data_2.as_ref().expect("norm layer");
    let want = (2.0f32 / 3.0 * TF_IDF_SCALE * 1.0).ln_1p();
    assert!((norm[1] - want).abs() < 1e-5, "{} vs {want}", norm[1]);
}
