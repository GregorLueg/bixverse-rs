//! decoupler-style ULM and MLM activity scores per cell, see Badia-i-Mompel,
//! et al., Bioinform. Adv., 2022. The models live in
//! [bixverse_methods::enrichment::lm_activity]; this streams the cell store
//! through them on the normalised counts.

use faer::Mat;
use std::time::Instant;

use crate::prelude::*;
use bixverse_methods::enrichment::lm_activity::{
    FeatureSetWeights, LmActivityParams, LmActivityRes, LmScorer, ObsCrossProducts,
};

/// Cells per batch. Bounds the f64 cross products and scores held at once to
/// a few `LM_ACTIVITY_CHUNK * n_sets` buffers. Not tuned.
const LM_ACTIVITY_CHUNK: usize = 50_000;

/// ULM or MLM activity scores across a set of cells
///
/// Every cell is one fit over all `total_genes` of the store, the implicit
/// zeros included, on the normalised counts. Cells are read and scored in
/// batches of [LM_ACTIVITY_CHUNK].
///
/// ### Params
///
/// * `reader` - Reader over the cell-based count store.
/// * `net` - The feature sets over the genes of the store.
/// * `cells_to_keep` - Indices of the cells to score.
/// * `params` - Method, statistic and p-value switch. `None` gives ULM t
///   values without p-values.
/// * `verbose` - If `0` -> silent or `1` for normal verbosity, `2` for detailed
///   verbosity.
///
/// ### Returns
///
/// The scores, cells x sets, in the order of `cells_to_keep`.
pub fn calculate_lm_activity<S: SingleCellReading>(
    reader: &S,
    net: &FeatureSetWeights,
    cells_to_keep: &[usize],
    params: Option<LmActivityParams>,
    verbose: usize,
) -> Result<LmActivityRes<f32>, ScErrors> {
    let verbosity = parse_verbosity_level(verbose);
    let params = params.unwrap_or_default();

    let n_genes = reader.get_header().total_genes;
    if n_genes != net.n_features {
        return Err(MethodsErrors::LmActivityShapeMismatch {
            expected: net.n_features,
            got: n_genes,
        }
        .into());
    }
    let scorer = LmScorer::new(net, &params)?;

    let n_cells = cells_to_keep.len();
    let total_chunks = n_cells.div_ceil(LM_ACTIVITY_CHUNK);
    let mut es: Vec<f32> = Vec::with_capacity(n_cells * net.n_sets);
    let mut pv: Option<Vec<f32>> = params
        .p_values
        .then(|| Vec::with_capacity(n_cells * net.n_sets));

    let start_total = Instant::now();
    for (chunk_idx, cell_idx) in cells_to_keep.chunks(LM_ACTIVITY_CHUNK).enumerate() {
        let start_chunk = Instant::now();

        let cells = reader.read_cells_parallel(cell_idx)?;
        let cp = ObsCrossProducts::new(net, cells.len(), |o| {
            let cell = &cells[o];
            cell.indices
                .iter()
                .zip(&cell.data_norm)
                .map(|(&i, v)| (i as usize, v.to_f32() as f64))
        });
        let (chunk_es, chunk_pv) = scorer.score(&cp);

        es.extend(chunk_es.iter().map(|&x| x as f32));
        if let (Some(pv), Some(chunk_pv)) = (pv.as_mut(), chunk_pv) {
            pv.extend(chunk_pv.iter().map(|&x| x as f32));
        }

        if verbosity.detailed_verbosity() {
            println!(
                "Scored chunk {} out of {} in {:.2?}",
                chunk_idx + 1,
                total_chunks,
                start_chunk.elapsed()
            );
        }
    }

    if verbosity.normal_verbosity() {
        println!(
            "Calculated linear model activity scores: {:.2?}",
            start_total.elapsed()
        );
    }

    let n_sets = net.n_sets;
    let to_mat = |v: Vec<f32>| Mat::from_fn(n_cells, n_sets, |i, j| v[i * n_sets + j]);
    Ok(LmActivityRes {
        es: to_mat(es),
        p_values: pv.map(to_mat),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use bixverse_methods::enrichment::lm_activity::{LmMethod, LmStat, lm_activity};
    use bixverse_sc_data::data_io::CellGeneSparseWriter;

    /// Scratch store that removes itself on drop.
    struct TempStore(std::path::PathBuf);

    impl Drop for TempStore {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    /// Write a cell-based store and open a reader over it.
    ///
    /// ### Params
    ///
    /// * `temp` - Guard owning the path
    /// * `dense` - `dense[cell][gene]` raw counts
    ///
    /// ### Returns
    ///
    /// The reader.
    fn reader_for(temp: &TempStore, dense: &[Vec<u32>]) -> ParallelSparseReader {
        let path = temp.0.to_str().expect("temp path is valid UTF-8");
        let mut writer = CellGeneSparseWriter::new(path, true, dense.len(), dense[0].len(), 1e4)
            .expect("writer opens");
        for (cell_idx, cell) in dense.iter().enumerate() {
            let (raw, indices): (Vec<u32>, Vec<u32>) = cell
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v > 0)
                .map(|(g, &v)| (v, g as u32))
                .unzip();
            writer
                .write_cell_chunk(CsrCellChunk::from_data(&raw, &indices, cell_idx, 1e4, true))
                .expect("write cell chunk");
        }
        writer.finalise().expect("finalise");
        ParallelSparseReader::new(path).expect("reader opens")
    }

    #[test]
    fn test_lm_activity_sc_matches_dense() {
        let (n_cells, n_genes) = (12, 60);
        let dense: Vec<Vec<u32>> = (0..n_cells)
            .map(|c| {
                (0..n_genes)
                    .map(|g| {
                        if (g * 5 + c * 3) % 4 == 0 {
                            ((g + c) % 9 + 1) as u32
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect();
        let temp = TempStore(std::env::temp_dir().join("bixverse_lm_activity_sc.bin"));
        let reader = reader_for(&temp, &dense);

        let cells: Vec<usize> = (0..n_cells).collect();
        let chunks = reader.read_cells_parallel(&cells).unwrap();
        let mut mat = Mat::<f64>::zeros(n_genes, n_cells);
        for (c, chunk) in chunks.iter().enumerate() {
            for (&g, v) in chunk.indices.iter().zip(&chunk.data_norm) {
                mat[(g as usize, c)] = v.to_f32() as f64;
            }
        }

        let sets = vec![
            vec![0, 4, 8, 12, 16],
            vec![1, 4, 9, 30, 31, 50],
            vec![2, 3, 40, 41],
        ];
        let weights = vec![
            vec![1.0, 1.0, -1.0, 0.5, 2.0],
            vec![1.0, -0.5, 1.0, 1.0, 1.0, -1.0],
            vec![1.0, 1.0, 1.0, 1.0],
        ];
        let net = FeatureSetWeights::new(&sets, Some(&weights), n_genes).unwrap();

        for method in [LmMethod::Ulm, LmMethod::Mlm] {
            let params = LmActivityParams::new(method, LmStat::TValue, true);
            let sc = calculate_lm_activity(&reader, &net, &cells, Some(params), 0).unwrap();
            let reference = lm_activity(mat.as_ref(), &net, Some(params)).unwrap();
            let (sc_p, ref_p) = (sc.p_values.unwrap(), reference.p_values.unwrap());
            for c in 0..n_cells {
                for s in 0..sets.len() {
                    assert_relative_eq!(
                        sc.es[(c, s)] as f64,
                        reference.es[(c, s)],
                        max_relative = 1e-6
                    );
                    assert_relative_eq!(sc_p[(c, s)] as f64, ref_p[(c, s)], max_relative = 1e-5);
                }
            }
        }
    }

    #[test]
    fn test_lm_activity_sc_rejects_gene_mismatch() {
        let dense = vec![vec![1_u32, 0, 2, 0, 3, 0, 1, 1]];
        let temp = TempStore(std::env::temp_dir().join("bixverse_lm_activity_sc_mismatch.bin"));
        let reader = reader_for(&temp, &dense);
        let net = FeatureSetWeights::new::<f64>(&[vec![0, 1, 2]], None, 9).unwrap();
        assert!(matches!(
            calculate_lm_activity(&reader, &net, &[0], None, 0),
            Err(ScErrors::Methods(
                MethodsErrors::LmActivityShapeMismatch { .. }
            ))
        ));
    }
}
