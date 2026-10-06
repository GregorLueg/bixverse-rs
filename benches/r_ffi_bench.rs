//! The R <-> Rust conversions in `utils::r_rust_interface`, timed inside an
//! embedded R session.
//!
//! * `faer_to_r` - dense faer to R matrix, f64 and f32 (widened), plus a
//!   transposed (strided) view.
//! * `r_to_faer_fp32` - R double matrix narrowed into a faer `Mat<f32>`.
//! * `sparse_to_list` / `list_to_sparse` - the CSR export and its round trip.
//! * `named_vec` - named numeric vector to `(Vec<String>, Vec<f64>)`.
//!
//! Run with:
//! ```text
//! cargo bench --bench r_ffi_bench
//! ```
//!
//! `FFI_BENCH_LARGE=1` switches to the larger shapes.

use std::hint::black_box;
use std::time::{Duration, Instant};

use extendr_api::prelude::*;
use faer::Mat;

use bixverse_rs::prelude::*;

///////////////
// Constants //
///////////////

/// Timed repetitions per cell, after one warm-up. The median is reported.
const REPS: usize = 7;

/// `(rows, columns)` of the dense cells, default shape.
const DENSE_SHAPE: (usize, usize) = (10_000, 2_000);

/// `(rows, columns)` of the dense cells under `FFI_BENCH_LARGE`.
const DENSE_SHAPE_LARGE: (usize, usize) = (50_000, 2_000);

/// `(rows, columns, nnz per row)` of the sparse cells, default shape.
const SPARSE_SHAPE: (usize, usize, usize) = (20_000, 30_000, 500);

/// `(rows, columns, nnz per row)` of the sparse cells under `FFI_BENCH_LARGE`.
const SPARSE_SHAPE_LARGE: (usize, usize, usize) = (100_000, 30_000, 1_000);

/// Length of the named vector.
const NAMED_LEN: usize = 1_000_000;

/////////////
// Helpers //
/////////////

/// Median wall time of `REPS` runs after one warm-up.
///
/// ### Params
///
/// * `f` - The closure to time.
///
/// ### Returns
///
/// The median duration.
fn time<F: FnMut()>(mut f: F) -> Duration {
    f();
    let mut times: Vec<Duration> = (0..REPS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    times.sort();
    times[REPS / 2]
}

/// Print one result row.
///
/// ### Params
///
/// * `name` - Cell name.
/// * `d` - Median duration.
/// * `bytes` - Bytes written to the destination, for the throughput column.
fn report(name: &str, d: Duration, bytes: usize) {
    let gbs = bytes as f64 / d.as_secs_f64() / 1e9;
    println!(
        "{name:<28} {:>10.2} ms {:>8.2} GB/s",
        d.as_secs_f64() * 1e3,
        gbs
    );
}

/// Deterministic synthetic CSR.
///
/// ### Params
///
/// * `shape` - `(rows, columns, nnz per row)`.
///
/// ### Returns
///
/// The CSR matrix with f32 values.
fn synthetic_csr(shape: (usize, usize, usize)) -> CompressedSparseData2<f32> {
    let (nrow, ncol, per_row) = shape;
    let step = ncol / per_row;
    let nnz = nrow * per_row;
    let data: Vec<f32> = (0..nnz).map(|i| (i % 97) as f32 + 1.0).collect();
    let indices: Vec<u32> = (0..nnz)
        .map(|i| ((i % per_row) * step + (i / per_row) % step) as u32)
        .collect();
    let indptr: Vec<u32> = (0..=nrow).map(|r| (r * per_row) as u32).collect();
    CompressedSparseData2 {
        data,
        indices,
        indptr,
        cs_type: CompressedSparseFormat::Csr,
        data_2: None,
        shape: (nrow, ncol),
    }
}

//////////
// Main //
//////////

fn main() {
    let large = std::env::var("FFI_BENCH_LARGE").is_ok();
    let (nrow, ncol) = if large {
        DENSE_SHAPE_LARGE
    } else {
        DENSE_SHAPE
    };
    let sparse_shape = if large {
        SPARSE_SHAPE_LARGE
    } else {
        SPARSE_SHAPE
    };

    extendr_engine::with_r(|| {
        println!(
            "dense {nrow} x {ncol}, sparse {sparse_shape:?}, threads {}",
            rayon::current_num_threads()
        );

        // Sequential branches (below the parallel threshold), parity only.
        let small = Mat::<f64>::from_fn(7, 5, |i, j| (i * 5 + j) as f64);
        assert_eq!(
            r_matrix_to_faer(&faer_to_r_matrix(small.as_ref())),
            small.as_ref()
        );
        assert_eq!(
            r_matrix_to_faer(&faer_to_r_matrix(small.transpose())),
            small.transpose()
        );
        let small_fp32 = r_matrix_to_faer_fp32(&faer_to_r_matrix(small.as_ref()));
        assert!((0..5).all(|j| (0..7).all(|i| small_fp32[(i, j)] == small[(i, j)] as f32)));
        let small_csr = synthetic_csr((10, 20, 4));
        let back =
            list_to_sparse_matrix::<f32>(sparse_data_to_list(small_csr.clone()).unwrap(), false)
                .unwrap();
        assert_eq!(
            (back.data, back.indices, back.indptr),
            (small_csr.data, small_csr.indices, small_csr.indptr)
        );

        // faer -> R
        let m64 = Mat::<f64>::from_fn(nrow, ncol, |i, j| (i * 31 + j * 7) as f64 * 1e-3);
        let m32 = Mat::<f32>::from_fn(nrow, ncol, |i, j| (i * 31 + j * 7) as f32 * 1e-3);
        let out_bytes = nrow * ncol * 8;

        let d = time(|| {
            black_box(faer_to_r_matrix(m64.as_ref()));
        });
        report("faer_to_r f64", d, out_bytes);

        let d = time(|| {
            black_box(faer_to_r_matrix(m32.as_ref()));
        });
        report("faer_to_r f32", d, out_bytes);

        let d = time(|| {
            black_box(faer_to_r_matrix(m64.transpose()));
        });
        report("faer_to_r f64 transposed", d, out_bytes);

        // Parity, so a layout bug cannot pass as a speed-up.
        let r = faer_to_r_matrix(m64.as_ref());
        let back = r_matrix_to_faer(&r);
        assert_eq!(back, m64.as_ref());
        let r = faer_to_r_matrix(m64.transpose());
        assert_eq!(r_matrix_to_faer(&r), m64.transpose());
        let r32 = faer_to_r_matrix(m32.as_ref());
        assert!(
            (0..ncol)
                .all(|j| (0..nrow).all(|i| r_matrix_to_faer(&r32)[(i, j)] == m32[(i, j)] as f64))
        );

        // R -> faer f32. Tall is the embedding shape the callers pass; wide is
        // bound by faer's sequential `Mat::zeros`.
        let tall = faer_to_r_matrix(m64.as_ref());
        let wide = r;
        for (name, x) in [
            ("r_to_faer_fp32 tall", &tall),
            ("r_to_faer_fp32 wide", &wide),
        ] {
            let d = time(|| {
                black_box(r_matrix_to_faer_fp32(x));
            });
            report(name, d, nrow * ncol * 4);
            let narrowed = r_matrix_to_faer_fp32(x);
            let src = r_matrix_to_faer(x);
            assert!(
                (0..src.ncols())
                    .all(|j| (0..src.nrows()).all(|i| narrowed[(i, j)] == src[(i, j)] as f32))
            );
        }
        drop((tall, wide, r32));

        // Sparse
        let csr = synthetic_csr(sparse_shape);
        let nnz = csr.data.len();
        let d = time(|| {
            black_box(sparse_data_to_list(csr.clone()).unwrap());
        });
        report("sparse_to_list (incl clone)", d, nnz * 16);
        let d = time(|| {
            black_box(csr.clone());
        });
        report("  clone alone", d, nnz * 12);

        let list = sparse_data_to_list(csr.clone()).unwrap();
        let d = time(|| {
            black_box(list_to_sparse_matrix::<f32>(list.clone(), false).unwrap());
        });
        report("list_to_sparse", d, nnz * 12);
        let back = list_to_sparse_matrix::<f32>(list, false).unwrap();
        assert_eq!(back.data, csr.data);
        assert_eq!(back.indices, csr.indices);
        assert_eq!(back.indptr, csr.indptr);

        // Named vector
        let values: Vec<f64> = (0..NAMED_LEN).map(|i| i as f64).collect();
        let names: Vec<String> = (0..NAMED_LEN).map(|i| format!("g{i}")).collect();
        let mut named: Robj = values.clone().into();
        named.set_names(names.clone()).unwrap();
        let d = time(|| {
            black_box(r_named_vec_data(named.clone()).unwrap());
        });
        report("named_vec", d, NAMED_LEN * 8);
        let (n, v) = r_named_vec_data(named).unwrap();
        assert_eq!(n, names);
        assert_eq!(v, values);
        Ok::<(), Error>(())
    })
    .expect("R session");
}
