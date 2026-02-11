use faer::Mat;
use rayon::prelude::*;

/// Apply cosine normalisation (L2 normalisation) to each row
///
/// Normalizes each row (cell) to unit L2 norm. Rows with near-zero norm
/// are left as zeros to avoid division by zero.
///
/// ### Params
///
/// * `mat` - The matrix on which to apply the Cosine normalisation per row
///
/// ### Returns
///
/// Per row L2-normalised data
pub fn cosine_normalise(mat: &Mat<f32>) -> Mat<f32> {
    let nrows = mat.nrows();
    let ncols = mat.ncols();

    // compute norms once per row (in parallel)
    let norms: Vec<f32> = (0..nrows)
        .into_par_iter()
        .map(|row| mat.get(row, ..).norm_l2())
        .collect();

    // create normalised matrix
    Mat::from_fn(nrows, ncols, |row, col| {
        let norm = norms[row];
        if norm > 1e-8 {
            mat[(row, col)] / norm
        } else {
            0.0 // Zero-norm row stays zero
        }
    })
}

///////////
// Tests //
//////////

#[cfg(test)]
mod tests_batch_utils {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cosine_normalise_basic() {
        // Regular vectors with different magnitudes
        let mat = Mat::from_fn(3, 3, |i, j| {
            match i {
                0 => [3.0, 4.0, 0.0][j], // norm = 5
                1 => [1.0, 0.0, 0.0][j], // norm = 1
                2 => [0.5, 0.5, 0.5][j], // norm = sqrt(0.75)
                _ => unreachable!(),
            }
        });

        let normalised = cosine_normalise(&mat);

        // Check all rows have unit norm
        for row in 0..3 {
            let norm: f32 = (0..3)
                .map(|col| normalised[(row, col)].powi(2))
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
        }

        // Check specific values
        assert_relative_eq!(normalised[(0, 0)], 3.0 / 5.0, epsilon = 1e-6);
        assert_relative_eq!(normalised[(0, 1)], 4.0 / 5.0, epsilon = 1e-6);
        assert_relative_eq!(normalised[(1, 0)], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_normalise_zero_vector() {
        let mat = Mat::from_fn(3, 3, |i, j| {
            match i {
                0 => [1.0, 0.0, 0.0][j],
                1 => [0.0, 0.0, 0.0][j], // Zero vector
                2 => [0.0, 1.0, 0.0][j],
                _ => unreachable!(),
            }
        });

        let normalised = cosine_normalise(&mat);

        // Zero vector should stay zero
        for col in 0..3 {
            assert_eq!(normalised[(1, col)], 0.0);
        }

        // Other rows should be normalised
        assert_relative_eq!(normalised[(0, 0)], 1.0, epsilon = 1e-6);
        assert_relative_eq!(normalised[(2, 1)], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_normalise_already_normalised() {
        // Create already normalised vectors
        let mat = Mat::from_fn(2, 3, |i, j| match i {
            0 => [1.0, 0.0, 0.0][j],
            1 => [0.0, 1.0, 0.0][j],
            _ => unreachable!(),
        });

        let normalised = cosine_normalise(&mat);

        // Should remain unchanged (within floating point precision)
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(normalised[(i, j)], mat[(i, j)], epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn test_cosine_normalise_small_values() {
        // Very small values (but not zero)
        let mat = Mat::from_fn(2, 3, |i, j| {
            match i {
                0 => [1e-5, 2e-5, 2e-5][j], // norm = 3e-5
                1 => [0.01, 0.02, 0.02][j], // norm = 0.03
                _ => unreachable!(),
            }
        });

        let normalised = cosine_normalise(&mat);

        // Both should normalize to unit vectors
        for row in 0..2 {
            let norm: f32 = (0..3)
                .map(|col| normalised[(row, col)].powi(2))
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_cosine_normalise_large_values() {
        // Very large values
        let mat = Mat::from_fn(2, 3, |i, j| match i {
            0 => [1000.0, 2000.0, 2000.0][j],
            1 => [5000.0, 0.0, 0.0][j],
            _ => unreachable!(),
        });

        let normalised = cosine_normalise(&mat);

        // Should normalize correctly despite large magnitudes
        for row in 0..2 {
            let norm: f32 = (0..3)
                .map(|col| normalised[(row, col)].powi(2))
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_cosine_normalise_near_zero_threshold() {
        // Test values right at the threshold
        let mat = Mat::from_fn(3, 3, |i, j| {
            match i {
                0 => [1e-9, 1e-9, 1e-9][j], // Below threshold (norm ≈ 1.7e-9)
                1 => [1e-7, 1e-7, 1e-7][j], // Above threshold (norm ≈ 1.7e-7)
                2 => [1e-8, 0.0, 0.0][j],   // Exactly at threshold (norm = 1e-8)
                _ => unreachable!(),
            }
        });

        let normalized = cosine_normalise(&mat);

        // Row 0: below threshold, should be zero
        for col in 0..3 {
            assert_eq!(normalized[(0, col)], 0.0);
        }

        // Row 1: above threshold, should be normalized
        let norm: f32 = (0..3)
            .map(|col| normalized[(1, col)].powi(2))
            .sum::<f32>()
            .sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-4);

        // Row 2: exactly at threshold (1e-8 > 1e-8 is false), should be zero
        for col in 0..3 {
            assert_eq!(normalized[(2, col)], 0.0);
        }
    }

    #[test]
    fn test_cosine_normalise_single_column() {
        // Edge case: single column (scalars)
        let mat = Mat::from_fn(3, 1, |i, _| match i {
            0 => 5.0,
            1 => 0.0,
            2 => -3.0,
            _ => unreachable!(),
        });

        let normalised = cosine_normalise(&mat);

        assert_relative_eq!(normalised[(0, 0)], 1.0, epsilon = 1e-6);
        assert_eq!(normalised[(1, 0)], 0.0);
        assert_relative_eq!(normalised[(2, 0)], -1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_normalise_mixed_cases() {
        // Mix of normal, zero, and edge cases
        let mat = Mat::from_fn(5, 4, |i, j| {
            match i {
                0 => [3.0, 4.0, 0.0, 0.0][j],       // Normal
                1 => [0.0, 0.0, 0.0, 0.0][j],       // Zero
                2 => [1.0, 1.0, 1.0, 1.0][j],       // Equal values
                3 => [100.0, 200.0, 300.0, 0.0][j], // Large
                4 => [0.001, 0.002, 0.002, 0.0][j], // Small
                _ => unreachable!(),
            }
        });

        let normalised = cosine_normalise(&mat);

        // Check zero row
        for col in 0..4 {
            assert_eq!(normalised[(1, col)], 0.0);
        }

        // Check all non-zero rows have unit norm
        for row in [0, 2, 3, 4] {
            let norm: f32 = (0..4)
                .map(|col| normalised[(row, col)].powi(2))
                .sum::<f32>()
                .sqrt();
            assert_relative_eq!(norm, 1.0, epsilon = 1e-5);
        }
    }
}
