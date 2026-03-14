# News

## 0.1.5

### Features

- Improved Hotspot performance
- Improved SEACells performance and reduced memory pressure
- Thresholded GRNBoost2 pending the data set size

### Fixes

- Bug in the Hotspot streaming version
- R can actually call GRNBoost2

## 0.1.4

### Features

- Sparse SVD and sparse randomised SVD were added for single cell.
- SIMD instructions for the single cell Hotspot method.
- Harmony batch correction for single cell.
- SCENIC-like approach for gene regulatory network (GRN) generation.
- Multi-file support for reading in several h5ad files at once.
- h5ad reader for situations where the file only has normalised counts.

### Fixes

*NA*

## 0.1.3

### Features

*NA*

### Fixes

- Hot fix in the auto detection in Scrublet which was broken.

## 0.1.2

### Features

- Modified graph label propagation algorithm to support max_hops, weights and
  directed graphs.

### Fixes

*NA*

## 0.1.1

### Features

`NA`

### Fixes

- Bumped the version of `ann-search-rs` to ensure consistency across packages/
  crates.

## 0.1.0 (official release)

This version ported over all of the Rust code from
[bixverse](https://github.com/GregorLueg/bixverse) as an independent crate.

### Features

- The hardcoded f32/f64 types were replaced with generic types.
- Redundancy in the codebase was removed.
- Restructuring of the codebase for better organisation.
- Wrong copy/paste documentation was properly updated.

### Fixes

*NA*
