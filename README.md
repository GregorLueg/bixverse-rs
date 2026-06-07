[![CI](https://github.com/GregorLueg/bixverse-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/bixverse-rs/actions/workflows/test.yml)
[![Crates.io](https://img.shields.io/crates/v/bixverse-rs.svg)](https://crates.io/crates/bixverse-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# bixverse-rs

## Description

After having lived in `/src/Rust/` folder of the
[bixverse R package](https://github.com/GregorLueg/bixverse), the
decision was made to pull out ALL of the functionality and do a refactor on
top of the code. This has now led to this independent crate that has
all the core functionality extracted out.

## Roadmap

### Methods

Slowly, but surely increase the number of methods that are implemented here. The
next targets would be

- [ ] NMF for dense and sparse data
- [ ] Palantir for single-cell trajectories, see
  [Setty, et al.](https://www.nature.com/articles/s41587-019-0068-4)
- [ ] NicheNet for single cell, see
  [Browaeys et al.](https://www.nature.com/articles/s41592-019-0667-5)

### GPU accelerations

- [x] GPU-accelerated sparse, randomised SVD
- [x] GPU-accelerated Harmony
- [ ] GPU-accelerated BBKNN version
- [ ] GPU-accelerated correlations

### Python interface

- [ ] Data loader to pull in the counts from the binary files to expose the data
  to Python-based deep learning frameworks (JAX, PyTorch).

## Updates

Updates on what's happening in this crate can be found
[here](https://github.com/GregorLueg/bixverse-rs/blob/main/docs/news.MD)

## Licence

MIT License

Copyright (c) 2026 Gregor Alexander Lueg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
