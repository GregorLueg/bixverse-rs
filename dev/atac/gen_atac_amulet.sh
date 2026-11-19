#!/usr/bin/env bash
# AMULET reference outputs on the chr22 fixture, for the clean-room
# implementation in crates/bixverse-atac/src/amulet.rs.
#
# AMULET (UcarLab/AMULET, GPL-3.0) runs as a black box: only its outputs
# enter this repository, never its code. Set up a checkout and a Python 3.11
# environment with numpy < 1.24 (pandas 1.5.3, scipy 1.10.1, statsmodels
# 0.14.0) outside the repository, then from the repository root:
#
#   AMULET_DIR=<checkout with AMULET.sh> AMULET_PY=<venv>/bin \
#     bash dev/atac/gen_atac_amulet.sh
#
# The restriction list is the fixture's own blacklist.bed, so the tests need
# no repeat annotation. Runs at the default overlap threshold (2) and at 1.
# Writes amulet/ under crates/bixverse-atac/tests/fixtures/pbmc3k_chr22/.

set -euo pipefail

FIX="$PWD/crates/bixverse-atac/tests/fixtures/pbmc3k_chr22"
OUT="$FIX/amulet"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

export PATH="$AMULET_PY:$PATH"
mkdir -p "$OUT"

{
    echo "barcode,is__cell_barcode"
    sed 's/$/,1/' "$FIX/cells.txt"
} > "$WORK/singlecell.csv"

for overlap in 2 1; do
    run="$WORK/overlap$overlap"
    mkdir -p "$run"
    (cd "$AMULET_DIR" && bash AMULET.sh --expectedoverlap "$overlap" \
        "$FIX/fragments.tsv.gz" "$WORK/singlecell.csv" human_autosomes.txt \
        "$FIX/blacklist.bed" "$run/" "$AMULET_DIR" > "$run/log.txt" 2>&1)
    for f in OverlapSummary MultipletProbabilities Overlaps; do
        gzip -n -c "$run/$f.txt" > "$OUT/overlap${overlap}_$f.txt.gz"
    done
    cp "$run/MultipletSummary.txt" "$OUT/overlap${overlap}_MultipletSummary.txt"
done

commit="$(cd "$AMULET_DIR" && cat .commit 2>/dev/null || echo unknown)"
echo "AMULET main $commit, run as a black box with defaults except --expectedoverlap" \
    > "$OUT/versions.txt"
