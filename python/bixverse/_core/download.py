"""Example data sets used by the vignettes, as R's `download_*()` helpers.

Same files and URLs as R: the pinned `bixverse-data` GitHub release first,
the Zenodo copy second. Each helper caches into `cache_dir` and returns the
path straight away when the extracted result is already there, so only the
first call touches the network.

`download_kang_pbmc` and `download_thymus_ageing` fetch h5ad conversions of
the qs2 `SingleCellExperiment`s R uses; they live in the GitHub release only.
"""

from __future__ import annotations

import gzip
import logging
import shutil
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from beartype import beartype

###########
# Globals #
###########

logger = logging.getLogger(__name__)

# Pinned rather than moving, so a given version always resolves the same data.
BIXVERSE_DATA_TAG: str = "v1.0.0"

DEFAULT_CACHE: Path = Path.home() / ".cache" / "bixverse"

DOWNLOAD_TIMEOUT_S: int = 300
DOWNLOAD_TRIES: int = 4
RETRY_BACKOFF_S: int = 5

ZENODO_MAIN: str = "20977604"


###########
# Helpers #
###########


@beartype
def data_urls(file: str, zenodo_record: str | None = None) -> list[str]:
    """Mirrors of an example data file, in the order to try them.

    Args:
        file: File name, identical in both locations.
        zenodo_record: Zenodo record of the fallback copy, `None` for files
            that only live in the GitHub release.

    Returns:
        The URLs.
    """
    gh = (
        "https://github.com/GregorLueg/bixverse-data/releases/download/"
        f"{BIXVERSE_DATA_TAG}/{file}"
    )
    if zenodo_record is None:
        return [gh]
    return [gh, f"https://zenodo.org/records/{zenodo_record}/files/{file}?download=1"]


@beartype
def _download(urls: list[str], dest: Path, *, quiet: bool) -> Path:
    """Download the first mirror that works, retrying with a linear backoff.

    Raises:
        RuntimeError: If every attempt failed.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, DOWNLOAD_TRIES + 1):
        for url in urls:
            try:
                if not quiet:
                    logger.info("Downloading %s", url)
                with (
                    urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp,
                    tmp.open("wb") as fh,
                ):
                    shutil.copyfileobj(resp, fh)
            except (urllib.error.URLError, OSError) as e:
                # a dropped connection is retried on the next mirror or round
                logger.warning("Download of %s failed: %s", url, e)
                tmp.unlink(missing_ok=True)
                continue
            if tmp.stat().st_size > 0:
                tmp.replace(dest)
                return dest
            tmp.unlink(missing_ok=True)
        if attempt < DOWNLOAD_TRIES:
            logger.warning(
                "Download failed (%i/%i), retrying.", attempt, DOWNLOAD_TRIES
            )
            time.sleep(RETRY_BACKOFF_S * attempt)
    raise RuntimeError(f"Download failed after {DOWNLOAD_TRIES} attempts: {urls[0]}")


@beartype
def _cache(cache_dir: Path | str | None) -> Path:
    """The cache directory, created if missing."""
    path = Path(cache_dir).expanduser() if cache_dir is not None else DEFAULT_CACHE
    path.mkdir(parents=True, exist_ok=True)
    return path


@beartype
def _fetch_tar(
    archive: str, url_file: str, result: str, cache_dir: Path | str | None, quiet: bool
) -> Path:
    """Download and extract a tarball unless its result is cached."""
    root = _cache(cache_dir)
    out = root / result
    if out.exists():
        return out
    dest = _download(data_urls(url_file, ZENODO_MAIN), root / archive, quiet=quiet)
    with tarfile.open(dest) as tar:
        tar.extractall(root, filter="data")
    return out


@beartype
def _fetch_gz(
    file: str, zenodo_record: str | None, cache_dir: Path | str | None, quiet: bool
) -> Path:
    """Download and gunzip a file unless the result is cached."""
    root = _cache(cache_dir)
    out = root / file.removesuffix(".gz")
    if out.exists():
        return out
    dest = _download(data_urls(file, zenodo_record), root / file, quiet=quiet)
    with gzip.open(dest, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    dest.unlink()
    return out


@beartype
def _fetch_file(
    file: str,
    urls: list[str],
    cache_dir: Path | str | None,
    quiet: bool,
) -> Path:
    """Download a file as it is unless it is cached."""
    out = _cache(cache_dir) / file
    if out.exists():
        return out
    return _download(urls, out, quiet=quiet)


#############
# Data sets #
#############


@beartype
def download_pbmc3k(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The 10x PBMC 3k filtered matrices.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The extracted `filtered_gene_bc_matrices/hg19` directory.

    Examples:
        >>> pbmc_dir = bv.io.download_pbmc3k()
        >>> mtx_params = bv.io.get_cell_ranger_params(pbmc_dir)
        >>> sc = bv.io.load_mtx("experiments/pbmc3k", mtx_params)
    """
    return _fetch_tar(
        "pbmc3k.tar.gz",
        "pbmc3k_filtered_gene_bc_matrices.tar.gz",
        "filtered_gene_bc_matrices/hg19",
        cache_dir,
        quiet,
    )


@beartype
def download_pbmc8k(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The 10x PBMC 8k data.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The extracted `pmbc-8k` directory.
    """
    return _fetch_tar("pmbc-8k.tar.gz", "pmbc-8k.tar.gz", "pmbc-8k", cache_dir, quiet)


@beartype
def download_demuxlet_pbmc(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """PBMCs with demuxlet doublet calls, for doublet detection.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The extracted `demuxlet_PBMCs` directory.
    """
    return _fetch_tar(
        "demuxlet_PBMCs.tar.gz",
        "demuxlet_PBMCs.tar.gz",
        "demuxlet_PBMCs",
        cache_dir,
        quiet,
    )


@beartype
def download_pbmc_batches(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """Two PBMC h5ad files, for batch correction.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The extracted `pbmc_batches` directory.
    """
    return _fetch_tar(
        "pbmc_batches.tar.gz",
        "pbmc_batches.tar.gz",
        "pbmc_batches",
        cache_dir,
        quiet,
    )


@beartype
def download_pbmc_1k_5p(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The unfiltered 10x PBMC 1k 5' matrix.

    Every barcode is in there, empty droplets included, which is what
    CellSweep needs. The file also carries Antibody Capture features. The
    GitHub release copy comes first; the 10x CDN refuses Python's downloader.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The h5 file.
    """
    file = "sc5p_v2_hs_PBMC_1k_raw_feature_bc_matrix.h5"
    url = f"https://cf.10xgenomics.com/samples/cell-vdj/4.0.0/sc5p_v2_hs_PBMC_1k/{file}"
    return _fetch_file(file, [*data_urls(file), url], cache_dir, quiet)


@beartype
def download_cd34_data(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The CD34 data set of the SEACells paper.

    Persad et al., Nat Biotechnol, 2023.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The h5ad file.
    """
    return _fetch_gz("cd34_multiome_rna.h5ad.gz", ZENODO_MAIN, cache_dir, quiet)


@beartype
def download_dialogue_uc(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The ulcerative colitis subset of the DIALOGUE paper.

    The published `log2(TPM / 10 + 1)` matrix, linearised back to the TPM/10
    scale and rounded so it loads like counts (Smillie et al., Cell, 2019;
    Jerby-Arnon and Regev, Nat Biotechnol, 2022).

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The h5ad file.
    """
    return _fetch_gz("dialogue_uc.h5ad.gz", "22105703", cache_dir, quiet)


@beartype
def download_marrow_cd34(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The bone marrow CD34 data of the Palantir paper.

    Setty et al., Nat Biotechnol, 2019.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The h5ad file.
    """
    return _fetch_gz("marrow_sample_scseq_counts.h5ad.gz", "21892320", cache_dir, quiet)


@beartype
def download_pbmc_totalseq_data(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The 10k PBMC TotalSeq-B data, with ADT counts.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The 10x h5 file.
    """
    file = "10k_Human_PBMC_TotalSeqB.h5"
    return _fetch_file(file, data_urls(file, ZENODO_MAIN), cache_dir, quiet)


@beartype
def download_kang_pbmc(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The Kang et al. IFN-beta stimulated PBMCs (Nat Biotechnol, 2018).

    The raw counts with the cell and gene annotations of R's qs2
    `SingleCellExperiment`, as h5ad.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The h5ad file.
    """
    return _fetch_gz("kang18_8vs8.h5ad.gz", None, cache_dir, quiet)


@beartype
def download_thymus_ageing(
    cache_dir: Path | str | None = None, *, quiet: bool = False
) -> Path:
    """The Baran-Gale et al. ageing thymus data (Development, 2020).

    The raw counts with the cell and gene annotations of R's qs2
    `SingleCellExperiment`, as h5ad.

    Args:
        cache_dir: Where to cache the data. Defaults to `~/.cache/bixverse`.
        quiet: Suppress progress logging.

    Returns:
        The h5ad file.
    """
    return _fetch_gz("thymus_ageing_droplet.h5ad.gz", None, cache_dir, quiet)
