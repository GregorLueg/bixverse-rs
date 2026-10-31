"""The DuckDB layer, the map and cache, persistence and count reads."""

import warnings

import numpy as np
import polars as pl
import pytest

import bixverse as bv
from bixverse._core import db

N_CELLS = 40
N_GENES = 12
GROUPS = ["b", "a", "c"]  # level order deliberately not sorted


@pytest.fixture
def exp(tmp_path):
    """An experiment directory built in-test, plus the dense raw counts."""
    dense = np.asarray(
        bv._bixverse._write_synthetic_store(
            str(tmp_path / "counts_cells.bin"), N_CELLS, N_GENES, 0.4, 7
        )
    )
    bv._bixverse.generate_gene_based_data(
        tmp_path / "counts_cells.bin", tmp_path / "counts_genes.bin"
    )

    obs = pl.DataFrame(
        {
            "cell_id": [f"cell_{i:02d}" for i in range(N_CELLS)],
            "grp": pl.Series(
                [GROUPS[i % 3] for i in range(N_CELLS)], dtype=pl.Enum(GROUPS)
            ),
            "lib_size": dense.sum(axis=1).astype(np.float64),
        }
    )
    with db.connect(db.db_path(tmp_path)) as con:
        db.populate_from_frame(con, "obs", obs)
        db.set_to_keep_column(con)
        db.populate_var_minimal(con, [f"gene_{j:02d}" for j in range(N_GENES)])
    return tmp_path, dense


def _raw_sql(path, sql):
    with db.connect(db.db_path(path), read_only=True) as con:
        return con.execute(sql).fetchall()


#############
# DuckDB io #
#############


def test_disk_index_is_one_based_python_is_zero_based(exp):
    path, _ = exp
    assert _raw_sql(path, "SELECT min(cell_idx), max(cell_idx) FROM obs") == [
        (1, N_CELLS)
    ]
    assert _raw_sql(path, "SELECT min(gene_idx) FROM var") == [(1,)]
    sc = bv.SingleCells(path)
    obs = sc.obs()
    assert obs["cell_idx"].to_list() == list(range(N_CELLS))
    assert sc.var()["gene_idx"].to_list() == list(range(N_GENES))


def test_column_types_match_what_r_writes(exp):
    path, _ = exp
    types = dict(
        _raw_sql(
            path,
            "SELECT column_name, data_type FROM duckdb_columns()"
            " WHERE table_name = 'obs'",
        )
    )
    assert types["cell_idx"] == "INTEGER"
    assert types["to_keep"] == "BOOLEAN"
    assert types["grp"] == "ENUM('b', 'a', 'c')"


def test_enum_comes_back_with_its_level_order(exp):
    path, _ = exp
    grp = bv.SingleCells(path).obs()["grp"]
    assert grp.dtype == pl.Enum(GROUPS)


def test_obs_selection(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sub = sc.obs(indices=[5, 2, 30], cols=["cell_idx", "cell_id"])
    assert sub.columns == ["cell_idx", "cell_id"]
    assert sub["cell_idx"].to_list() == [2, 5, 30]
    assert sub["cell_id"].to_list() == ["cell_02", "cell_05", "cell_30"]


def test_unknown_column_is_rejected_before_any_sql(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    with pytest.raises(KeyError):
        sc.obs(cols=['cell_id" FROM obs; DROP TABLE obs; --'])
    assert sc.obs().height == N_CELLS


def test_missing_table_raises(tmp_path):
    with db.connect(db.db_path(tmp_path)) as con, pytest.raises(bv.TableNotFoundError):
        db.get_table(con, "obs")


def test_cell_filter_round_trips_through_to_keep(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.set_cells_to_keep([9, 3, 1, 3])
    # unique and ascending, whatever order is given
    assert sc.cells_to_keep.tolist() == [1, 3, 9]
    assert sc.shape == (3, N_GENES)
    assert sorted(
        r[0] for r in _raw_sql(path, "SELECT cell_idx FROM obs WHERE to_keep")
    ) == [
        2,
        4,
        10,
    ]
    assert sc.obs(filtered=True)["cell_idx"].to_list() == [1, 3, 9]
    # a fresh handle reads the same filter back from the DuckDB
    assert bv.SingleCells(path).cells_to_keep.tolist() == [1, 3, 9]

    sc.reset_cells_to_keep()
    assert sc.n_cells == N_CELLS
    assert _raw_sql(path, "SELECT count(*) FROM obs WHERE to_keep") == [(N_CELLS,)]


def test_cell_filter_by_id(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    with pytest.warns(UserWarning, match="could not be matched"):
        sc.set_cells_to_keep(["cell_04", "nope", "cell_01"])
    assert sc.cells_to_keep.tolist() == [1, 4]


def test_add_obs_aligns_to_kept_cells(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.set_cells_to_keep([7, 2, 5])
    sc.add_obs(pl.DataFrame({"score": [0.2, 0.5, 0.7]}))  # cells 2, 5, 7
    obs = sc.obs(cols=["cell_idx", "score"])
    assert obs.filter(pl.col("score").is_not_null())["cell_idx"].to_list() == [2, 5, 7]
    assert obs["score"].to_list()[2] == pytest.approx(0.2)
    assert obs["score"].to_list()[7] == pytest.approx(0.7)
    assert obs["score"].null_count() == N_CELLS - 3

    with pytest.raises(ValueError, match="kept cells"):
        sc.add_obs(pl.DataFrame({"score": [1.0]}))


def test_add_obs_by_key_replaces_a_column_and_writes_enum(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    lab = pl.Series(["y", "x"], dtype=pl.Enum(["y", "x"]))
    sc.add_obs(pl.DataFrame({"cell_idx": [0, 39], "grp": ["q", "r"], "lab": lab}))
    obs = sc.obs(cols=["grp", "lab"])
    assert obs["grp"].to_list()[0] == "q"
    assert obs["grp"].to_list()[1] is None
    assert obs["lab"].dtype == pl.Enum(["y", "x"])
    assert obs["lab"].to_list()[39] == "x"

    with pytest.raises(ValueError, match="out of range"):
        sc.add_obs(pl.DataFrame({"cell_idx": [N_CELLS], "z": [1]}))


def test_add_var_needs_one_row_per_gene(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.add_var(pl.DataFrame({"is_marker": [j < 3 for j in range(N_GENES)]}))
    assert sc.var()["is_marker"].sum() == 3
    with pytest.raises(ValueError, match="rows"):
        sc.add_var(pl.DataFrame({"x": [1, 2]}))


def test_rename_and_drop(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.rename_obs_col("lib_size", "library")
    assert "library" in sc.obs().columns
    with pytest.warns(UserWarning, match="not found"):
        sc.rename_obs_col("absent", "x")
    with pytest.warns(UserWarning, match="protected"):
        sc.drop_obs_cols(["cell_id", "library"])
    cols = sc.obs().columns
    assert "library" not in cols
    assert "cell_id" in cols


def test_store_and_tables_must_agree(exp):
    path, _ = exp
    with db.connect(db.db_path(path)) as con:
        db.populate_var_minimal(con, ["only_one"])
    with pytest.raises(ValueError, match="do not match"):
        bv.SingleCells(path)


###########
# Parquet #
###########


def _write_part(path, ids, groups, info_levels):
    pl.DataFrame(
        {
            "cell_idx": pl.Series(range(1, len(ids) + 1), dtype=pl.Int32),
            "cell_id": ids,
            "grp": groups,
        }
    ).write_parquet(path)
    return db.ParquetFrameInfo(
        names=("cell_idx", "cell_id", "grp"),
        types=("integer", "string", "categorical" if info_levels else "string"),
        categories=(None, None, info_levels),
    )


@pytest.mark.parametrize("literal_max", [db.SC_ENUM_LITERAL_MAX, 1])
def test_parquet_to_duckdb_casts_categoricals(tmp_path, literal_max):
    pq = tmp_path / "obs.parquet"
    info = _write_part(pq, ["a", "b"], ["t2", "t1"], ("t2", "t1", "unused"))
    with db.connect(db.db_path(tmp_path)) as con:
        db.parquet_to_duckdb(con, "obs", pq, info, enum_literal_max=literal_max)
        out = db.get_table(con, "obs")
        type_names = [
            r[0]
            for r in con.execute(
                "SELECT type_name FROM duckdb_types() WHERE type_name = 'obs_grp_enum'"
            ).fetchall()
        ]
        # reloading drops the named type it made before
        db.parquet_to_duckdb(con, "obs", pq, info, enum_literal_max=literal_max)
    assert out["grp"].dtype == pl.Enum(["t2", "t1", "unused"])
    assert out["cell_idx"].to_list() == [0, 1]
    assert type_names == ([] if literal_max > 1 else ["obs_grp_enum"])


def test_parquet_info_must_match_the_file(tmp_path):
    pq = tmp_path / "obs.parquet"
    _write_part(pq, ["a"], ["t"], None)
    info = db.ParquetFrameInfo(
        names=("missing",), types=("string",), categories=(None,)
    )
    with db.connect(db.db_path(tmp_path)) as con, pytest.raises(KeyError):
        db.parquet_to_duckdb(con, "obs", pq, info)


def test_parquet_parts_stack_prefix_and_union_levels(tmp_path):
    p1, p2 = tmp_path / "a.parquet", tmp_path / "b.parquet"
    i1 = _write_part(p1, ["c1", "c2"], ["x", "y"], ("y", "x"))
    i2 = _write_part(p2, ["c1"], ["z"], None)  # string in this part
    parts = [db.ParquetPart(p1, i1, "e1"), db.ParquetPart(p2, i2, "e2")]
    with db.connect(db.db_path(tmp_path)) as con:
        db.parquet_parts_to_duckdb(con, "obs", parts)
        out = db.get_table(con, "obs")
    assert out.columns == ["cell_idx", "cell_id", "exp_id", "grp"]
    assert out["cell_idx"].to_list() == [0, 1, 2]
    assert out["cell_id"].to_list() == ["e1_c1", "e1_c2", "e2_c1"]
    assert out["grp"].dtype == pl.Enum(["y", "x", "z"])


##########
# Counts #
##########


def test_get_counts_matches_the_written_store(exp):
    path, dense = exp
    sc = bv.SingleCells(path)
    cells, genes = np.array([30, 4, 17]), np.array([11, 0, 5])
    c = bv.get_counts(sc, cells=cells, genes=genes)
    assert c.matrix.shape == (3, 3)
    assert c.matrix.indptr.dtype == np.uint32
    np.testing.assert_array_equal(c.scipy_csr().toarray(), dense[np.ix_(cells, genes)])
    np.testing.assert_array_equal(c.cells, cells)

    by_id = bv.get_counts(sc, cells=["cell_30"], genes=["gene_11", "gene_00"])
    np.testing.assert_array_equal(by_id.scipy_csr().toarray(), dense[[30]][:, [11, 0]])


def test_get_counts_respects_the_cell_filter(exp):
    path, dense = exp
    sc = bv.SingleCells(path)
    sc.set_cells_to_keep([8, 3])
    c = bv.get_counts(sc)
    np.testing.assert_array_equal(c.cells, [3, 8])
    np.testing.assert_array_equal(c.scipy_csr().toarray(), dense[[3, 8]])

    picked = bv.get_counts(sc, cells=[1, 3, 8])
    np.testing.assert_array_equal(picked.cells, [3, 8])
    unfiltered = bv.get_counts(sc, cells=[1], use_cells_to_keep=False)
    np.testing.assert_array_equal(unfiltered.scipy_csr().toarray(), dense[[1]])

    with pytest.raises(ValueError, match="no cells"):
        bv.get_counts(sc, cells=[0])


def test_get_counts_norm_layer_shares_the_pattern(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    raw = bv.get_counts(sc, cells=[0, 1]).scipy_csr().toarray()
    norm = bv.get_counts(sc, cells=[0, 1], layer="norm").scipy_csr().toarray()
    assert ((raw > 0) == (norm > 0)).all()
    assert not np.array_equal(raw, norm)


##############
# Provenance #
##############


def _pca(n):
    return np.ones((n, 2), dtype=np.float32)


def test_filter_change_flags_stamped_artefacts_without_deleting(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.cache.pca_factors = _pca(sc.n_cells)
    sc.stamp("pca")
    sc.cache.embeddings["umap"] = _pca(sc.n_cells)
    sc.stamp("umap", parents=["pca"])
    assert sc.check_state(["pca", "umap"]) is True

    with pytest.warns(UserWarning, match="pca, umap"):
        sc.set_cells_to_keep(list(range(10)))
    assert sc.cache.pca_factors is not None
    report = sc.cache_status()
    assert report["stale"].to_list() == [True, True]
    assert "cell set changed" in sc.check_state(["umap"])


def test_recomputing_a_parent_makes_children_stale(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.cache.pca_factors = _pca(sc.n_cells)
    sc.stamp("pca")
    sc.cache.knn = bv.KnnGraph(
        np.zeros((sc.n_cells, 3), np.uint32),
        np.zeros((sc.n_cells, 3), np.float32),
        "cosine",
    )
    sc.stamp("knn", parents=["pca"])

    sc.cache.pca_factors = _pca(sc.n_cells)  # assigning drops the stamp
    assert sc.cache.artefacts()["pca"] is None
    sc.stamp("pca")
    status = dict(sc.cache_status().select("artefact", "reason").iter_rows())
    assert (
        status["knn"] == "the artefact it was derived from was re-computed or removed"
    )


def test_replacing_an_embedding_unstamps_it(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.cache.embeddings["umap"] = _pca(sc.n_cells)
    sc.stamp("umap")
    sc.cache.embeddings["umap"] = _pca(sc.n_cells)
    assert sc.cache.artefacts()["umap"] is None


def test_reset_needs_force_when_something_is_cached(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.cache.pca_factors = _pca(sc.n_cells)
    with pytest.raises(ValueError, match="force"):
        sc.reset_cells_to_keep()
    sc.reset_cells_to_keep(force=True)
    assert sc.cache.pca_factors is None


def test_reserved_embedding_name_is_rejected(exp):
    path, _ = exp
    sc = bv.SingleCells(path)
    sc.cache.embeddings["knn"] = _pca(sc.n_cells)
    with pytest.raises(ValueError, match="reserved"):
        sc.cache_status()


###############
# Persistence #
###############


def test_save_and_load_round_trip(exp):
    path, _ = exp
    sc = bv.io.load_existing(path)
    sc.set_cells_to_keep([5, 1, 9, 2])
    sc.map.hvg = np.array([3, 0], dtype=np.uint32)
    sc.cache.pca_factors = _pca(4)
    sc.stamp("pca")
    sc.cache.embeddings["umap"] = np.arange(8, dtype=np.float32).reshape(4, 2)
    sc.stamp("umap", parents=["pca"])
    sc.cache.connectivities = bv.CsrMatrix(
        np.ones(2, np.float32),
        np.array([1, 0], np.uint32),
        np.array([0, 1, 2, 2, 2], np.uint64),
        (4, 4),
    )
    sc.cache.magic = bv.MagicLayer(
        np.ones((4, 2), np.float32), np.array(["g1", "g2"], object)
    )
    bv.io.save(sc)
    assert (path / "memory_py.npz").exists()
    assert (path / "memory_py.json").exists()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = bv.io.load_existing(path)
    assert back.cells_to_keep.tolist() == [1, 2, 5, 9]
    assert back.map.hvg.tolist() == [3, 0]
    np.testing.assert_array_equal(
        back.cache.embeddings["umap"], sc.cache.embeddings["umap"]
    )
    assert back.cache.connectivities.indptr.dtype == np.uint64
    assert back.cache.magic.features.tolist() == ["g1", "g2"]
    assert back.cache.stamps == sc.cache.stamps
    assert back.check_state(["pca", "umap"]) is True


def test_load_reports_a_filter_changed_on_the_other_side(exp):
    path, _ = exp
    sc = bv.io.load_existing(path)
    sc.set_cells_to_keep([1, 2])
    bv.io.save(sc)
    with db.connect(db.db_path(path)) as con:
        db.set_cells_to_keep(con, np.array([1, 2, 3], dtype=np.uint32))
    with pytest.warns(UserWarning, match="differs"):
        back = bv.io.load_existing(path)
    # obs.to_keep is the filter's single source of truth
    assert back.cells_to_keep.tolist() == [1, 2, 3]


def test_half_a_memory_file_is_an_error(exp):
    path, _ = exp
    (path / "memory_py.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        bv.io.load_existing(path)
