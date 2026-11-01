"""Pipelines without R: construction, printing, validation, step order and
the per group runs.

Parity with the R pipelines is in `test_pipeline_r.py`.
"""

import inspect

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintParamViolation

import bixverse as bv
from bixverse import graph, integrate, pipeline, pp
from bixverse.params import (
    BtMetacellsParams,
    MinQualityParams,
    NeighboursParams,
    SyntheticDataParams,
)

QC = MinQualityParams(min_unique_genes=1, min_lib_size=1, min_cells=1)
BT = BtMetacellsParams(target_no_metacells=8, k=8)
PREP = (
    pipeline.step_hvg(hvg_no=30) >> pipeline.step_pca(8) >> pipeline.step_neighbours()
)


@pytest.fixture(scope="module")
def data():
    """A small synthetic experiment."""
    return bv.io.generate_single_cell_test_data(
        SyntheticDataParams(n_cells=240, n_genes=60, n_batches=2), seed=5
    )


@pytest.fixture
def sc(data, tmp_path):
    """A fresh experiment with the first ten cells filtered out."""
    sc = bv.io.load_csr(tmp_path / "sc", data.counts, data.obs, data.var, qc=QC)
    sc.set_cells_to_keep(sc.cell_ids()[10:])
    return sc


################
# Construction #
################


def test_rshift_keeps_the_step_order():
    p = pipeline.step_hvg() >> pipeline.step_pca(10)
    p = p >> pipeline.step_neighbours() >> pipeline.step_clusters()
    assert isinstance(p, pipeline.Pipeline)
    assert len(p) == 4
    assert [s.name for s in p.steps] == ["hvg", "pca", "neighbours", "clusters"]
    assert len(pipeline.Pipeline()) == 0
    with pytest.raises(BeartypeCallHintParamViolation):
        _ = p >> "not_a_step"  # ty: ignore[unsupported-operator]


def test_list_and_rshift_build_the_same_pipeline():
    chained = pipeline.step_hvg(hvg_no=20) >> pipeline.step_pca(5)
    listed = pipeline.Pipeline([pipeline.step_hvg(hvg_no=20), pipeline.step_pca(5)])
    assert listed == chained
    assert isinstance(listed.steps, tuple)
    assert repr(listed) == repr(chained)
    with pytest.raises((TypeError, BeartypeCallHintParamViolation)):
        pipeline.Pipeline([pipeline.step_hvg(), "not_a_step"])  # ty: ignore[invalid-argument-type]


def test_step_captures_its_arguments():
    s = pipeline.step_hvg(hvg_no=30)
    assert s.name == "hvg"
    assert s.fn is pp.find_hvg
    assert s.kwargs["hvg_no"] == 30
    assert s.accepts == (bv.SingleCells, bv.SingleCellsSubset, bv.MetaCells)
    assert s.returns is None
    mc = pipeline.step_metacells("bootstrapped", params=BT)
    assert mc.name == "metacells (bootstrapped)"
    assert mc.accepts == (bv.SingleCells, bv.SingleCellsSubset)
    assert mc.returns is bv.MetaCells


def test_step_arguments_bind_up_front():
    with pytest.raises(TypeError):
        pipeline.step_metacells("bootstrapped", target_sizes=1e4)
    # SEACells has no default for its parameters
    with pytest.raises(TypeError):
        pipeline.step_metacells("seacells")
    with pytest.raises(TypeError):
        pipeline.Step("pca", pp.calculate_pca, {"n_pcs": 10})
    with pytest.raises(ValueError, match="Step classes"):
        pipeline.Step("x", pp.find_hvg, accepts=(str,))
    with pytest.raises(ValueError, match="Step classes"):
        pipeline.Step("x", pp.find_hvg, accepts=())


@pytest.mark.parametrize(
    ("step", "fn", "skip"),
    [
        (pipeline.step_hvg, pp.find_hvg, ()),
        (pipeline.step_pca, pp.calculate_pca, ("no_pcs",)),
        (pipeline.step_neighbours, graph.find_neighbours, ()),
        (pipeline.step_clusters, graph.find_clusters, ()),
        (pipeline.step_harmony, integrate.harmony, ()),
        (pipeline.step_harmony_v2, integrate.harmony_v2, ()),
        (pipeline.step_bbknn, integrate.bbknn, ()),
        (pipeline.step_fast_mnn, integrate.fast_mnn, ()),
    ],
)
def test_step_defaults_match_their_functions(step, fn, skip):
    want = inspect.signature(fn).parameters
    got = inspect.signature(step).parameters
    assert list(got) == [p for p in want if p != "sc"]
    for name, p in got.items():
        if name not in skip:
            assert p.default == want[name].default, name


def test_repr_matches_r_layout():
    s = pipeline.step_hvg(hvg_no=30)
    assert repr(s) == (
        "<Step> hvg(hvg_no=30, hvg_params=None, streaming=None, verbose=False)"
    )
    p = pipeline.step_hvg(hvg_no=30) >> pipeline.step_metacells(
        "bootstrapped", params=BT, target_size=1e4
    )
    assert repr(p).splitlines() == [
        "<Pipeline> 2 steps",
        "  1. hvg                       "
        "hvg_no=30, hvg_params=None, streaming=None, verbose=False",
        "  2. metacells (bootstrapped)  "
        "params=<BtMetacellsParams>, target_size=10000.0",
    ]
    one = pipeline.Pipeline((pipeline.step_harmony("batch_index"),))
    assert repr(one).splitlines()[0] == "<Pipeline> 1 step"
    assert repr(pipeline.Pipeline()) == "<Pipeline> 0 steps\n  (empty)"
    assert "hvg=[0, 1, 2]" in repr(pipeline.step_pca(10, hvg=[0, 1, 2]))
    assert "hvg=<ndarray>" in repr(pipeline.step_pca(10, hvg=np.arange(5)))


##############
# Validation #
##############


def test_validation_tracks_the_class():
    p = PREP >> pipeline.step_metacells("bootstrapped") >> pipeline.step_clusters()
    assert pipeline.validate_pipeline(p, bv.SingleCells) is bv.MetaCells
    assert pipeline.validate_pipeline(PREP, bv.MetaCells) is bv.MetaCells
    assert pipeline.validate_pipeline(pipeline.Pipeline(), bv.SingleCellsSubset) is (
        bv.SingleCellsSubset
    )
    with pytest.raises(TypeError, match=r"Step 5 \('harmony'\) cannot run on a"):
        pipeline.validate_pipeline(
            PREP
            >> pipeline.step_metacells("bootstrapped")
            >> pipeline.step_harmony("batch_index"),
            bv.SingleCells,
        )
    with pytest.raises(TypeError, match="Step 1"):
        pipeline.validate_pipeline(
            pipeline.Pipeline((pipeline.step_metacells("bootstrapped"),)),
            bv.MetaCells,
        )
    with pytest.raises(ValueError, match="cls must be one of"):
        pipeline.validate_pipeline(PREP, str)


#########
# Apply #
#########


def test_apply_runs_the_steps_in_order(sc):
    seen = []

    def record(obj: bv.SingleCells, tag: str) -> None:
        seen.append((obj, tag))

    p = pipeline.Step("a", record, {"tag": "a"}) >> pipeline.Step(
        "b", record, {"tag": "b"}
    )
    assert pipeline.apply_pipeline(p, sc) is sc
    assert seen == [(sc, "a"), (sc, "b")]
    assert pipeline.apply_pipeline(pipeline.Pipeline(), sc) is sc


def test_apply_validates_before_running(sc):
    p = (
        pipeline.step_hvg(hvg_no=30)
        >> pipeline.step_metacells("bootstrapped")
        >> pipeline.step_harmony("batch_index")
    )
    with pytest.raises(TypeError):
        pipeline.apply_pipeline(p, sc)
    assert sc.map.hvg is None


def test_apply_matches_the_direct_calls(data, sc, tmp_path):
    p = PREP >> pipeline.step_clusters(name="pipe_leiden")
    pipeline.apply_pipeline(p, sc)
    direct = bv.io.load_csr(tmp_path / "direct", data.counts, data.obs, data.var, qc=QC)
    direct.set_cells_to_keep(direct.cell_ids()[10:])
    pp.find_hvg(direct, hvg_no=30)
    pp.calculate_pca(direct, 8)
    graph.find_neighbours(direct)
    graph.find_clusters(direct, name="pipe_leiden")
    assert sc.map.hvg is not None and direct.map.hvg is not None
    np.testing.assert_array_equal(sc.map.hvg, direct.map.hvg)
    np.testing.assert_array_equal(sc.cache.pca_factors, direct.cache.pca_factors)
    assert sc.obs()["pipe_leiden"].to_list() == direct.obs()["pipe_leiden"].to_list()


def test_apply_hands_the_meta_cells_on(sc):
    p = (
        PREP
        >> pipeline.step_metacells(
            "bootstrapped", params=BtMetacellsParams(target_no_metacells=30, k=8)
        )
        >> pipeline.step_hvg(hvg_no=20)
        >> pipeline.step_pca(5)
        >> pipeline.step_neighbours(neighbours_params=NeighboursParams(k=5))
        >> pipeline.step_clusters(name="mc_leiden")
    )
    mc = pipeline.apply_pipeline(p, sc)
    assert isinstance(mc, bv.MetaCells)
    assert mc.hvg is not None and len(mc.hvg) == 20
    assert mc.obs()["mc_leiden"].null_count() == 0
    assert sc.map.hvg is not None and len(sc.map.hvg) == 30
    assert "mc_leiden" not in sc.obs_columns()


def test_apply_refuses_a_generator_that_returns_none(sc):
    p = pipeline.Pipeline((pipeline.step_metacells("bootstrapped", params=BT),))
    with pytest.warns(UserWarning), pytest.raises(TypeError, match="NoneType"):
        pipeline.apply_pipeline(p, sc)


def test_multimodal_counts_as_single_cells():
    assert issubclass(bv.SingleCellsMultiModal, bv.SingleCells)
    assert pipeline.validate_pipeline(PREP, bv.SingleCells) is bv.SingleCells


#############
# Per group #
#############


def test_per_group_runs_on_every_level(sc):
    levels = (
        sc.obs(cols=["cell_grp"], filtered=True)["cell_grp"]
        .unique(maintain_order=True)
        .to_list()
    )
    out = pipeline.apply_pipeline_per_group(
        pipeline.Pipeline((pipeline.step_hvg(hvg_no=20),)), sc, "cell_grp"
    )
    assert list(out) == levels
    for g, sub in out.items():
        assert isinstance(sub, bv.SingleCellsSubset)
        assert sub.group == g
        assert sub.map.hvg is not None and len(sub.map.hvg) == 20
    assert sc.map.hvg is None

    restricted = pipeline.apply_pipeline_per_group(
        pipeline.Pipeline((pipeline.step_hvg(hvg_no=20),)),
        sc,
        "cell_grp",
        groups=levels[:1],
    )
    assert list(restricted) == levels[:1]


def test_per_group_on_a_numeric_column(sc):
    # R's `apply_pipeline_per_group` refuses anything but a string column
    out = pipeline.apply_pipeline_per_group(
        pipeline.Pipeline((pipeline.step_hvg(hvg_no=20),)), sc, "batch_index"
    )
    assert list(out) == ["1", "2"]
    n = sum(sub.shape[0] for sub in out.values())
    assert n == len(sc.cells_to_keep)


def test_per_group_errors_up_front(sc):
    with pytest.raises(ValueError, match="not found in obs"):
        pipeline.apply_pipeline_per_group(PREP, sc, "not_a_column")
    with pytest.raises(TypeError, match="cannot run on a"):
        pipeline.apply_pipeline_per_group(
            pipeline.step_metacells("bootstrapped") >> pipeline.step_harmony("b"),
            sc,
            "cell_grp",
        )


def test_per_group_meta_cells_merge(sc):
    p = PREP >> pipeline.step_metacells("bootstrapped", params=BT)
    out = pipeline.apply_pipeline_per_group(p, sc, "cell_grp")
    assert all(isinstance(mc, bv.MetaCells) for mc in out.values())
    merged = bv.metacells.merge_meta_cells(out)  # ty: ignore[invalid-argument-type]
    direct = pipeline.meta_cells_per_group(
        sc, "cell_grp", mc_params={"params": BT}, pipeline=PREP
    )
    assert merged.obs().equals(direct.obs())
    np.testing.assert_array_equal(merged.raw.data, direct.raw.data)
    assert bv.metacells.meta_cells_per_group is pipeline.meta_cells_per_group


def test_meta_cells_per_group_errors_up_front(sc):
    with pytest.raises(TypeError, match="must return a `SingleCellsSubset`"):
        pipeline.meta_cells_per_group(
            sc, "cell_grp", pipeline=PREP >> pipeline.step_metacells("bootstrapped")
        )
    with pytest.raises(TypeError):
        pipeline.meta_cells_per_group(sc, "cell_grp", mc_params={"param": BT})
    with pytest.raises(ValueError, match="not found in obs"):
        pipeline.meta_cells_per_group(sc, "not_a_column", pipeline=PREP)
    assert sc.map.hvg is None
