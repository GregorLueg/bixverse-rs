"""SCENIC parameters, written by hand.

R's `params_scenic()` picks its learner block at call time, so its spec was
left out of the generator. `ScenicParams` holds the learner as one of the
three generated learner bundles and `to_rust` splices its fields in after the
top-level ones, which is the flat list R builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from beartype import BeartypeConf, beartype

from bixverse.params._checks import check_range
from bixverse.params._generated import (
    ScenicExtraTreesDefaults,
    ScenicGradientBoostingDefaults,
    ScenicRandomForestDefaults,
)

###########
# Globals #
###########

LearnerParams = (
    ScenicRandomForestDefaults
    | ScenicExtraTreesDefaults
    | ScenicGradientBoostingDefaults
)

LEARNERS: dict[str, type] = {
    "randomforest": ScenicRandomForestDefaults,
    "extratrees": ScenicExtraTreesDefaults,
    "grnboost2": ScenicGradientBoostingDefaults,
}

_CONF = BeartypeConf(is_pep484_tower=True)


##########
# Checks #
##########


@beartype
def _check_learner(learner_type: str, lp: LearnerParams) -> None:
    """The learner checks of R's `checkScenicParams`.

    Raises:
        ValueError: On a value outside its range.
    """
    check_range("min_samples_leaf", lp.min_samples_leaf, 1, None, False, True)
    check_range("n_features_split", lp.n_features_split, 0, None, False, True)
    check_range("max_depth", lp.max_depth, 1, None, False, True)
    match lp:
        case ScenicRandomForestDefaults() | ScenicExtraTreesDefaults():
            check_range("n_trees", lp.n_trees, 1, None, False, True)
            check_range("subsample_frac", lp.subsample_frac, 0, 1, True, False)
            if isinstance(lp, ScenicRandomForestDefaults):
                check_range("subsample_rate", lp.subsample_rate, 0, 1, True, False)
            else:
                check_range("n_thresholds", lp.n_thresholds, 1, None, False, True)
        case ScenicGradientBoostingDefaults():
            check_range("n_trees_max", lp.n_trees_max, 1, None, False, True)
            check_range("learning_rate", lp.learning_rate, 0, 1, True, False)
            check_range("early_stop_window", lp.early_stop_window, 1, None, False, True)
            check_range("subsample_rate", lp.subsample_rate, 0, 1, True, False)
        case _:
            raise TypeError(f"Unknown learner block for '{learner_type}'.")


#################
# ScenicParams #
#################


@beartype(conf=_CONF)
@dataclass(frozen=True, kw_only=True, slots=True)
class ScenicParams:
    """Constructor for SCENIC parameters, R's `params_scenic()`.

    Attributes:
        min_counts: Minimum total counts a gene needs to be included.
        min_cells: Minimum proportion of cells, in `(0, 1]`, that must express a
            gene for it to be considered.
        learner_type: Regression learner to use.
        gene_batch_strategy: Strategy for grouping target genes into batches.
            Only used for `"randomforest"` and `"extratrees"`.
        gene_batch_size: Genes per batch. `None` picks it automatically.
            Ignored for `"grnboost2"`.
        n_pcs: PCs for the correlated gene batch strategy.
        n_subsample: Cell subsampling threshold of the correlated strategy: at or
            above it, that many cells are drawn before the randomised SVD.
        learner_params: The learner block, of the class `learner_type` names.
            `None` takes that class's defaults, as R's `learner_params = list()`.
    """

    min_counts: int = 50
    min_cells: float = 0.03
    learner_type: Literal["randomforest", "extratrees", "grnboost2"] = "randomforest"
    gene_batch_strategy: Literal["correlated", "random"] = "correlated"
    gene_batch_size: int | None = None
    n_pcs: int = 50
    n_subsample: int = 100000
    learner_params: LearnerParams | None = None

    def __post_init__(self) -> None:
        check_range("min_counts", self.min_counts, 1, None, False, True)
        check_range("min_cells", self.min_cells, 0, 1, True, False)
        check_range("gene_batch_size", self.gene_batch_size, 1, None, False, True)
        check_range("n_pcs", self.n_pcs, 1, None, False, True)
        check_range("n_subsample", self.n_subsample, 1, None, False, True)
        want = LEARNERS[self.learner_type]
        lp = self.learner_params
        if lp is None:
            object.__setattr__(self, "learner_params", want())
        elif not isinstance(lp, want):
            raise ValueError(
                f"learner_type = '{self.learner_type}' needs {want.__name__} as "
                f"learner_params, got {type(lp).__name__}."
            )
        assert self.learner_params is not None
        _check_learner(self.learner_type, self.learner_params)

    @property
    def learner(self) -> LearnerParams:
        """The resolved learner block."""
        assert self.learner_params is not None
        return self.learner_params
