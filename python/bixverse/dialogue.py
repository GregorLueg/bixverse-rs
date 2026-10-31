"""DIALOGUE multicellular programmes over single cells, subsets and meta
cells, the port of the bixverse R `dialogue_sc`. See
`bixverse._core.dialogue`.
"""

from bixverse._core.dialogue import (
    DialogueResult,
    dialogue,
    get_params,
    get_results,
)

__all__ = ["DialogueResult", "dialogue", "get_params", "get_results"]
