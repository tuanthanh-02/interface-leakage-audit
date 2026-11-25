"""Compatibility shim: the stratum ladders now live in profiles.py.

They moved because they are per-corpus (SynthPAI's RELATIONSHIP has six values
where TAB has three) and keeping two copies in sync by hand is how the earlier
dataset-detector bugs happened. Names re-exported here so existing imports keep
working; new code should import profiles directly.
"""
from .profiles import (MIN_DOC_N, MIN_PAIR_N, OCC_LADDER, OCC_RULES,  # noqa
                       REL_LADDERS, REL_TYPES, SEX_VALUES, check_strata,
                       occ_stratum)

# TAB's ladder, for callers written before profiles existed.
REL_LADDER = REL_LADDERS["tab"]
REL_VALUES = ("married", "widowed", "divorced")
