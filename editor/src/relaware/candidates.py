"""Frozen evaluator candidate sets C_eval (plan §6.4).

One fixed option list per (subject, attribute), identical across every method,
placement and state. Frozen in Phase 0 and stored inside the registry file.
The method never sees these; its own candidates come from the public ontology
in method/edits.py.

Composition rule: the gold value plus distractors drawn from the global gold
value pool, at least one from every stratum, so the option list's composition
does not itself give the answer away. Deterministic per (subject, attribute):
seeded by a hash, no global RNG state.
"""
import os
import hashlib
import random

from . import strata

N_OPTIONS = {"SEX": 2, "RELATIONSHIP": 5, "OCCUPATION": 8}
# candidate order repetitions (plan §18 Tier A/D). Default 3 is the frozen
# instrument; RELAWARE_N_PERM=8 is the Phase-5 / final-table setting (Task C
# of the follow-on plan). Changing it changes absolute L values, so it must
# never differ between two arms being compared -- setting it via environment
# guarantees that within one run.
N_PERM = int(os.environ.get("RELAWARE_N_PERM", "3"))

# Fixed REL option list: the three observed gold values plus two distractor
# statuses, so a "married" gold cannot be inferred from a 3-option list.
REL_OPTIONS = ("married", "widowed", "divorced", "single",
               "in a relationship", "engaged")


def _rng(*key):
    h = hashlib.sha256("\x00".join(str(k) for k in key).encode()).hexdigest()
    return random.Random(h)


def occ_pool(reg_entries):
    """Global OCCUPATION value pool grouped by OCC-4 stratum, from gold only."""
    pool = {}
    for e in reg_entries:
        if e["attribute"] == "OCCUPATION" and e["stratum"]["OCC-4"]:
            pool.setdefault(e["stratum"]["OCC-4"], set()).add(e["gold_value"])
    return {k: sorted(v) for k, v in pool.items()}


def build_for(entry, pool):
    """The frozen option list for one registry entry (gold value included,
    position determined later by the permutation, never fixed at index 0
    in the stored artifact)."""
    a = entry["attribute"]
    if a == "SEX":
        opts = list(strata.SEX_VALUES)
    elif a == "RELATIONSHIP":
        # the frozen option list must contain every gold value the corpus
        # actually uses, or the gold is absent and the item unscoreable
        opts = sorted(set(REL_OPTIONS) | {entry["gold_value"]})
    else:
        # Balanced composition: with 8 options over the 4 OCC strata, every
        # stratum contributes exactly 2 (the gold counting toward its own
        # stratum's quota). An unbalanced list leaks by elimination: if the
        # gold's stratum always held >= 2 options while others could hold 1,
        # a singleton stratum would provably not contain the gold.
        gold = entry["gold_value"]
        rng = _rng("cand", entry["eval_subject_id"], a)
        strata_names = sorted(pool)
        quota, rem = divmod(N_OPTIONS[a], len(strata_names))
        opts = {gold}
        gold_stratum = next((sn for sn in strata_names
                             if gold in pool[sn]), None)
        for i, sn in enumerate(strata_names):
            want = quota + (1 if i < rem else 0)
            have = 1 if sn == gold_stratum else 0
            others = [v for v in sorted(pool[sn]) if v != gold and v not in opts]
            rng.shuffle(others)
            for v in others[:max(0, want - have)]:
                opts.add(v)
        # a thin stratum may not fill its quota; top up from the global pool
        flat = sorted({v for vs in pool.values() for v in vs} - opts)
        rng.shuffle(flat)
        for v in flat:
            if len(opts) >= N_OPTIONS[a]:
                break
            opts.add(v)
        opts = sorted(opts)
    rng = _rng("order", entry["eval_subject_id"], a)
    rng.shuffle(opts)
    return opts


def permutations(entry, n_opts, n_perm=N_PERM):
    """N_PERM deterministic option orders for one entry."""
    out = []
    for p in range(n_perm):
        idx = list(range(n_opts))
        _rng("perm", entry["eval_subject_id"], entry["attribute"], p).shuffle(idx)
        out.append(idx)
    return out


def attach(reg):
    """Attach frozen candidate sets to every registry entry, in place."""
    pool = occ_pool(reg["entries"])
    for e in reg["entries"]:
        e["candidates"] = build_for(e, pool)
        if e["gold_value"] not in e["candidates"]:
            raise RuntimeError(f"gold value missing from candidates: {e}")
    return reg
