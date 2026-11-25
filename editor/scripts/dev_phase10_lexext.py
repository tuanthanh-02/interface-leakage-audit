#!/usr/bin/env python
"""Phase 10 -- the occupation lexicon extension (preregs/phase10_p4_lexicon_prereg.md).

Two halves, kept apart so their contributions never merge:

  X1  morphological variants generated MECHANICALLY from the frozen occ_role entries themselves.
      No observed failure was consulted, so this half is not fitted to anything.
  X2  role nouns absent from occ_role that appear in the DEV class-B evidence. This half IS
      fitted to development data and is labelled as such everywhere it is used.

`profiles.py` is frozen and not edited; the extension is a separate pattern applied at sweep time.
"""
import json, os, re, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(REPO) != "spia":
    REPO = os.path.join(REPO, "spia")
sys.path.insert(0, REPO)
from src.relaware import registry, profiles

# ---------------------------------------------------------------- X1, mechanical
_SUFFIX = [("er", ["ing", "ers", "ed"]), ("or", ["ing", "ors", "ion"]),
           ("ist", ["ists", "ism"]), ("ian", ["ians"]), ("ant", ["ants"]),
           ("ent", ["ents"]), ("man", ["men"]), ("woman", ["women"])]


def _base_terms(occ_role):
    """the plain word entries of the frozen regex, with its machinery stripped"""
    out = set()
    for x in re.split(r"[|()\\]", occ_role):
        x = (x or "").strip()
        if re.fullmatch(r"[a-z][a-z\-]{3,}", x):
            out.add(x)
    return sorted(out)


def x1_variants(occ_role):
    """-ing / -ion / plural / hashtag forms of what the lexicon already has"""
    v = set()
    for w in _base_terms(occ_role):
        for end, adds in _SUFFIX:
            if w.endswith(end):
                stem = w[: -len(end)]
                for a in adds:
                    cand = stem + a
                    if len(cand) > 4:
                        v.add(cand)
        v.add(w + "s")
        v.add("#" + w)          # PANORAMA and SynthPAI carry hashtags
        v.add("#" + w + "s")
    return sorted(v - set(_base_terms(occ_role)))


# ---------------------------------------------------------------- X2, fitted to DEV
# Role nouns the DEV class-B evidence showed the lexicon has no entry for at all.
# This list is FITTED to development observations and is never presented as mechanical.
X2_FITTED = ["editor", "publisher", "reporter", "correspondent", "developer", "programmer",
             "accountant", "auditor", "banker", "broker", "agent", "realtor", "surveyor",
             "electrician", "plumber", "mechanic", "carpenter", "driver", "pilot", "sailor",
             "cleaner", "cashier", "waiter", "waitress", "receptionist", "secretary",
             "assistant", "clerk", "translator", "interpreter", "coach", "trainer",
             "instructor", "tutor", "counsellor", "counselor", "midwife", "pharmacist",
             "dentist", "veterinarian", "psychologist", "economist", "statistician",
             "mathematician", "biologist", "chemist", "physicist", "geologist", "historian",
             "philosopher", "actor", "actress", "musician", "singer", "dancer", "painter",
             "photographer", "editor-in-chief", "headmaster", "principal", "dean", "warden"]


def extension_pattern(ds, include_x2=True):
    prof = profiles.for_registry(registry.load(TREE[ds] + "/registry.json"))
    terms = x1_variants(prof.lex["occ_role"])
    if include_x2:
        terms = terms + [w for w in X2_FITTED if w not in terms]
    terms = sorted(set(terms), key=lambda w: (-len(w), w))
    return r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", terms


TREE = {"synthpai": "runs/relaware_synthpai", "tab": "runs/relaware",
        "medqa": "runs/relaware_medqa", "panorama": "runs/relaware_panorama"}

if __name__ == "__main__":
    os.chdir(REPO)
    out = {}
    for ds in sorted(TREE):
        prof = profiles.for_registry(registry.load(TREE[ds] + "/registry.json"))
        base = _base_terms(prof.lex["occ_role"])
        x1 = x1_variants(prof.lex["occ_role"])
        x2 = [w for w in X2_FITTED if w not in x1 and w not in base]
        out[ds] = {"n_frozen_terms": len(base), "n_X1_mechanical": len(x1),
                   "n_X2_fitted": len(x2), "X1_sample": x1[:10], "X2_sample": x2[:10]}
        print(f"  {ds:9s} frozen {len(base):3d}  + X1 mechanical {len(x1):3d}"
              f"  + X2 fitted {len(x2):3d}")
        print(f"            X1 e.g. {x1[:6]}")
    p = "runs/dev/sacc_improvement/plan/lexicon_extension.json"
    json.dump({"note": "X1 is generated from the frozen lexicon's own entries and is not fitted; "
                       "X2 is fitted to DEV class-B observations and is labelled as such",
               "X2_fitted_terms": X2_FITTED, "per_corpus": out},
              open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)
    print("->", p)
