"""SACC-v1.1 ladder repair. Prereg: runs/dev/preregs/sacc_v11_ladder_prereg.md

`src/relaware/method/edits.py` is inside the SACC-Core v1 manifest and is NOT
modified. This module wraps it: detection, scope, priority, overlap resolution
and the right-to-left replacement pass are the frozen ones. Only the mapping
from a cue to its replacement string changes, in three ways:

  1. L1 entries for role nouns v1 had none for, so they generalize one rung
     instead of being wiped to the attribute-free L2 phrase.
  2. L1 entries for the relationship VALUES. v1 sent `single` to
     "a person close to them", which contradicts the source; v1.1 sends it to
     "not currently married", which is true of single and also covers divorced
     and widowed.
  3. KEEP (level 0) for lexicon surfaces that are not role words at all --
     `the court`, `patient`, `at work`, `day job`, `my job`, `my career`.
     No truthful occupation superordinate exists for them, and v1 wrote
     "the Court of Appeal's decision" -> "their professional activity of
     Appeal's decision". The ladder must not fire where it cannot stay true.
"""
import re
from . import edits

# 1. role nouns missing from _OCC_L1, each an immediate superordinate
OCC_L1_ADD = [
    (r"\bchef\b|\bbarista\b|\bcook\b", "a food service professional"),
    (r"\bstudent\b", "a person in education"),
    (r"\bresident\b|\battending\b|\btherapist\b|\bparamedic\b",
     "a healthcare professional"),
    (r"\bdesigner\b|\bartist\b|\bcurator\b|\bwriter\b", "a creative professional"),
    (r"\barchitect\b", "a design professional"),
    (r"\blibrarian\b", "an information professional"),
    (r"\bmanager\b|\bsupervisor\b", "a management professional"),
    (r"\btechnician\b|\banalyst\b|\bdeveloper\b", "a technical professional"),
    (r"\bconsultant\b", "a professional adviser"),
    (r"\bresearcher\b|\bscientist\b", "a research professional"),
    (r"freelanc\w+|self-employed", "independently employed"),
]

# 2. relationship VALUES, absent from _KIN_L1 and wrongly wiped by v1
REL_L1_ADD = {
    "single": "not currently married",        # true of single; also covers
                                              # divorced and widowed -> coarser
    "engaged": "in a committed relationship",  # covers engaged and married
    "spouse": "partner",                       # partner covers spouse
}

# 3. lexicon surfaces with no truthful occupation superordinate -> KEEP
NO_TRUE_SUPERORDINATE = re.compile(
    r"^(the court|court|patient|patients|at work|day job|my (job|career|"
    r"clients|patients|students|colleagues|boss|office|shift|company)|"
    r"i work (as|at|in)|as a laser)$", re.I)


def replacement(attribute, cue, level, next_word=""):
    """v1.1 replacement. Falls through to the frozen v1 rule when no repair
    applies, so every unrepaired cue is byte-identical to v1."""
    if level == 0:
        return cue["surface"]
    surf = cue["surface"]
    low = surf.strip().lower()

    if attribute == "OCCUPATION":
        if NO_TRUE_SUPERORDINATE.match(low):
            return surf                                   # rule 3: keep
        v1 = edits.replacement(attribute, cue, level, next_word)
        if v1 == edits._OCC_L2:                           # v1 fell through
            for pat, rep in OCC_L1_ADD:                   # rule 1
                if re.search(pat, low, re.I):
                    return rep
            return surf     # genuine role word we have no entry for: keep,
                            # rather than wipe the attribute
        return v1

    if attribute == "RELATIONSHIP":
        v1 = edits.replacement(attribute, cue, level, next_word)
        if v1 == edits._KIN_L2:                           # v1 fell through
            for k, rep in REL_L1_ADD.items():             # rule 2
                if k in low:
                    return rep
            return surf
        return v1

    return edits.replacement(attribute, cue, level, next_word)
