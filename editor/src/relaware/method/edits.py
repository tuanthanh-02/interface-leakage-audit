"""Edit application: generalization ladders over cue groups (plan §12.1 step 4).

Every cue group has at most three candidate levels (§16.1: final candidates
per cue group <= 3):

    L0  keep       -- no change
    L1  generalize -- replace each cue with a coarser truthful phrase
    L2  neutralize -- replace each cue with an attribute-free phrase

Replacements are truthful generalizations (INTACT's constraint, which the
plan inherits for the GENERALIZE policy): "solicitor" -> "a legal
professional" -> "a professional"; "his wife" -> "a family member" -> "a
person close to them". SEX cues are pronouns/titles, where L1 already has to
be the neutral form -- there is no truthful midpoint between "he" and "they"
-- so for SEX, L1 == L2 == neutral pronouns and the ladder is effectively
two levels.

The editor works on (start, end) spans and applies all replacements in one
right-to-left pass, so offsets never shift under it.
"""
import re

PRONOUN_MAP = {
    "he": "they", "she": "they", "him": "them", "her": "them",
    "his": "their", "hers": "theirs", "himself": "themselves",
    "herself": "themselves",
}
TITLES = {"mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss"}

_KIN_L1 = {"wife": "spouse", "husband": "spouse", "widow": "bereaved spouse",
           "widower": "bereaved spouse", "married": "in a long-term "
           "relationship", "marriage": "long-term relationship",
           "divorced": "formerly in a relationship",
           "divorce": "end of a relationship", "fiancee": "partner",
           "fiance": "partner", "the couple": "the two individuals",
           "wedding": "family event", "partner": "partner"}
_KIN_L2 = "a person close to them"

_OCC_L1 = [
    (r"lawyer|solicitor|barrister|attorney|notary|legal (?:adviser|practice|profession)",
     "a legal professional"),
    (r"judge|magistrate", "a judicial officer"),
    (r"prosecutor", "a legal official"),
    (r"law firm|chambers", "a professional practice"),
    (r"doctor|physician|surgeon|nurse", "a healthcare professional"),
    (r"hospital|clinic", "a workplace"),
    (r"teacher|professor", "an education professional"),
    (r"school|university", "an institution"),
    (r"police|military|army|officer", "a public servant"),
    (r"minister|government|diplomat|politician", "a public official"),
    (r"journalist", "a media professional"),
    (r"engineer", "a technical professional"),
    (r"business(?:man|woman)?|company", "a commercial occupation"),
    (r"farm(?:er)?", "a manual occupation"),
    (r"unemploy(?:ed|ment)", "not in formal employment"),
    (r"retire[dment]*|pension(?:er)?", "no longer working"),
    (r"represent(?:ed|ing) (?:the|his|her|a)", "acted for a"),
    (r"patients? of", "clients of"),
]
_OCC_L2 = "their professional activity"


# "her" is the one form that is both objective ("saw her") and possessive
# ("her husband"); the male forms split into him/his so a blind map degrades
# female-subject texts more -- exactly the asymmetry a SEX-balanced utility
# score would then mismeasure. Heuristic: possessive when the next word is
# not a function word.
_HER_OBJECTIVE_NEXT = {"that", "to", "of", "and", "or", "in", "on", "at",
                       "for", "with", "by", "about", "as", "from", "into",
                       "was", "is", "had", "has", "the", "a", "an"}

_KIN_NEUTRAL = {"husband": "spouse", "wife": "spouse", "widow": "bereaved "
                "spouse", "widower": "bereaved spouse", "father": "parent",
                "mother": "parent", "son": "child", "daughter": "child",
                "brother": "sibling", "sister": "sibling"}


def _sex_replacement(surface, next_word=""):
    low = surface.lower().rstrip(".")
    if low in TITLES:
        return ""                       # drop the title, keep the name
    if low in _KIN_NEUTRAL:             # gendered kinship noun (cue kind
        rep = _KIN_NEUTRAL[low]         # "gendered_kin")
    elif low == "her":
        nxt = next_word.lower()
        rep = "them" if (not nxt or not nxt.isalpha()
                         or nxt in _HER_OBJECTIVE_NEXT) else "their"
    else:
        rep = PRONOUN_MAP.get(low, "they")
    if surface[0].isupper():
        rep = rep.capitalize()
    return rep


def _kin_replacement(surface, level):
    if level >= 2:
        return _KIN_L2
    low = surface.lower()
    for k, v in _KIN_L1.items():
        if k in low:
            return v
    return _KIN_L2


def _occ_replacement(surface, level):
    if level >= 2:
        return _OCC_L2
    low = surface.lower()
    for pat, rep in _OCC_L1:
        if re.search(pat, low):
            return rep
    return _OCC_L2


def replacement(attribute, cue, level, next_word=""):
    """The replacement string for one cue at one ladder level."""
    if level == 0:
        return cue["surface"]
    if attribute == "SEX":
        return _sex_replacement(cue["surface"], next_word)
    if attribute == "RELATIONSHIP":
        return _kin_replacement(cue["surface"], level)
    if attribute == "OCCUPATION":
        return _occ_replacement(cue["surface"], level)
    if attribute == "MENTION":
        return cue["surface"] if level == 0 else "the individual"
    return cue["surface"]


# priority when spans from different groups overlap: the more specific
# generalization wins -- a RELATIONSHIP edit of "wife" already removes the
# gender, so it outranks the SEX edit of the same span
_ATTR_PRIORITY = {"RELATIONSHIP": 0, "OCCUPATION": 1, "SEX": 2, "MENTION": 3}


def _next_word(text, pos):
    m = re.match(r"\s*([A-Za-z']+)", text[pos:pos + 30])
    return m.group(1) if m else ""


def apply_bundle(text, groups, levels, base=()):
    """Apply one edit bundle. groups: {attr: [cue...]}; levels: {attr: 0|1|2}.

    `base` is a list of (start, end, literal) replacements applied at highest
    priority -- Phase 1 uses it for the gold-span masking every method arm
    shares, so Pronoun Closure really is Span-Only plus pronouns, not a
    different denominator.

    Cues are deduped within each group at detection time, but SEX and
    RELATIONSHIP (and base spans) can still claim overlapping spans across
    groups; a single overlap sweep here keeps exactly one edit per region --
    base first, then by _ATTR_PRIORITY, longer span first. Applied
    right-to-left so offsets never shift."""
    edits = [(s0, e0, rep, -1, e0 - s0) for s0, e0, rep in base]
    for attr, cues in groups.items():
        lv = levels.get(attr, 0)
        if lv == 0:
            continue
        for c in cues:
            edits.append((c["start"], c["end"],
                          replacement(attr, c, lv, _next_word(text, c["end"])),
                          _ATTR_PRIORITY.get(attr, 9), c["end"] - c["start"]))
    # overlap sweep: order by priority then length, accept if disjoint
    accepted = []
    for s0, e0, rep, pri, ln in sorted(edits, key=lambda x: (x[3], -x[4])):
        if all(e0 <= a or s0 >= b for a, b, _ in accepted):
            accepted.append((s0, e0, rep))
    out = text
    for s0, e0, rep in sorted(accepted, key=lambda x: -x[0]):
        if rep == "" and e0 < len(out) and out[e0:e0 + 1] == " ":
            e0 += 1                     # collapse the space a title leaves
        if rep[:2].lower() in ("a ", "an") and re.search(
                r"\b(a|an|the)\s+$", out[:s0], re.I):
            rep = rep.split(" ", 1)[1]  # "was a" + "a legal ..." -> one article
        out = out[:s0] + rep + out[e0:]
    return out
