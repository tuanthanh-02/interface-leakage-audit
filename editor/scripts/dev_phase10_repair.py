#!/usr/bin/env python
"""Phase 10 P1 -- the M1 replacer: SACC-v1's candidates and attribution, a repaired replacement.

Rules: runs/dev/sacc_improvement/plan/repair_rules.yaml (frozen before this file produced output).
Tests: runs/dev/sacc_improvement/unit/repair_unit_tests.jsonl  --  `--selftest` runs them.

Nothing in src/relaware/method/ is modified. v1's edits.py and v1.1's edits_v11.py are imported
read-only; only the mapping from a cue to (action, replacement, companion edits) is new, plus the
finite-verb agreement repair, which is the one edit M1 makes OUTSIDE a cue span and which is
recorded as such in the trace.
"""
import json, os, re, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(REPO) != "spia":
    REPO = os.path.join(REPO, "spia")
sys.path.insert(0, REPO)
from src.relaware.method import edits as V1
from src.relaware.method import edits_v11 as V11

MASK = "[REDACTED]"
NEGATORS = {"not", "never", "no", "without", "neither", "nor", "n't", "cannot"}
FINITE_MAP = {"is": "are", "was": "were", "has": "have", "does": "do", "'s": "'re",
              "isn't": "aren't", "wasn't": "weren't", "hasn't": "haven't", "doesn't": "don't"}
PHRASE = {"divorce proceedings": "legal proceedings", "divorce petition": "legal petition",
          "marriage certificate": "civil status certificate", "wedding ceremony": "family ceremony"}
REFLEXIVE = {"himself", "herself"}
_ART = re.compile(r"\b(a|an|the)\s+$", re.I)

_NLP = None
def nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def _cap_like(surface, rep):
    return rep[:1].upper() + rep[1:] if surface[:1].isupper() and rep else rep


def _span_tokens(doc, s0, e0):
    return [t for t in doc if t.idx < e0 and t.idx + len(t.text) > s0]


def _negated(tok):
    """A negation scope over the cue. Conservative: over-triggering masks instead of
    generalising, which is the safe direction for a truth guard."""
    node, seen = tok, 0
    while seen < 12:
        if node.lower_ in NEGATORS:
            return True
        for ch in node.children:
            if ch.dep_ == "neg" or ch.lower_ in NEGATORS:
                return True
        if node.head is node:
            break
        node, seen = node.head, seen + 1
    return False


def _pronoun_rep(tok, surface):
    low = surface.strip().lower()
    if low in REFLEXIVE:
        return "themselves"
    if tok is not None and tok.tag_ == "PRP$":
        return "their"
    if low == "hers" or (low == "his" and tok is not None and tok.tag_ == "PRP"):
        return "theirs"
    if low in ("his", "her") and tok is None:
        return "their"
    nom = tok is not None and (tok.dep_ in ("nsubj", "nsubjpass") or "Nom" in tok.morph.get("Case"))
    return "they" if nom else "them"


def agreement_edits(tok):
    """The finite verb governed by a neutralised nominative pronoun. Outside the cue span."""
    if tok is None or tok.dep_ not in ("nsubj", "nsubjpass"):
        return []
    head = tok.head
    auxes = [c for c in head.children if c.dep_ in ("aux", "auxpass")]
    if any(c.dep_ == "conj" for c in head.children) and len(
            [c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")]) > 1:
        return []                                   # verb shared by a coordination: skip
    out = []
    for c in [head] + auxes:
        lw = c.lower_
        if lw in FINITE_MAP and c.tag_ in ("VBZ", "VBD"):
            out.append((c.idx, c.idx + len(c.text), FINITE_MAP[lw], "finite_verb_agreement"))
        elif c is head and not auxes and c.tag_ == "VBZ" and lw not in FINITE_MAP:
            out.append((c.idx, c.idx + len(c.text), c.lemma_, "finite_verb_agreement.vbz_lemma"))
    return out


def plan_cue(text, doc, attribute, cue, _depth=0):
    """-> dict(action, replacement, start, end, rule, companion=[...]) for ONE cue.

    action is REPLACE / MASK / KEEP. KEEP is not protection and is counted separately."""
    s0, e0 = cue["start"], cue["end"]
    surface = text[s0:e0]
    toks = _span_tokens(doc, s0, e0)
    head_tok = toks[0] if toks else None
    low = surface.strip().lower()
    out = dict(start=s0, end=e0, surface=surface, attribute=attribute, companion=[], rule=None)

    # -- verified phrase replacements come first (fallback order, step 1)
    if low in PHRASE:
        return dict(out, action="REPLACE", replacement=_cap_like(surface, PHRASE[low]),
                    rule="phrase_replacements")

    # -- possessive determiner inside the span: keep the link, generalise only the rest
    if _depth == 0 and len(toks) > 1 and toks[0].tag_ == "PRP$" and attribute in ("SEX", "RELATIONSHIP"):
        det = "their" if toks[0].lower_ in ("his", "her") else toks[0].text
        sub = plan_cue(text, doc, attribute,
                       {"surface": text[toks[1].idx:e0], "start": toks[1].idx, "end": e0},
                       _depth=1)
        rest = sub["replacement"] if sub["action"] != "KEEP" else sub["surface"]
        return dict(out, action=sub["action"] if sub["action"] != "KEEP" else "REPLACE",
                    replacement=_cap_like(surface, f"{det} {rest.strip()}"),
                    rule="possessive_relation+" + str(sub["rule"]))

    if attribute == "SEX":
        if low.rstrip(".") in V1.TITLES:
            return dict(out, action="REPLACE", replacement="", rule="title_drop")
        if low in V1._KIN_NEUTRAL:
            return dict(out, action="REPLACE",
                        replacement=_cap_like(surface, V1._KIN_NEUTRAL[low]), rule="kin_neutral")
        rep = _pronoun_rep(head_tok, surface)
        comp = agreement_edits(head_tok) if rep == "they" else []
        return dict(out, action="REPLACE", replacement=_cap_like(surface, rep),
                    rule="pronoun_case", companion=comp)

    if attribute in ("OCCUPATION", "RELATIONSHIP"):
        if attribute == "OCCUPATION" and V11.NO_TRUE_SUPERORDINATE.match(low):
            return dict(out, action="KEEP", replacement=surface, rule="no_true_superordinate")
        if head_tok is not None and _negated(head_tok):
            return dict(out, action="MASK", replacement=MASK, rule="negation_guard")
        rep = V11.replacement(attribute, cue, 1, V1._next_word(text, e0))
        fell_through = rep in (V1._OCC_L2, V1._KIN_L2) or rep == surface
        if fell_through:
            return dict(out, action="MASK", replacement=MASK, rule="fall_through_removed")
        return dict(out, action="REPLACE", replacement=_cap_like(surface, rep), rule="v11_ladder")

    if attribute == "MENTION":
        return dict(out, action="REPLACE", replacement="the individual", rule="mention")
    return dict(out, action="KEEP", replacement=surface, rule="unhandled_attribute")


def apply_edits(text, planned):
    """Right-to-left and disjoint. Article handling, per repair_rules.yaml span_expansion:

      preceded by "the"  keep the definite article and drop the replacement's own
                         ("the police" -> "the public servant", not "a public servant":
                          rewriting it would turn a specific body into an indefinite one)
      preceded by a/an   rewrite that article so it agrees with the replacement
                         ("an engineer" -> "a technical professional", never "an technical")

    Capitalisation is re-derived from the character the APPLIED span starts at, not from the cue
    surface: expanding onto a lower-case article must not produce "to A public official".
    The applied span is written back into the plan so the trace records what was really edited."""
    ed = []
    for p in planned:
        p["applied"] = None
        if p["action"] == "KEEP":
            continue
        s0, e0, rep = p["start"], p["end"], p["replacement"]
        m = _ART.search(text[:s0])
        if m and rep[:2].lower() in ("a ", "an"):
            if m.group(1).lower() == "the":
                rep = rep.split(" ", 1)[1]
            else:
                s0 = m.start(1)
                if not text[s0:s0 + 1].isupper():
                    rep = rep[:1].lower() + rep[1:]
        if rep == "" and e0 < len(text) and text[e0:e0 + 1] == " ":
            e0 += 1
        p["applied"] = [s0, e0]
        ed.append((s0, e0, rep, 0))
        for cs, ce, crep, _r in p["companion"]:
            ed.append((cs, ce, crep, 1))
    acc = []
    for s0, e0, rep, pri in sorted(ed, key=lambda x: (x[3], -(x[1] - x[0]))):
        if all(e0 <= a or s0 >= b for a, b, _ in acc):
            acc.append((s0, e0, rep))
    out = text
    for s0, e0, rep in sorted(acc, key=lambda x: -x[0]):
        out = out[:s0] + rep + out[e0:]
    return out


def repair(text, cues_by_attr):
    """cues_by_attr: {attribute: [cue,...]} with base-text offsets. -> (new_text, trace)."""
    doc = nlp()(text)
    planned, seen = [], set()
    for attr in sorted(cues_by_attr, key=lambda a: V1._ATTR_PRIORITY.get(a, 9)):
        for cue in cues_by_attr[attr]:
            k = (cue["start"], cue["end"])
            if k in seen:
                continue
            seen.add(k)
            planned.append(plan_cue(text, doc, attr, cue))
    return apply_edits(text, planned), planned


def selftest():
    tests = [json.loads(l) for l in open(os.path.join(
        REPO, "runs/dev/sacc_improvement/unit/repair_unit_tests.jsonl"))]
    bad = 0
    for t in tests:
        s = t["sentence"]
        i = s.lower().find(t["cue"].lower())          # the test names the cue, the text owns its case
        assert i >= 0, f"{t['test_id']}: cue not in sentence"
        cue = {"surface": s[i:i + len(t["cue"])], "start": i, "end": i + len(t["cue"])}
        got, _ = repair(s, {t["attribute"]: [cue]})
        ok = got == t["expect_output"]
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {t['test_id']} {t['rule']}")
        if not ok:
            print(f"        in:  {s!r}\n        got: {got!r}\n        exp: {t['expect_output']!r}")
    print(f"{len(tests) - bad}/{len(tests)} repair unit tests pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
