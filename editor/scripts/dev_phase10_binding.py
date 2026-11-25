#!/usr/bin/env python
"""Phase 10 P2 -- subject-attribute binding (plan §5).

Rules: runs/dev/sacc_improvement/plan/binding_rules.yaml, frozen before this file produced output.
Tests: runs/dev/sacc_improvement/unit/binding_unit_tests.jsonl -- `--selftest` runs them.

The lexicon, the detector and the M1 replacer are unchanged. This module decides, for one cue and
one protected subject, WHO the cue is about, and returns one of the five statuses of §5.2 with the
rule that decided it. No trained coreference, no LLM.
"""
import json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.basename(REPO) != "spia":
    REPO = os.path.join(REPO, "spia")
sys.path.insert(0, REPO)

NONPERSON_ENT = {"ORG", "GPE", "FAC", "LOC", "NORP", "EVENT", "LAW"}
INSTITUTION = {"court", "tribunal", "hospital", "clinic", "ward", "school", "university",
               "college", "company", "firm", "chambers", "practice", "government", "ministry",
               "department", "office", "police", "army", "military", "bank", "agency",
               "council", "prison", "embassy", "institution", "workplace"}
RELATIONAL = {"wife", "husband", "spouse", "partner", "fiance", "fiancee", "widow", "widower",
              "mother", "father", "parent", "son", "daughter", "child", "brother", "sister",
              "sibling", "couple", "marriage", "wedding", "divorce", "engagement"}
FIRST_PERSON = {"i", "me", "my", "mine", "myself", "we", "our", "us"}
PRONOUNS = {"he", "she", "him", "her", "his", "hers", "himself", "herself", "they", "them", "their"}


def _head(doc, s0, e0):
    toks = [t for t in doc if t.idx < e0 and t.idx + len(t.text) > s0]
    if not toks:
        return None, []
    inside = {t.i for t in toks}
    for t in toks:                       # the token whose head lies outside the span
        if t.head.i not in inside or t.head.i == t.i:
            return t, toks
    return toks[-1], toks


def _ent_type(tok):
    return tok.ent_type_ if tok is not None and tok.ent_type_ else None


def _mentions_subject(text, mentions, corpus):
    low = (text or "").strip().lower().strip('"')
    if not low:
        return False
    if corpus == "synthpai" and low in FIRST_PERSON:
        return True
    for m in mentions:
        ml = m.lower()
        if low == ml or low in ml or ml in low:
            return True
        parts = [p for p in ml.replace(",", " ").split() if len(p) > 3]
        if parts and low in parts:
            return True
    return False


def _span_text(tok):
    """the noun phrase a referent token heads, as written"""
    if tok is None:
        return ""
    return tok.doc[tok.left_edge.i:tok.right_edge.i + 1].text


def bind(doc, s0, e0, attribute, mentions, corpus, n_subjects=1, own_turns=(), author=False):
    """-> dict(status, rule_id, syntax_role, direct_referent, supported_pairs)

    own_turns: spans of the protected subject's own speaker turns, from the frozen
    cues._own_turns, used when the corpus profile has turn_scope.
    author: the corpus profile's subject_is_author flag."""
    head, toks = _head(doc, s0, e0)
    out = {"status": "AMBIGUOUS", "rule_id": None, "syntax_role": head.dep_ if head else None,
           "direct_referent": None, "supported_pairs": []}
    if head is None:
        return dict(out, rule_id="no_token")
    surf = doc.text[s0:e0].strip().lower()

    # ---- 1. institution / place / body, checked first
    bare = surf.replace("the ", "").replace("a ", "").strip()
    if surf not in RELATIONAL and (_ent_type(head) in NONPERSON_ENT
                                   or bare in INSTITUTION or head.lemma_.lower() in INSTITUTION):
        return dict(out, status="NONPERSON_USE",
                    rule_id="nonperson.named_entity" if _ent_type(head) in NONPERSON_ENT
                    else "nonperson.institution_noun", direct_referent=_span_text(head))

    # ---- 2. referent rules, first match wins
    poss = next((c for c in head.children if c.dep_ == "poss"), None)
    if poss is not None:
        ref_tok, rule = poss, "R3_possessive"
        if poss.lower_ in FIRST_PERSON:
            rule = "R4_first_person"
    elif head.dep_ in ("attr", "acomp") and head.head is not None:
        ref_tok = next((c for c in head.head.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        rule = "R1_subject_complement"
    elif head.dep_ == "appos":
        ref_tok, rule = head.head, "R2_apposition"
    elif head.dep_ in ("dobj", "pobj") and surf not in PRONOUNS:
        ref_tok, rule = head, "R5_object_of_verb"
    elif head.dep_ in ("nsubj", "nsubjpass") and surf not in PRONOUNS:
        ref_tok, rule = head, "R6_bare_subject"
    elif surf in PRONOUNS:
        if n_subjects <= 1:
            return dict(out, status="DIRECT_TARGET", rule_id="R7_pronoun_single_subject",
                        direct_referent=mentions[0] if mentions else None,
                        supported_pairs=[["SUBJECT", attribute]])
        # R8: a protected subject named in this or the previous sentence, nothing else between
        sent = head.sent
        before = [x for x in doc[:head.i] if x.idx >= max(0, sent.start_char - 400)]
        people = [x for x in before if x.ent_type_ == "PERSON" or x.lower_ in FIRST_PERSON]
        # recency alone must not decide (plan §5.3): if more than one distinct person is in the
        # window, the pronoun is undecided, whether or not the nearest one is a protected subject
        distinct = {_span_text(x).strip().lower() for x in people}
        if len(distinct) == 1 and _mentions_subject(_span_text(people[-1]), mentions, corpus):
            return dict(out, status="DIRECT_TARGET", rule_id="R8_pronoun_nearby",
                        direct_referent=_span_text(people[-1]),
                        supported_pairs=[["SUBJECT", attribute]])
        return dict(out, status="AMBIGUOUS",
                    rule_id="R8_pronoun_nearby.blocked_multiple" if len(distinct) > 1
                    else "R8_pronoun_nearby.blocked", direct_referent=None)
    elif any(x <= s0 and e0 <= y for x, y in own_turns):
        st = "RELATIONAL_SHARED" if surf in RELATIONAL else "DIRECT_TARGET"
        return dict(out, status=st, rule_id="R4b_own_turn",
                    direct_referent=mentions[0] if mentions else None,
                    supported_pairs=[["SUBJECT", "RELATIONSHIP"], ["RELATIVE", "SEX"]]
                    if st == "RELATIONAL_SHARED" else [["SUBJECT", attribute]])
    elif author:
        st = "RELATIONAL_SHARED" if surf in RELATIONAL else "DIRECT_TARGET"
        return dict(out, status=st, rule_id="R4c_subject_is_author", direct_referent="the author",
                    supported_pairs=[["SUBJECT", "RELATIONSHIP"], ["RELATIVE", "SEX"]]
                    if st == "RELATIONAL_SHARED" else [["SUBJECT", attribute]])
    else:
        return dict(out, rule_id="no_rule_fired")
    if ref_tok is None:
        return dict(out, rule_id=rule + ".no_referent")
    ref = _span_text(ref_tok)
    out.update(rule_id=rule, direct_referent=ref)

    # ---- 3. classification
    is_subject = _mentions_subject(ref, mentions, corpus) or \
        (rule == "R4_first_person" and corpus == "synthpai")
    if surf in RELATIONAL and is_subject:
        return dict(out, status="RELATIONAL_SHARED",
                    supported_pairs=[["SUBJECT", "RELATIONSHIP"], ["RELATIVE", "SEX"]])
    if rule in ("R5_object_of_verb", "R6_bare_subject"):
        # the cue describes its own person; that person is the subject only if the cue IS a mention
        return dict(out, status="DIRECT_TARGET" if _mentions_subject(doc.text[s0:e0], mentions, corpus)
                    else "OTHER_ENTITY", supported_pairs=[["SUBJECT", attribute]]
                    if _mentions_subject(doc.text[s0:e0], mentions, corpus) else [])
    if is_subject:
        return dict(out, status="DIRECT_TARGET", supported_pairs=[["SUBJECT", attribute]])
    return dict(out, status="OTHER_ENTITY")


def selftest():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    tests = [json.loads(l) for l in open(os.path.join(
        REPO, "runs/dev/sacc_improvement/unit/binding_unit_tests.jsonl"))]
    bad = 0
    for t in tests:
        s = t["sentence"]; i = s.lower().find(t["cue"].lower())
        assert i >= 0, t["test_id"]
        doc = nlp(s)
        r = bind(doc, i, i + len(t["cue"]), t["attribute"], t["subject_mentions"],
                 t["corpus"], n_subjects=len(t["subject_mentions"]))
        ok = r["status"] == t["expect_status"]
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {t['test_id']} {t['expect_status']:18s} {t['why']}")
        if not ok:
            print(f"        got {r['status']} via {r['rule_id']} referent={r['direct_referent']!r}")
    print(f"{len(tests) - bad}/{len(tests)} binding unit tests pass")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(selftest() if "--selftest" in sys.argv else 0)
