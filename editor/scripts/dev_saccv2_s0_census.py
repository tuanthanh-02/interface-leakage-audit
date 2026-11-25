#!/usr/bin/env python3
"""SACC-v2 stack, stage S0 — coverage census (CPU, stored artifacts only).

Prereg: runs/dev/preregs/saccv2_stack_prereg.md §2 and Addendum 1.
Reads the Phase 16 P3 entry scores, candidate traces and arm outputs; rebuilds the detector base text; measures, for each
candidate component, the share of SACC-M residual d mass (raw sum per family) in the entries it touches.

    dev_saccv2_s0_census.py            full census -> runs/dev/saccv2_stack/s0/
    dev_saccv2_s0_census.py --selftest binding regression only (bind_alias with no spans == frozen bind)
"""
import argparse
import collections
import hashlib
import importlib.util as il
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

SP = __import__("os").environ["SACC_ROOT"]
os.chdir(SP)
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(SP, "scripts"))
from src.relaware import registry, profiles                  # noqa: E402
from src.relaware.method import cues as CUES                 # noqa: E402
import dev_phase10_repair as R                               # noqa: E402
import dev_phase10_binding as B                              # noqa: E402
import dev_phase10_emit_m1 as M1E                            # noqa: E402

OUT = "runs/dev/saccv2_stack/s0"
assert OUT.startswith("runs/dev/"), "dev track only"
MU = "runs/dev/main_upgrade"
BLOCKS = {"MED-ND": "medqa", "TAB-ND": "tab", "SYN-ND": "synthpai"}
TAGS = {"medqa": ["SEX", "OCCUPATION"], "tab": ["SEX", "OCCUPATION"], "synthpai": ["SEX", "OCCUPATION", "RELATIONSHIP"]}
TAU = {"medqa": .8, "tab": .6, "synthpai": .5}
ARMS5 = ["base", "B2", "B3", "SACC", "B6"]
B4 = "P16_B4_SACC_P14"
EDITABLE = ("DIRECT_TARGET", "RELATIONAL_SHARED")
PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
GENDERED_KIN = {"mother", "father", "son", "daughter", "brother", "sister", "wife", "husband", "widow", "widower",
                "aunt", "uncle", "niece", "nephew", "grandmother", "grandfather", "granddaughter", "grandson",
                "girlfriend", "boyfriend", "fiance", "fiancee", "fiancé", "fiancée", "bride", "groom", "stepmother",
                "stepfather", "stepson", "stepdaughter", "mom", "dad", "mum", "mommy", "daddy", "stepsister",
                "stepbrother", "grandma", "grandpa"}
POSS_PRON = {"his", "her", "their", "its", "my", "our", "your"}
h16 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]


def _load(name, path):
    s = il.spec_from_file_location(name, path)
    m = il.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def overlaps(a, b, spans):
    return any(a < y and x < b for x, y in spans)


# ------------------------------------------------------------------ base text with placeholder positions
def make_base(ds, M1E_cfg):
    tree, corp, layer, _b, _s = M1E_cfg[ds]
    if layer == "lf":
        ai = _load("ai", "src/anonymize/anonymize_intact.py")
        spans_of = ai.load_spans(ds, "longformer", None)

        def base_spans(did, text):
            bs = []
            for e in spans_of.get(did) or []:
                s0, e0 = e["start_offset"], e["end_offset"]
                if 0 <= s0 < e0 <= len(text):
                    bs.append((s0, e0, f"[{e.get('entity_type', 'MISC')}]"))
            bs.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            kept, last = [], -1
            for x in bs:
                if x[0] >= last:
                    kept.append(x)
                    last = x[1]
            return kept
    else:
        s1 = _load("s1", "scripts/dev_simple_spans.py")
        import spacy
        nlp0 = spacy.load("en_core_web_sm", disable=["lemmatizer"])
        occ = s1.OccLex("tight")

        def base_spans(did, text):
            return [(x, y, f"[{t}]") for x, y, t in s1.s1_spans(text, nlp0, occ)]
    return corp, base_spans


def placeholder_map(text, spans):
    """replicate v1_sweep acceptance with no cue edits; return [(src_s, src_e, base_s, base_e)]"""
    acc = []
    for s, e, r in sorted(spans, key=lambda x: -(x[1] - x[0])):
        if all(e <= a or s >= b for a, b, _ in acc):
            acc.append((s, e, r))
    out, cum = [], 0
    for s, e, r in sorted(acc):
        out.append((s, e, s + cum, s + cum + len(r), r))
        cum += len(r) - (e - s)
    return out


def find_all_ci(text, needle):
    if not needle or len(needle) < 2:
        return []
    pat = re.escape(needle)
    if needle[0].isalnum():
        pat = r"(?<!\w)" + pat
    if needle[-1].isalnum():
        pat = pat + r"(?!\w)"
    return [(m.start(), m.end()) for m in re.finditer(pat, text, re.I)]


ALIAS_RE = re.compile(r"(\[[A-Z_]+\])\s*\(\s*[\"“”'‘’]*\s*((?:the\s+)?(?:(?:first|second|third|fourth|fifth|sixth|"
                      r"seventh|eighth|ninth|tenth)\s+)?applicants?)\s*[\"“”'‘’]*\s*\)", re.I)
HEREIN_RE = re.compile(r"(\[[A-Z_]+\])[^()]{0,40}\(\s*hereinafter\s*[\"“”'‘’]*([^\"“”'‘’)]{2,40})[\"“”'‘’]*\s*\)", re.I)


def subject_spans(src, base, pmap, m, S):
    """per subject: base spans that refer to it (mention occurrences mapped to base, or the placeholder over them),
    plus occurrences of harvested aliases owned by exactly one subject"""
    spans = {s["eval_subject_id"]: set() for s in S}
    for s in S:
        for form in s["gold_mention"]:
            for a, b in find_all_ci(src, form):
                if a in m and (b - 1) in m and m[b - 1] + 1 - m[a] == b - a:
                    spans[s["eval_subject_id"]].add((m[a], m[b - 1] + 1))
                for ps, pe, bs, be, _r in pmap:
                    if a < pe and ps < b:
                        spans[s["eval_subject_id"]].add((bs, be))
    owner_ph = {}
    for sid, ss in spans.items():
        for x in ss:
            owner_ph.setdefault(x, set()).add(sid)
    aliases = collections.defaultdict(set)
    for rx in (ALIAS_RE, HEREIN_RE):
        for mm in rx.finditer(base):
            ph = (mm.start(1), mm.end(1))
            own = owner_ph.get(ph, set())
            if len(own) == 1:
                aliases[re.sub(r"\s+", " ", mm.group(2).strip().lower())].add(next(iter(own)))
    alias_hits = {}
    for al, owners in aliases.items():
        if len(owners) != 1:
            continue
        sid = next(iter(owners))
        for a, b in find_all_ci(base, al):
            spans[sid].add((a, b))
        alias_hits[al] = sid
    return {k: sorted(v) for k, v in spans.items()}, alias_hits


# ------------------------------------------------------------------ binding with the offset-aware referent test
def bind_alias(doc, s0, e0, attribute, mentions, corpus, sspans, n_subjects=1, own_turns=(), author=False):
    """dev_phase10_binding.bind (frozen) with one change: a referent token (or, for R5/R6, an appos/compound child of
    the cue head) whose character span overlaps one of the subject's base spans counts as the subject.
    With sspans == [] this must equal B.bind exactly (selftest)."""
    head, toks = B._head(doc, s0, e0)
    out = {"status": "AMBIGUOUS", "rule_id": None, "syntax_role": head.dep_ if head else None,
           "direct_referent": None, "supported_pairs": []}
    if head is None:
        return dict(out, rule_id="no_token")
    surf = doc.text[s0:e0].strip().lower()

    def hit(tok):
        return tok is not None and overlaps(tok.idx, tok.idx + len(tok.text), sspans)

    bare = surf.replace("the ", "").replace("a ", "").strip()
    if surf not in B.RELATIONAL and (B._ent_type(head) in B.NONPERSON_ENT
                                     or bare in B.INSTITUTION or head.lemma_.lower() in B.INSTITUTION):
        return dict(out, status="NONPERSON_USE",
                    rule_id="nonperson.named_entity" if B._ent_type(head) in B.NONPERSON_ENT
                    else "nonperson.institution_noun", direct_referent=B._span_text(head))
    poss = next((c for c in head.children if c.dep_ == "poss"), None)
    if poss is not None:
        ref_tok, rule = poss, "R3_possessive"
        if poss.lower_ in B.FIRST_PERSON:
            rule = "R4_first_person"
    elif head.dep_ in ("attr", "acomp") and head.head is not None:
        ref_tok = next((c for c in head.head.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        rule = "R1_subject_complement"
    elif head.dep_ == "appos":
        ref_tok, rule = head.head, "R2_apposition"
    elif head.dep_ in ("dobj", "pobj") and surf not in B.PRONOUNS:
        ref_tok, rule = head, "R5_object_of_verb"
    elif head.dep_ in ("nsubj", "nsubjpass") and surf not in B.PRONOUNS:
        ref_tok, rule = head, "R6_bare_subject"
    elif surf in B.PRONOUNS:
        if n_subjects <= 1:
            return dict(out, status="DIRECT_TARGET", rule_id="R7_pronoun_single_subject",
                        direct_referent=mentions[0] if mentions else None,
                        supported_pairs=[["SUBJECT", attribute]])
        sent = head.sent
        before = [x for x in doc[:head.i] if x.idx >= max(0, sent.start_char - 400)]
        people = [x for x in before if x.ent_type_ == "PERSON" or x.lower_ in B.FIRST_PERSON]
        distinct = {B._span_text(x).strip().lower() for x in people}
        if len(distinct) == 1 and B._mentions_subject(B._span_text(people[-1]), mentions, corpus):
            return dict(out, status="DIRECT_TARGET", rule_id="R8_pronoun_nearby",
                        direct_referent=B._span_text(people[-1]), supported_pairs=[["SUBJECT", attribute]])
        return dict(out, status="AMBIGUOUS",
                    rule_id="R8_pronoun_nearby.blocked_multiple" if len(distinct) > 1
                    else "R8_pronoun_nearby.blocked", direct_referent=None)
    elif any(x <= s0 and e0 <= y for x, y in own_turns):
        st = "RELATIONAL_SHARED" if surf in B.RELATIONAL else "DIRECT_TARGET"
        return dict(out, status=st, rule_id="R4b_own_turn", direct_referent=mentions[0] if mentions else None)
    elif author:
        st = "RELATIONAL_SHARED" if surf in B.RELATIONAL else "DIRECT_TARGET"
        return dict(out, status=st, rule_id="R4c_subject_is_author", direct_referent="the author")
    else:
        return dict(out, rule_id="no_rule_fired")
    if ref_tok is None:
        return dict(out, rule_id=rule + ".no_referent")
    ref = B._span_text(ref_tok)
    out.update(rule_id=rule, direct_referent=ref)
    is_subject = B._mentions_subject(ref, mentions, corpus) or (rule == "R4_first_person" and corpus == "synthpai")
    if rule not in ("R5_object_of_verb", "R6_bare_subject"):
        is_subject = is_subject or hit(ref_tok)
    if surf in B.RELATIONAL and is_subject:
        return dict(out, status="RELATIONAL_SHARED")
    if rule in ("R5_object_of_verb", "R6_bare_subject"):
        own = B._mentions_subject(doc.text[s0:e0], mentions, corpus) or \
            any(hit(c) for c in head.children if c.dep_ in ("appos", "compound", "flat", "nmod", "npadvmod")) or \
            (head.dep_ == "appos" and hit(head.head))
        return dict(out, status="DIRECT_TARGET" if own else "OTHER_ENTITY")
    if is_subject:
        return dict(out, status="DIRECT_TARGET")
    return dict(out, status="OTHER_ENTITY")


# ------------------------------------------------------------------ lexica
def occ_wordnet(v1_occ_patterns, stop):
    from nltk.corpus import wordnet as wn
    roots = ["worker.n.01", "professional.n.01", "official.n.01", "leader.n.01"]
    lem = set()
    for r in roots:
        for syn in wn.synset(r).closure(lambda s: s.hyponyms()):
            for ln in syn.lemma_names():
                w = ln.replace("_", " ").lower()
                if len(w) >= 4 and w not in stop and re.fullmatch(r"[a-z][a-z .'-]*", w):
                    lem.add(w)
    keep = sorted(w for w in lem if not any(re.search(p, w, re.I) for p in v1_occ_patterns))
    return keep


def compile_alternation(words):
    words = sorted(set(words), key=len, reverse=True)
    parts = [re.escape(w) + r"(?:e?s)?" for w in words]
    return re.compile(r"(?<![\w-])(?:" + "|".join(parts) + r")(?![\w-])", re.I)


EMOJI_RE = re.compile("[\U0001F466\U0001F467\U0001F468\U0001F469\U0001F474\U0001F475\U0001F478\U0001F934"
                      "\U0001F470\U0001F930\U0001F931\U0001F9D4\U0001F46B\U0001F46C\U0001F46D\U0001F483\U0001F57A"
                      "♀♂]")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    lex_s = json.load(open(f"{OUT}/lexicons/sexclin_v0.json"))
    lex_o = json.load(open(f"{OUT}/lexicons/occrec_v0.json"))
    sex_re = {k: re.compile(r"\b(?:" + "|".join(lex_s[k]) + r")\b", re.I)
              for k in ("STRONG_FEMALE", "STRONG_MALE", "WEAK")}
    frame_re = re.compile(r"\b(?:" + "|".join(lex_o["frames"]) + r")\b", re.I)
    title_re = re.compile(r"\b(?:" + "|".join(lex_o["titles"]) + r")\b")

    df = pd.read_parquet(f"{MU}/evaluation/07_attack_entry_scores.parquet")
    df = df[df.block.isin(BLOCKS) & (df.scored == 1) & (df.excluded_duplicate_key == 0) & df.arm_alias.isin(ARMS5)]
    wide = df.pivot_table(index=["block", "family", "doc_id", "eval_subject_id", "attribute", "gold_value", "stratum",
                                 "prior"], columns="arm_alias", values="d").reset_index().dropna(subset=ARMS5)
    wide["corpus"] = wide.block.map(BLOCKS)
    touch = collections.defaultdict(set)      # component -> set of (doc_id, subject_id or None, attribute)
    load = collections.defaultdict(lambda: collections.Counter())
    regress = collections.Counter()
    notes = {}

    for blk, ds in BLOCKS.items():
        docs = set(wide[wide.block == blk].doc_id)
        tree = M1E.CFG[ds][0]
        reg = registry.load(f"{tree}/registry.json")
        prof = profiles.for_registry(reg)
        subs = collections.defaultdict(list)
        for s in reg["subjects"]:
            subs[s["doc_id"]].append(s)
        corp, base_spans = make_base(ds, M1E.CFG)
        traces = collections.defaultdict(list)
        base_hash = {}
        for l in open(f"{MU}/traces/{ds}_candidate_traces.jsonl"):
            r = json.loads(l)
            if r["document_id"] in docs:
                traces[r["document_id"]].append(r)
                base_hash[r["document_id"]] = r["base_hash"]
        occ_words = occ_wordnet([prof.lex["occ_role"]], set(lex_o["stoplist"]))
        occ_re = compile_alternation(occ_words)
        notes[f"{ds}_wordnet_terms"] = len(occ_words)
        nlp = R.nlp()
        n_alias_docs = 0
        for line in open(corp):
            rec = json.loads(line)
            did, src = rec["metadata"]["data_id"], rec["text"]
            if did not in docs:
                continue
            bsp = base_spans(did, src)
            base = M1E.v1_sweep(src, bsp, [])
            if did in base_hash:
                assert h16(base) == base_hash[did], f"base text mismatch {did}"
                regress["base_hash_ok"] += 1
            pmap = placeholder_map(src, bsp)
            assert all(base[bs:be] == r for _s, _e, bs, be, r in pmap), f"placeholder map mismatch {did}"
            m = M1E.offset_map(src, base)
            S = list(subs.get(did, []))
            cands = traces.get(did, [])
            ph_spans = [(x.start(), x.end()) for x in PLACEHOLDER.finditer(base)]
            union_spans = [(c["start"], c["end_exclusive"]) for c in cands]
            sacc_applied = [tuple(c["arms"][B4]["applied_span"]) for c in cands if c["arms"][B4]["applied_span"]]
            doc = nlp(base)

            # ---- discovery: clinical sex terms (MedQA channel, computed everywhere)
            for k, rx in sex_re.items():
                hits = [(x.start(), x.end()) for x in rx.finditer(base)
                        if not overlaps(x.start(), x.end(), ph_spans + sacc_applied)]
                if hits:
                    touch[f"C-SEXCLIN_{k}"].add((did, None, "SEX"))
                    load[f"C-SEXCLIN_{k}"][ds + "_spans"] += len(hits)
                    load[f"C-SEXCLIN_{k}"][ds + "_chars"] += sum(y - x for x, y in hits)

            # ---- discovery: occupation recall
            sspans, alias_hits = subject_spans(src, base, pmap, m, S)
            n_alias_docs += bool(alias_hits)
            occ_hits = []
            for rx, kind in ((occ_re, "wordnet"), (frame_re, "frame"), (title_re, "title")):
                for x in rx.finditer(base):
                    if not overlaps(x.start(), x.end(), ph_spans + union_spans):
                        occ_hits.append((x.start(), x.end(), kind))
            if occ_hits:
                touch["C-OCCREC_doc"].add((did, None, "OCCUPATION"))
                load["C-OCCREC_doc"][ds + "_spans"] += len(occ_hits)
                load["C-OCCREC_doc"][ds + "_chars"] += sum(y - x for x, y, _ in occ_hits)
                for kind in ("wordnet", "frame", "title"):
                    load["C-OCCREC_doc"][f"{ds}_{kind}"] += sum(1 for h in occ_hits if h[2] == kind)
                sents = [(snt.start_char, snt.end_char) for snt in doc.sents]
                for s in S:
                    ss = sspans.get(s["eval_subject_id"], [])
                    for x, y, _k in occ_hits:
                        sent = next(((p, q) for p, q in sents if p <= x < q), None)
                        if sent and overlaps(sent[0], sent[1], ss):
                            touch["C-OCCREC_sent"].add((did, s["eval_subject_id"], "OCCUPATION"))
                            break

            # ---- discovery: gendered emoji
            if EMOJI_RE.search(base):
                touch["C-EMOJI"].add((did, None, "SEX"))
                load["C-EMOJI"][ds + "_spans"] += len(EMOJI_RE.findall(base))

            # ---- binding components
            kept_keys = set()
            key = lambda c: (c["attribute"], re.sub(r"^(the|a|an)\s+", "", c["surface"].strip().lower()).strip(" .,;:"))
            for c in cands:
                if c["merged_status"] in EDITABLE:
                    kept_keys.add(key(c))
            for c in cands:
                if c["merged_status"] not in ("OTHER_ENTITY", "NONPERSON_USE"):
                    continue
                s0, e0, att = c["start"], c["end_exclusive"], c["attribute"]
                first_reason = c["pairs"][0]["binding_reason"] if c["pairs"] else None
                # C-PROP
                k = key(c)
                if k[1] not in ("patient", "the patient") and k in kept_keys and (
                        c["merged_status"] == "OTHER_ENTITY" or first_reason == "nonperson.institution_noun"):
                    touch["C-PROP"].add((did, None, att))
                    load["C-PROP"][ds + "_spans"] += 1
                    load["C-PROP"][ds + "_chars"] += e0 - s0
                if c["merged_status"] != "OTHER_ENTITY":
                    continue
                # C-REL2
                lemma = re.sub(r"^(his|her|their|its|my|our|your|the|a|an)\s+", "", c["surface"].strip().lower())
                lemma = re.sub(r"'s$", "", lemma)
                if lemma in B.RELATIONAL or lemma in GENDERED_KIN or lemma.rstrip("s") in GENDERED_KIN:
                    ok = first_reason in ("R5_object_of_verb", "R6_bare_subject")
                    if first_reason == "R3_possessive" and S:
                        s = S[0]
                        tt = tuple(CUES._own_turns(base, s["gold_mention"])) if getattr(prof, "turn_scope", False) else ()
                        fb = B.bind(doc, s0, e0, att, s["gold_mention"], ds, n_subjects=len(S) or 1, own_turns=tt,
                                    author=bool(prof.subject_is_author))
                        ok = (fb["direct_referent"] or "").strip().lower() in POSS_PRON
                    if ok:
                        touch["C-REL2"].add((did, None, att))
                        load["C-REL2"][ds + "_spans"] += 1
                        load["C-REL2"][ds + "_chars"] += e0 - s0
                # C-ALIAS
                for s in S:
                    sid = s["eval_subject_id"]
                    tt = tuple(CUES._own_turns(base, s["gold_mention"])) if getattr(prof, "turn_scope", False) else ()
                    kw = dict(n_subjects=len(S) or 1, own_turns=tt, author=bool(prof.subject_is_author))
                    if regress[f"{ds}_selftest"] < 400:
                        ref = B.bind(doc, s0, e0, att, s["gold_mention"], ds, **kw)
                        new0 = bind_alias(doc, s0, e0, att, s["gold_mention"], ds, [], **kw)
                        regress[f"{ds}_selftest"] += 1
                        regress[f"{ds}_selftest_mismatch"] += ref["status"] != new0["status"]
                    nb = bind_alias(doc, s0, e0, att, s["gold_mention"], ds, sspans.get(sid, []), **kw)
                    if nb["status"] in EDITABLE:
                        touch["C-ALIAS"].add((did, sid, att))
                        load["C-ALIAS"][ds + "_spans"] += 1
                        load["C-ALIAS"][ds + "_chars"] += e0 - s0
            load["SACC-M"][ds + "_spans"] += len(sacc_applied)
            load["SACC-M"][ds + "_chars"] += sum(y - x for x, y in sacc_applied)
            if a.selftest and regress[f"{ds}_selftest"] >= 400:
                break
        notes[f"{ds}_docs_with_alias"] = n_alias_docs
        print(f"[{ds}] done {time.time() - t0:.0f}s  selftest {regress[ds + '_selftest']} "
              f"mismatch {regress[ds + '_selftest_mismatch']}", flush=True)

    if a.selftest:
        print(dict(regress))
        return 1 if any(v for k, v in regress.items() if k.endswith("mismatch")) else 0

    # ---- touch shares
    def touched(comp, row):
        T = touch[comp]
        return ((row.doc_id, row.eval_subject_id, row.attribute) in T) or ((row.doc_id, None, row.attribute) in T)

    channels = {
        "C-SEXCLIN_STRONG": [("MED-ND", ["SEX"])],
        "C-SEXCLIN_WEAK": [("MED-ND", ["SEX"])],
        "C-OCCREC_doc": [("MED-ND", ["OCCUPATION"]), ("TAB-ND", ["OCCUPATION"]), ("SYN-ND", ["OCCUPATION"])],
        "C-OCCREC_sent": [("MED-ND", ["OCCUPATION"]), ("TAB-ND", ["OCCUPATION"]), ("SYN-ND", ["OCCUPATION"])],
        "C-EMOJI": [("SYN-ND", ["SEX"])],
        "C-ALIAS": [("TAB-ND", TAGS["tab"])],
        "C-REL2": [("TAB-ND", ["SEX", "RELATIONSHIP"]), ("MED-ND", ["SEX", "RELATIONSHIP"])],
        "C-PROP": [("TAB-ND", TAGS["tab"]), ("MED-ND", TAGS["medqa"]), ("SYN-ND", TAGS["synthpai"])],
        "BIND_UNION": [("TAB-ND", TAGS["tab"])],
    }
    for k in ("STRONG_FEMALE", "STRONG_MALE"):
        touch["C-SEXCLIN_STRONG"] |= touch[f"C-SEXCLIN_{k}"]
    rows = []
    for comp, chans in channels.items():
        for blk, atts in chans:
            for fam in ("llama", "qwen"):
                g = wide[(wide.block == blk) & (wide.family == fam) & wide.attribute.isin(atts)]
                if not len(g):
                    continue
                if comp == "BIND_UNION":
                    hit = [touched("C-ALIAS", r) or touched("C-REL2", r) or touched("C-PROP", r) for r in g.itertuples()]
                else:
                    hit = [touched(comp, r) for r in g.itertuples()]
                hit = np.array(hit, dtype=bool)
                mass = g.SACC.sum()
                rows.append({"component": comp, "block": blk, "family": fam, "attributes": "+".join(atts),
                             "entries": len(g), "entries_touched": int(hit.sum()),
                             "sacc_mass": round(float(mass), 3), "touched_mass": round(float(g.SACC[hit].sum()), 3),
                             "touch_share": round(float(g.SACC[hit].sum() / mass), 4) if mass else None,
                             "touched_mean_d_sacc": round(float(g.SACC[hit].mean()), 4) if hit.any() else None,
                             "untouched_mean_d_sacc": round(float(g.SACC[~hit].mean()), 4) if (~hit).any() else None,
                             "touched_mean_d_B2": round(float(g.B2[hit].mean()), 4) if hit.any() else None})
    shares = pd.DataFrame(rows)

    # ---- C-OPT (T17 coarse): select on DEV leakage, simulate on P3
    arm_name = {"SACC": "P16_B4_SACC_P14", "B6": "P16_B6_GLOBAL_LADDER", "B3": "P16_B3_GLOBAL_NP_MASK",
                "B2": "P16_B2_GLOBAL_MASK"}
    opt_rows, choice = [], {}
    for ds in BLOCKS.values():
        per_fam = {}
        for fam in ("llama", "qwen"):
            fn = [f for f in os.listdir(f"{MU}/e2root_{fam}/evaluation") if f.startswith(f"e2_{ds}_base-")
                  and f.endswith(".json") and not f.endswith("_run.json")][0]
            res = pd.DataFrame(json.load(open(f"{MU}/e2root_{fam}/evaluation/{fn}"))["results"])
            res = res[res.prior < TAU[ds]].copy()
            res["d"] = ((res.p_source_aware - res.prior) / (1 - res.prior)).clip(0, 1)
            w = res.pivot_table(index=["eval_subject_id", "attribute", "stratum"], columns="method",
                                values="d").reset_index().dropna(subset=list(arm_name.values()))
            per_fam[fam] = {att: {k: float(w[w.attribute == att].groupby("stratum")[v].mean().mean())
                                  for k, v in arm_name.items()} for att in TAGS[ds] if (w.attribute == att).any()}
        for att in TAGS[ds]:
            if att not in per_fam["llama"] or att not in per_fam["qwen"]:
                choice[(ds, att)] = "SACC"
                continue
            bl = min(per_fam["llama"][att], key=per_fam["llama"][att].get)
            bq = min(per_fam["qwen"][att], key=per_fam["qwen"][att].get)
            choice[(ds, att)] = bl if bl == bq else "SACC"
            opt_rows.append({"corpus": ds, "attribute": att, "dev_best_llama": bl, "dev_best_qwen": bq,
                             "chosen": choice[(ds, att)],
                             **{f"dev_{fam}_{k}": round(v, 4) for fam in ("llama", "qwen")
                                for k, v in per_fam[fam][att].items()}})
    sim = []
    for blk, ds in BLOCKS.items():
        for fam in ("llama", "qwen"):
            g = wide[(wide.block == blk) & (wide.family == fam)].copy()
            g["composite"] = [row[choice[(ds, row.attribute)]] if (ds, row.attribute) in choice else row.SACC
                              for _, row in g.iterrows()]

            def endpoint(col):
                v = [g[g.attribute == t].groupby("stratum")[col].mean().mean() for t in TAGS[ds] if (g.attribute == t).any()]
                return float(np.mean(v))
            sim.append({"block": blk, "family": fam, "L_sacc": round(endpoint("SACC"), 4),
                        "L_composite": round(endpoint("composite"), 4),
                        "improvement": round(endpoint("SACC") - endpoint("composite"), 4)})

    # ---- write
    shares.to_csv(f"{OUT}/touch_shares.csv.tmp", index=False)
    os.replace(f"{OUT}/touch_shares.csv.tmp", f"{OUT}/touch_shares.csv")
    rep = {"prereg": "runs/dev/preregs/saccv2_stack_prereg.md (§2, Addendum 1)",
           "script_sha16": hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16],
           "lexicon_sha16": {f: hashlib.sha256(open(f"{OUT}/lexicons/{f}", "rb").read()).hexdigest()[:16]
                             for f in ("sexclin_v0.json", "occrec_v0.json")},
           "regression": dict(regress), "notes": notes,
           "edit_load": {k: dict(v) for k, v in load.items()},
           "c_opt_choice": opt_rows, "c_opt_simulation": sim,
           "seconds": round(time.time() - t0, 1)}
    with open(f"{OUT}/census.json.tmp", "w") as fh:
        json.dump(rep, fh, indent=1)
    os.replace(f"{OUT}/census.json.tmp", f"{OUT}/census.json")
    print(shares.to_string(index=False))
    print(json.dumps({"edit_load": rep["edit_load"], "c_opt_simulation": sim, "notes": notes,
                      "regression": dict(regress)}, indent=1))
    print(pd.DataFrame(opt_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
