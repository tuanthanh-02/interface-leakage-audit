#!/usr/bin/env python3
"""SACC-v2 stack, stage S1 — emit the unified method V2-U, its value-invariant realization V2-U-VA and R0–R4 (CPU).

Prereg: runs/dev/preregs/saccv2_stack_prereg.md, Addendum 4 (supersedes Addendum 2's corpus-specific design), 2a, 2b.
No rule below reads the corpus identifier; `ds` only selects input files and the detector layer.

Pools (local ∪ global), the per-subject frozen binding and the frozen merge are the Phase 16 emitter's
(`dev_phase16_emit_arms.py`, itself `dev_phase10_emit_m2.py@f8ce2529`), copied line for line. On top of them:
an attribute policy (SACC / B2 / B6 per corpus and attribute), C-SEXCLIN, C-REL2, C-EMOJI, the post hoc C-ALIAS-S,
a value-invariant realization (va_v0.json) and count-matched random controls.

    dev_saccv2_s1_emit.py --ds medqa            emit every document; gates K-S1 (a), (b), (d) on every document
    dev_saccv2_s1_emit.py --unit                gate K-S1 (c): value-invariance unit tests of the V2-VA realization
    dev_saccv2_s1_emit.py --fresh medqa|tab     Addendum 6: the same code with the fresh cohort's registry, D1 detections,
                                                gold and span layer; writes s3/arms and emits the detector arm as well

Writes runs/dev/saccv2_stack/s1/arms/{ds}_{ARM}/anonymized.jsonl, s1/traces/{ds}_emit_run.json, s1/traces/{ds}_manifest.csv
"""
import argparse
import collections
import csv
import hashlib
import importlib.util as il
import json
import os
import random
import re
import sys
import time

SP = __import__("os").environ["SACC_ROOT"]
os.chdir(SP)
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(SP, "scripts"))
from src.relaware import registry, profiles                  # noqa: E402
from src.relaware.method import cues as CUES                 # noqa: E402
from src.relaware.method import edits as V1                  # noqa: E402
import dev_phase10_repair as R                               # noqa: E402
import dev_phase10_binding as B                              # noqa: E402
import dev_phase10_emit_m1 as M1E                            # noqa: E402
import dev_saccv2_s0_census as C0                            # noqa: E402

OUT = "runs/dev/saccv2_stack/s1"
assert OUT.startswith("runs/dev/"), "dev track only"
MU = "runs/dev/main_upgrade"
TAGS = ("SEX", "OCCUPATION", "RELATIONSHIP")
EDITABLE = ("DIRECT_TARGET", "RELATIONAL_SHARED")
EXCLUDED = {"OTHER_ENTITY", "NONPERSON_USE"}
ARMS = ("V2U", "V2U_VA", "V2U_R0", "V2U_R1", "V2U_R2", "V2U_R3", "V2U_R4")
FRESH = {"medqa": ("runs/dev/saccv2_stack/fresh/medqa", "runs/dev/saccv2_stack/fresh/medqa/spia_medqa_fresh.jsonl", "s1", None),
         "tab": ("runs/dev/saccv2_stack/fresh/tab", "runs/dev/saccv2_stack/fresh/tab/spia_tab_fresh.jsonl", "lf",
                 "runs/dev/saccv2_stack/fresh/tab/longformer_spans.jsonl")}
FRESH_OUT = "runs/dev/saccv2_stack/s3"
STORED = {"B2": "P16_B2_GLOBAL_MASK", "B4": "P16_B4_SACC_P14", "B6": "P16_B6_GLOBAL_LADDER"}
VA = json.load(open(f"{OUT}/lexicons/va_v0.json"))
SEXCLIN = json.load(open("runs/dev/saccv2_stack/s0/lexicons/sexclin_v0.json"))
SEXCLIN_RE = re.compile(r"\b(?:" + "|".join(SEXCLIN["STRONG_FEMALE"] + SEXCLIN["STRONG_MALE"]) + r")\b", re.I)
EMOJI_MAP = VA["EMOJI"]["map"]
EMOJI_STRIP = re.compile("‍[♀♂]️?")
EMOJI_RE = re.compile("|".join(re.escape(k) for k in sorted(EMOJI_MAP, key=len, reverse=True)))
PLACEHOLDER = re.compile(r"\[[A-Z_]+\]")
h16 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]


def sha16_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


# ---------------------------------------------------------------- parse cache (R.repair re-parses the same base text)
class _CachedNLP:
    def __init__(self, nlp):
        self.nlp, self.key, self.doc = nlp, None, None

    def __call__(self, text):
        if text != self.key:
            self.key, self.doc = text, self.nlp(text)
        return self.doc


R._NLP = _CachedNLP(R.nlp())


# ---------------------------------------------------------------- realization
def plan_items(base_txt, by_attr):
    _t, plan = R.repair(base_txt, dict(by_attr))
    return plan


def mask_item(q):
    ap = q.get("applied") or [q["start"], q["end"]]
    return dict(start=ap[0], end=ap[1], surface=q["surface"], attribute=q["attribute"], companion=[],
                rule="C1_fixed_placeholder", action="REPLACE", replacement=R.MASK)


_WN_PERSON = {}


def is_person_noun(lemma):
    if lemma not in _WN_PERSON:
        from nltk.corpus import wordnet as wn
        person = wn.synset("person.n.01")
        ok = False
        for s in wn.synsets(lemma, pos="n"):
            if person in s.closure(lambda x: x.hypernyms()) or s == person:
                ok = True
                break
        _WN_PERSON[lemma] = ok
    return _WN_PERSON[lemma]


def va_item(text, doc, q):
    """value-invariant realization of one planned span (va_v0.json); None -> mask"""
    s0, e0, att = q["start"], q["end"], q["attribute"]
    surface = text[s0:e0]
    low = surface.strip().lower()
    toks = R._span_tokens(doc, s0, e0)
    head = toks[-1] if toks else None
    det = ""
    m = re.match(r"^(my|his|her|their|our|your)\s+(.*)$", low)
    if m and att in ("SEX", "RELATIONSHIP", "OCCUPATION"):
        det = {"his": "their", "her": "their"}.get(m.group(1), m.group(1)) + " "
        low = m.group(2)
    plural = head is not None and head.tag_ in ("NNS", "NNPS")
    fill = None
    if att == "SEX":
        if low in V1.PRONOUN_MAP or low == "hers":
            tok = toks[0] if toks else None
            rep = R._pronoun_rep(tok, surface)
            comp = R.agreement_edits(tok) if rep == "they" else []
            return dict(start=s0, end=e0, surface=surface, attribute=att, companion=comp, rule="va_pronoun",
                        action="REPLACE", replacement=R._cap_like(surface, rep))
        if low.rstrip(".") in V1.TITLES:
            return dict(start=s0, end=e0, surface=surface, attribute=att, companion=[], rule="va_title",
                        action="REPLACE", replacement="")
        pair = VA["SEX"]["nouns"].get(low) or VA["SEX"]["nouns"].get(low.rstrip("s"))
        if pair:
            fill = pair[1] if (plural or low in ("men", "women", "gentlemen", "ladies")) else pair[0]
    elif att == "RELATIONSHIP":
        r = VA["RELATIONSHIP"]
        base = low[:-1] if plural and low.endswith("s") else low
        if base in r["person_nouns"]:
            fill = r["person_filler"][1 if plural else 0]
        elif base in r["event_nouns"]:
            fill = r["event_filler"][1 if plural else 0]
        elif base in VA["SEX"]["nouns"] and VA["SEX"]["nouns"][base][0] in ("parent", "child", "sibling", "grandparent",
                                                                             "grandchild", "relative"):
            fill = VA["SEX"]["nouns"][base][1 if plural else 0]
    elif att == "OCCUPATION":
        o = VA["OCCUPATION"]
        if head is not None and len(toks) == 1 or (det and head is not None):
            lem = head.lemma_.lower() if head is not None else low
            if lem in o["workplace_nouns"]:
                fill = o["workplace_filler"][1 if plural else 0]
            elif head is not None and head.pos_ == "NOUN" and is_person_noun(lem):
                fill = o["role_filler"][1 if plural else 0]
    if fill is None:
        return None
    rep = det + fill
    if not det and not plural and R._ART.search(text[:s0]):
        rep = "a " + rep                       # the frozen apply_edits absorbs a/an or drops it after "the"
    return dict(start=s0, end=e0, surface=surface, attribute=att, companion=[], rule="va_fill", action="REPLACE",
                replacement=R._cap_like(surface, rep))


def realize(base_txt, doc, by_attr, realization, extras, va=False):
    """realization: attribute -> 'ladder' | 'mask'; extras: planned dicts (non-candidate edits) appended last"""
    plan = plan_items(base_txt, by_attr)
    planned = []
    for q in plan:
        if q["action"] == "KEEP":
            continue
        if va:
            it = va_item(base_txt, doc, q)
            planned.append(it if it is not None else mask_item(q))
        elif realization.get(q["attribute"]) == "ladder":
            planned.append({k: v for k, v in q.items() if k != "applied"})
        else:
            planned.append(mask_item(q))
    planned += [dict(x) for x in extras]
    txt = R.apply_edits(base_txt, planned)
    return txt, plan, planned


EMOJI_SEQ = re.compile("(" + "|".join(re.escape(k) for k in sorted(EMOJI_MAP, key=len, reverse=True)) + "|[^\\s])"
                       "(\u200d[\u2640\u2642]\ufe0f?)?")


def emoji_extras(base_txt, taken):
    """one edit per gendered emoji sequence: base codepoint mapped (va_v0 map) and a ZWJ gender sign deleted; the
    replacement is never empty, so the frozen apply_edits never absorbs a following space"""
    ex = []
    for mm in EMOJI_SEQ.finditer(base_txt):
        base_cp, sign = mm.group(1), mm.group(2)
        if base_cp not in EMOJI_MAP and not sign:
            continue
        rep = EMOJI_MAP.get(base_cp, base_cp)
        if not C0.overlaps(mm.start(), mm.end(), taken + [(x["start"], x["end"]) for x in ex]):
            ex.append(dict(start=mm.start(), end=mm.end(), surface=mm.group(0), attribute="SEX", companion=[],
                           rule="emoji", action="REPLACE", replacement=rep))
    return ex


def mask_extra(s0, e0, text, rule):
    return dict(start=s0, end=e0, surface=text[s0:e0], attribute="SEX", companion=[], rule=rule, action="REPLACE",
                replacement=R.MASK)


def plan_spans(plan):
    return [tuple(q.get("applied") or [q["start"], q["end"]]) for q in plan if q["action"] != "KEEP"]


# ---------------------------------------------------------------- unit tests, gate K-S1 (c)
UNIT = [
    ("She said her husband was a lawyer.", "He said his wife was a lawyer.",
     [("She", "SEX"), ("her", "SEX"), ("husband", "RELATIONSHIP"), ("lawyer", "OCCUPATION")],
     [("He", "SEX"), ("his", "SEX"), ("wife", "RELATIONSHIP"), ("lawyer", "OCCUPATION")]),
    ("A 45-year-old woman presents with pain; her mother had diabetes.",
     "A 45-year-old man presents with pain; his father had diabetes.",
     [("woman", "SEX"), ("her", "SEX"), ("mother", "SEX")], [("man", "SEX"), ("his", "SEX"), ("father", "SEX")]),
    ("The applicant and his brother met the nurse.", "The applicant and her sister met the doctor.",
     [("his", "SEX"), ("brother", "SEX"), ("nurse", "OCCUPATION")],
     [("her", "SEX"), ("sister", "SEX"), ("doctor", "OCCUPATION")]),
    ("After the divorce she moved to a hospital.", "After the wedding he moved to a clinic.",
     [("divorce", "RELATIONSHIP"), ("she", "SEX"), ("hospital", "OCCUPATION")],
     [("wedding", "RELATIONSHIP"), ("he", "SEX"), ("clinic", "OCCUPATION")]),
]


def unit():
    bad = 0
    for a_txt, b_txt, a_c, b_c in UNIT:
        outs = []
        for txt, cs in ((a_txt, a_c), (b_txt, b_c)):
            by_attr, pos = collections.defaultdict(list), 0
            for w, att in cs:
                i = txt.find(w, pos)
                by_attr[att].append({"surface": w, "start": i, "end": i + len(w)})
                pos = i + len(w)
            doc = R.nlp()(txt)
            outs.append(realize(txt, doc, by_attr, {}, [], va=True)[0])
        ok = outs[0] == outs[1]
        bad += not ok
        print(("ok  " if ok else "FAIL"), outs[0], "|", outs[1])
    print(f"{len(UNIT) - bad}/{len(UNIT)} value-invariance tests pass")
    return 1 if bad else 0


# ---------------------------------------------------------------- main
def fresh_base_spans(ds, layer, spans_file):
    """the frozen base-span layers with the fresh cohort's inputs (Addendum 6)"""
    if layer == "lf":
        ai = C0._load("ai", "src/anonymize/anonymize_intact.py")
        spans_of = ai.load_spans(ds, "longformer", spans_file)

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
        return base_spans
    s1 = C0._load("s1", "scripts/dev_simple_spans.py")
    import spacy
    nlp0 = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    occ = s1.OccLex("tight")
    return lambda did, text: [(x, y, f"[{t}]") for x, y, t in s1.s1_spans(text, nlp0, occ)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", choices=("medqa", "tab", "synthpai"))
    ap.add_argument("--fresh", choices=sorted(FRESH))
    ap.add_argument("--unit", action="store_true")
    a = ap.parse_args()
    if a.unit:
        return unit()
    t0 = time.time()
    fresh = a.fresh
    ds = a.fresh or a.ds
    if fresh:
        tree, corp, layer, spans_file = FRESH[fresh]
    else:
        tree, corp, layer, _b, _s = M1E.CFG[ds]
    dev = set() if fresh else {json.loads(l)["doc_id"] for l in open("runs/dev/phase8/e1/packets_v1.jsonl")
                               if json.loads(l)["corpus"] == ds}
    reg = registry.load(f"{tree}/registry.json")
    det = json.load(open(f"{tree}/detections_d1.json"))
    assert det["registry_hash"] == reg["_freeze_hash"]
    cues_of = {tuple(k.split("||")): v for k, v in det["cues"].items()}
    subs = collections.defaultdict(list)
    for s in reg["subjects"]:
        subs[s["doc_id"]].append(s)
    prof = profiles.for_registry(reg)
    SC = C0._load("sc", "scripts/dev_phase8_scope_control.py")
    base_spans = fresh_base_spans(ds, layer, spans_file) if fresh else C0.make_base(ds, M1E.CFG)[1]
    stored = {} if fresh else {k: {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"]
                                   for l in open(f"{MU}/arms/{ds}_{v}/anonymized.jsonl")} for k, v in STORED.items()}
    out_dir = f"{FRESH_OUT}/arms" if fresh else f"{OUT}/arms"
    trace_dir = f"{FRESH_OUT}/traces" if fresh else f"{OUT}/traces"
    arm_names = (("base", "SACC_M", "B2", "B6") + ARMS) if fresh else ARMS
    realization = {t: "mask" for t in TAGS}
    texts = {arm: [] for arm in arm_names}
    rows, st = [], collections.Counter()
    gate = collections.Counter()

    for line in open(corp):
        rec = json.loads(line)
        did, src = rec["metadata"]["data_id"], rec["text"]
        bsp = base_spans(did, src)
        base_txt = M1E.v1_sweep(src, bsp, [])
        m = M1E.offset_map(src, base_txt)
        doc = R._NLP(base_txt)
        S = list(subs.get(did, []))
        mentions = [x for s in S for x in s["gold_mention"]]
        st["docs"] += 1

        # ---- pools and union (Phase 16 emitter, line for line)
        pools = {"local": collections.defaultdict(list), "global": collections.defaultdict(list)}
        seen = set()
        for s in S:
            for t in TAGS:
                for c in cues_of.get((s["eval_subject_id"], t), []):
                    k = (c["start"], c["end"], t)
                    if k in seen:
                        continue
                    seen.add(k)
                    s0, e0 = c["start"], c["end"]
                    if s0 in m and (e0 - 1) in m and m[e0 - 1] + 1 - m[s0] == e0 - s0:
                        pools["local"][t].append({"surface": c["surface"], "start": m[s0],
                                                  "end": m[e0 - 1] + 1, "v1_subject": s["eval_subject_id"]})
        g = SC.global_groups(base_txt, list(TAGS), prof)
        for t in TAGS:
            for c in g.get(t, []):
                pools["global"][t].append({"surface": base_txt[c["start"]:c["end"]], "start": c["start"], "end": c["end"]})
        union = {}
        for pool in ("local", "global"):
            for t, cs in pools[pool].items():
                for c in cs:
                    key = (c["start"], c["end"], t)
                    if key not in union:
                        union[key] = dict(c, attribute=t, in_local=False, in_global=False, v1_subjects=[])
                    union[key]["in_" + pool] = True

        # ---- frozen binding and merge; C-REL2 and C-ALIAS-S re-inclusion flags
        pmap = C0.placeholder_map(src, bsp)
        sspans, _al = C0.subject_spans(src, base_txt, pmap, m, S)
        ph_spans = [(x.start(), x.end()) for x in PLACEHOLDER.finditer(base_txt)]
        subj_ph = {}
        for sid, ss in sspans.items():
            for x in ss:
                if x in ph_spans:
                    subj_ph.setdefault(x, set()).add(sid)
        for (s0, e0, t), c in sorted(union.items()):
            per = []
            for s in (S or [None]):
                mm = s["gold_mention"] if s else mentions
                tt = tuple(CUES._own_turns(base_txt, mm)) if getattr(prof, "turn_scope", False) else ()
                b = B.bind(doc, s0, e0, t, mm, ds, n_subjects=len(S) or 1, own_turns=tt, author=bool(prof.subject_is_author))
                per.append({"subject_id": s["eval_subject_id"] if s else None, **b})
            best = next((x for x in per if x["status"] in EDITABLE), per[0])
            c["_status"], c["_reason"], c["_ref"] = best["status"], best["rule_id"], best["direct_referent"]
            c["_nonperson"] = best["status"] == "NONPERSON_USE"
            # C-REL2 (S0 rule)
            c["_rel2"] = False
            if c["_status"] == "OTHER_ENTITY":
                lemma = re.sub(r"^(his|her|their|its|my|our|your|the|a|an)\s+", "", c["surface"].strip().lower())
                lemma = re.sub(r"'s$", "", lemma)
                if lemma in B.RELATIONAL or lemma in C0.GENDERED_KIN or lemma.rstrip("s") in C0.GENDERED_KIN:
                    r0 = per[0]
                    c["_rel2"] = r0["rule_id"] in ("R5_object_of_verb", "R6_bare_subject") or (
                        r0["rule_id"] == "R3_possessive" and (r0["direct_referent"] or "").strip().lower() in C0.POSS_PRON)
            # C-ALIAS-S (surface apposition to a subject-owned placeholder)
            c["_alias"] = False
            if c["_status"] == "OTHER_ENTITY":
                for (p0, p1), owners in subj_ph.items():
                    if len(owners) != 1:
                        continue
                    after = base_txt[p1:s0]
                    before = base_txt[e0:p0]
                    if (0 <= s0 - p1 <= 12 and re.fullmatch(r"\s*,\s*(?:(?:a|an|the)\s+)?", after)) or \
                            (0 <= p0 - e0 <= 3 and re.fullmatch(r"\s*,?\s*", before)):
                        c["_alias"] = True
                        break
            st["cand"] += 1
            st["rel2"] += c["_rel2"]
            st["alias"] += c["_alias"]

        def sel(rule, c):
            if rule == "B2" or rule == "B6":
                return True
            return c["_status"] not in EXCLUDED

        def by_attr_for(keep):
            ba = collections.defaultdict(list)
            for (s0, e0, t), c in union.items():
                if keep(t, c):
                    ba[t].append({"surface": c["surface"], "start": s0, "end": e0})
            return ba

        # ---- gate (a) on the evaluated corpora: all-SACC / all-B2 / all-B6 reproduce the stored arms;
        #      on a fresh cohort the same three rules are emitted as the reference arms instead
        ref_txt = {}
        for name, rule, rz in (("B4", "SACC", "mask"), ("B2", "B2", "mask"), ("B6", "B6", "ladder")):
            ba = by_attr_for(lambda t, c, rule=rule: sel(rule, c))
            txt = realize(base_txt, doc, ba, {t: rz for t in TAGS}, [])[0]
            ref_txt[name] = txt
            if fresh:
                continue
            ok = txt == stored[name].get(did)
            gate[f"a_{name}_identical"] += ok
            gate[f"a_{name}_checked"] += 1
            if not ok and gate[f"a_{name}_fail_examples"] < 3:
                gate[f"a_{name}_fail_examples"] += 1
                print(f"  gate (a) {name} differs on {did}")

        # ---- SACC-M reference plan
        sacc_ba = by_attr_for(lambda t, c: sel("SACC", c))
        sacc_plan = plan_items(base_txt, sacc_ba)
        sacc_spans = plan_spans(sacc_plan)
        sacc_sel = {k for k, c in union.items() if sel("SACC", c)}

        # ---- V2-U selection: the SACC-M rule plus C-REL2 and C-ALIAS-S re-inclusion, identical on every corpus
        def keep_v2u(t, c):
            return sel("SACC", c) or c["_rel2"] or c["_alias"]

        def extras_for(plan):
            taken = plan_spans(plan) + ph_spans
            ex = []
            for mm in SEXCLIN_RE.finditer(base_txt):
                if not C0.overlaps(mm.start(), mm.end(), taken + [(x["start"], x["end"]) for x in ex]):
                    ex.append(mask_extra(mm.start(), mm.end(), base_txt, "sexclin"))
            ex += emoji_extras(base_txt, taken + [(x["start"], x["end"]) for x in ex])
            return ex

        outs = {}
        full_ba = by_attr_for(keep_v2u)
        full_plan = plan_items(base_txt, full_ba)
        full_extras = extras_for(full_plan)
        outs["V2U"], _p, full_planned = realize(base_txt, doc, full_ba, realization, full_extras)
        outs["V2U_VA"], _vp, va_planned = realize(base_txt, doc, full_ba, realization, full_extras, va=True)
        edits_of = {"V2U": None, "V2U_VA": va_planned}

        # ---- R0–R4 (Addendum 2a pool)
        full_spans = plan_spans(full_plan) + [(x["start"], x["end"]) for x in full_extras if x["rule"] == "sexclin"]
        n_add = sum(1 for sp in full_spans if not any(sp[0] < y and x < sp[1] for x, y in sacc_spans))
        cand_units = sorted({(s0, e0) for (s0, e0, t) in union if (s0, e0, t) not in sacc_sel
                             and not C0.overlaps(s0, e0, sacc_spans)})
        clin_units = [(mm.start(), mm.end()) for mm in SEXCLIN_RE.finditer(base_txt)]
        tok_units = [(tk.idx, tk.idx + len(tk.text)) for tk in doc if tk.pos_ in ("NOUN", "PROPN", "ADJ", "VERB")
                     and tk.is_alpha and len(tk.text) >= 3]
        blocked = sacc_spans + ph_spans
        pool, pool_seen = [], set()
        for u, kind in [(u, "cand") for u in cand_units] + [(u, "clin") for u in clin_units] + [(u, "tok") for u in tok_units]:
            if kind != "cand" and C0.overlaps(u[0], u[1], blocked):
                continue
            if u not in pool_seen:
                pool_seen.add(u)
                pool.append((u, kind))
        for k in range(5):
            rng = random.Random(f"SACCV2|R{k}|{did}")
            draw = rng.sample(pool, min(n_add, len(pool))) if n_add else []
            chosen_c = {u for u, kind in draw if kind == "cand"}
            ba = by_attr_for(lambda t, c, s0e0=None: sel("SACC", c))
            for (s0, e0, t), c in union.items():
                if (s0, e0) in chosen_c and not sel("SACC", c):
                    ba[t].append({"surface": c["surface"], "start": s0, "end": e0})
            rplan = plan_items(base_txt, ba)
            taken = plan_spans(rplan) + ph_spans
            ex = []
            for u, kind in draw:
                if kind in ("clin", "tok") and not C0.overlaps(u[0], u[1], taken + [(x["start"], x["end"]) for x in ex]):
                    ex.append(mask_extra(u[0], u[1], base_txt, "random_" + kind))
            ex += emoji_extras(base_txt, taken + [(x["start"], x["end"]) for x in ex])
            outs[f"V2U_R{k}"], _rp, rplanned = realize(base_txt, doc, ba, realization, ex)
            edits_of[f"V2U_R{k}"] = rplanned
            st[f"R{k}_drawn"] += len(draw)

        if fresh:
            outs["base"], outs["SACC_M"], outs["B2"], outs["B6"] = base_txt, ref_txt["B4"], ref_txt["B2"], ref_txt["B6"]

        # ---- gates (b), (d) and manifest
        for arm, txt in outs.items():
            texts[arm].append({"metadata": {"data_id": did}, "anonymized_text": txt})
            rows.append({"corpus": ds, "doc_id": did, "in_dev": int(did in dev), "arm": arm, "base_chars": len(base_txt),
                         "chars_removed": len(base_txt) - len(txt), "output_sha16": h16(txt),
                         "differs_from_sacc_m": int(txt != (ref_txt["B4"] if fresh else stored["B4"].get(did)))})
            gate["d_redacted_in_placeholder"] += len(re.findall(r"\[[A-Z_]*\[REDACTED\]", txt))
        edits_of["V2U"] = full_planned
        gate["b_sacc_spans"] += len(sacc_spans)
        for arm, pl in edits_of.items():
            ed = [(x["start"], x["end"]) for x in pl]
            gate[f"b_covered_{arm}"] += sum(1 for x, y in sacc_spans if C0.overlaps(x, y, ed))
        st["n_add"] += n_add
        st["full_edits"] += len(full_planned)
        st["sacc_edits"] += len(sacc_spans)

    # ---- write (atomic)
    empty = [arm for arm, recs in texts.items() if not recs]
    if empty or st["docs"] == 0:
        sys.exit(f"refusing to write: {st['docs']} documents processed and these arms are empty: {empty}. "
                 "An emitted arm is never written with zero rows; a partial or instrumented run must not overwrite one.")
    for arm, recs in texts.items():
        d = f"{out_dir}/{ds}_{arm}"
        os.makedirs(d, exist_ok=True)
        with open(f"{d}/anonymized.jsonl.tmp", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(f"{d}/anonymized.jsonl.tmp", f"{d}/anonymized.jsonl")
    os.makedirs(trace_dir, exist_ok=True)
    with open(f"{trace_dir}/{ds}_manifest.csv.tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(f"{trace_dir}/{ds}_manifest.csv.tmp", f"{trace_dir}/{ds}_manifest.csv")
    gates_hold = (fresh or all(gate[f"a_{n}_identical"] == gate[f"a_{n}_checked"] for n in ("B4", "B2", "B6"))) and \
        all(gate[f"b_covered_{arm}"] == gate["b_sacc_spans"] for arm in ARMS) and \
        gate["d_redacted_in_placeholder"] == 0
    run = {"ds": ds, "prereg": "runs/dev/preregs/saccv2_stack_prereg.md (Addendum 4, 2a, 2b)", "seconds": round(time.time() - t0, 1),
           "design": "V2-U unified: SACC-M binding + C-REL2 + C-ALIAS-S + C-SEXCLIN + C-EMOJI, mask realization; no corpus-specific rule",
           "counts": dict(st), "gates": dict(gate),
           "gates_hold": gates_hold,
           "code_sha16": {p: sha16_file(p) for p in ("scripts/dev_saccv2_s1_emit.py", "scripts/dev_saccv2_s0_census.py",
                                                     "scripts/dev_phase10_repair.py", "scripts/dev_phase10_binding.py",
                                                     "scripts/dev_phase10_emit_m1.py")},
           "lexicon_sha16": {"va_v0.json": sha16_file(f"{OUT}/lexicons/va_v0.json"),
                             "sexclin_v0.json": sha16_file("runs/dev/saccv2_stack/s0/lexicons/sexclin_v0.json")},
           "fresh": fresh, "tree": tree, "gold": corp, "layer": layer,
           "arm_sha16": {arm: sha16_file(f"{out_dir}/{ds}_{arm}/anonymized.jsonl") for arm in arm_names}}
    with open(f"{trace_dir}/{ds}_emit_run.json.tmp", "w") as fh:
        json.dump(run, fh, indent=1)
    os.replace(f"{trace_dir}/{ds}_emit_run.json.tmp", f"{trace_dir}/{ds}_emit_run.json")
    print(json.dumps({"ds": ds, "gates_hold": gates_hold, "gates": dict(gate), "counts": dict(st),
                      "seconds": run["seconds"]}, indent=1))
    return 0 if gates_hold else 1


if __name__ == "__main__":
    sys.exit(main())
