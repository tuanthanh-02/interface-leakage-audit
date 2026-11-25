#!/usr/bin/env python
"""Phase 10 P2 -- emit M2-local and M2 (plan §5.4), with the common candidate inventory.

    M1        v1's candidates, v1's same-sentence scope, M1 replacer          (already emitted)
    M2-local  v1's candidates, NEW binding, same replacer
    M2        whole-document candidates from the SAME lexicon, NEW binding, same replacer

The inventory is written before any arm selects anything, so precision and recall are computed
over one pool rather than over each method's own output (plan §9.4).

Writes traces/candidate_inventory.jsonl, traces/binding_trace.jsonl,
outputs/M2_local.jsonl, outputs/M2.jsonl.
"""
import argparse, collections, difflib, json, os, sys
import importlib.util as il
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(REPO)
sys.path.insert(0, REPO); sys.path.insert(0, "scripts")
from src.relaware import registry, profiles
from src.relaware.method import edits as V1
from src.relaware.method import cues as CUES
import dev_phase10_repair as R, dev_phase10_binding as B, dev_phase10_lexext as LX
import dev_phase10_emit_m1 as M1E
D = "runs/dev/sacc_improvement"; assert D.startswith("runs/dev/")
TAGS = ("SEX", "OCCUPATION", "RELATIONSHIP")


def _load(n, p):
    s = il.spec_from_file_location(n, p); m = il.module_from_spec(s); s.loader.exec_module(m); return m


SC = _load("sc", "scripts/dev_phase8_scope_control.py")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True, choices=sorted(M1E.CFG))
    ap.add_argument("--lexext", default="none", choices=["none", "x1", "x1x2"],
                    help="occupation lexicon extension (preregs/phase10_p4_lexicon_prereg.md): "
                         "x1 = mechanical variants only, x1x2 = plus the DEV-fitted role nouns")
    ap.add_argument("--all", action="store_true",
                    help="every document of the corpus, not only DEV_REGRESSION_238 "
                         "(preregs/phase10_confirmation_prereg.md)")
    a = ap.parse_args()
    tree, corp, layer, basearm, saccarm = M1E.CFG[a.ds]
    dev = {json.loads(l)["doc_id"] for l in open("runs/dev/phase8/e1/packets_v1.jsonl")
           if json.loads(l)["corpus"] == a.ds}
    want = None if a.all else dev          # None = every document
    reg = registry.load(f"{tree}/registry.json")
    det = json.load(open(f"{tree}/detections_d1.json"))
    assert det["registry_hash"] == reg["_freeze_hash"]
    cues_of = {tuple(k.split("||")): v for k, v in det["cues"].items()}
    subs = collections.defaultdict(list)
    for s in reg["subjects"]:
        subs[s["doc_id"]].append(s)
    prof = profiles.for_registry(reg)          # the same accessor dev_phase8_scope_control uses
    if a.lexext != "none":
        # frozen profiles.py is NOT edited; the sweep is handed a shim whose occ_role is the union
        pat, terms = LX.extension_pattern(a.ds, include_x2=(a.lexext == "x1x2"))
        lex = dict(prof.lex); lex["occ_role"] = f"(?:{lex['occ_role']}|{pat})"
        prof = type("ExtProfile", (), {"lex": lex, "name": prof.name + "+" + a.lexext,
                                       "turn_scope": getattr(prof, "turn_scope", False),
                                       "subject_is_author": prof.subject_is_author})()
        print(f"  lexicon extension {a.lexext}: +{len(terms)} terms")
    ARM_SUFFIX = "" if a.lexext == "none" else ("_M3lex" if a.lexext == "x1x2" else "_M3lexX1")
    if layer == "lf":
        ai = _load("ai", "src/anonymize/anonymize_intact.py"); spans_of = ai.load_spans(a.ds, "longformer", None)
        def base_spans(did, text):
            bs = []
            for e in spans_of.get(did) or []:
                s0, e0 = e["start_offset"], e["end_offset"]
                if 0 <= s0 < e0 <= len(text): bs.append((s0, e0, f"[{e.get('entity_type','MISC')}]"))
            bs.sort(key=lambda x: (x[0], -(x[1] - x[0]))); kept, last = [], -1
            for x in bs:
                if x[0] >= last: kept.append(x); last = x[1]
            return kept
    else:
        s1 = _load("s1", "scripts/dev_simple_spans.py"); import spacy
        nlp0 = spacy.load("en_core_web_sm", disable=["lemmatizer"]); occ = s1.OccLex("tight")
        def base_spans(did, text):
            return [(x, y, f"[{t}]") for x, y, t in s1.s1_spans(text, nlp0, occ)]

    inv, trace, out_local, out_m2 = [], [], [], []
    out_m1b, out_m2b, out_c1, out_c2 = [], [], [], []
    st = collections.Counter()
    for line in open(corp):
        rec = json.loads(line); did, text = rec["metadata"]["data_id"], rec["text"]
        if want is not None and did not in want:
            continue
        st["dev" if did in dev else "untouched"] += 1
        base_txt = M1E.v1_sweep(text, base_spans(did, text), [])
        m = M1E.offset_map(text, base_txt)
        doc = R.nlp()(base_txt)
        S = subs.get(did, [])
        mentions = [x for s in S for x in s["gold_mention"]]
        st["docs"] += 1
        # ---- the two candidate pools, both in base coordinates
        pools = {"local": collections.defaultdict(list), "global": collections.defaultdict(list)}
        seen = set()
        for s in S:
            for t in TAGS:
                for c in cues_of.get((s["eval_subject_id"], t), []):
                    k = (c["start"], c["end"], t)
                    if k in seen: continue
                    seen.add(k)
                    s0, e0 = c["start"], c["end"]
                    if s0 in m and (e0 - 1) in m and m[e0 - 1] + 1 - m[s0] == e0 - s0:
                        pools["local"][t].append({"surface": c["surface"], "start": m[s0],
                                                  "end": m[e0 - 1] + 1,
                                                  "v1_subject": s["eval_subject_id"]})
        g = SC.global_groups(base_txt, list(TAGS), prof)      # already in base coordinates
        for t in TAGS:
            for c in g.get(t, []):
                pools["global"][t].append({"surface": base_txt[c["start"]:c["end"]],
                                           "start": c["start"], "end": c["end"]})
        # ---- bind every candidate of the union, once, and record it
        union = {}
        for pool in ("local", "global"):
            for t, cs in pools[pool].items():
                for c in cs:
                    key = (c["start"], c["end"], t)
                    if key not in union:
                        union[key] = dict(c, attribute=t, in_local=False, in_global=False,
                                          v1_subjects=[])
                    union[key]["in_" + pool] = True
                    if c.get("v1_subject") and c["v1_subject"] not in union[key]["v1_subjects"]:
                        union[key]["v1_subjects"].append(c["v1_subject"])
        EDITABLE = ("DIRECT_TARGET", "RELATIONAL_SHARED")
        for (s0, e0, t), c in sorted(union.items()):
            # binding is per SUBJECT: "is this cue about X" has one answer per protected person.
            # The document-level action is to edit if it is editable for ANY of them.
            per = []
            for s in (S or [None]):
                mm = s["gold_mention"] if s else mentions
                tt = tuple(CUES._own_turns(base_txt, mm)) if getattr(prof, "turn_scope", False) else ()
                b = B.bind(doc, s0, e0, t, mm, a.ds, n_subjects=len(S) or 1,
                           own_turns=tt, author=bool(prof.subject_is_author))
                per.append({"subject_id": s["eval_subject_id"] if s else None, **b})
            best = next((x for x in per if x["status"] in EDITABLE), per[0])
            st["cand"] += 1; st[best["status"]] += 1
            row = {"corpus": a.ds, "doc_id": did, "cue_id": f"{did}|{s0}|{e0}|{t}",
                   "base_offsets": [s0, e0], "cue_text": c["surface"], "attribute": t,
                   "in_v1_scope": c["in_local"], "in_global_sweep": c["in_global"],
                   "syntax_role": best["syntax_role"], "direct_referent": best["direct_referent"],
                   "supported_subject_attribute_pairs": best["supported_pairs"],
                   "binding_status": best["status"], "rule_id": best["rule_id"],
                   "bound_subject_id": best["subject_id"],
                   "v1_subject_ids": c.get("v1_subjects", []),
                   "per_subject": [{"subject_id": x["subject_id"], "status": x["status"],
                                    "rule_id": x["rule_id"]} for x in per],
                   "support_offsets": [s0, e0]}
            inv.append(row); trace.append(row)
            c["_status"] = best["status"]
        # ---- select and apply
        # round 1 (positive): edit only what binding proves belongs to the subject
        # round 2 (subtractive, preregs/phase10_p2r2_subtractive_prereg.md): keep v1's selection
        #          and remove only what binding proves belongs to someone else or to an institution
        POSITIVE = {"DIRECT_TARGET", "RELATIONAL_SHARED"}
        EXCLUDED = {"OTHER_ENTITY", "NONPERSON_USE"}
        for arm, pool_key, mode, sink in (("M2_local", "in_local", "positive", out_local),
                                          ("M2", None, "positive", out_m2),
                                          ("M1b", "in_local", "subtractive", out_m1b),
                                          ("M2b", None, "subtractive", out_m2b),
                                          ("C1", None, "mask", out_c1),
                                          ("C2", None, "unbound", out_c2)):
            by_attr = collections.defaultdict(list)
            for (s0, e0, t), c in union.items():
                if mode == "positive":
                    keep = c["_status"] in POSITIVE
                elif mode == "unbound":
                    keep = True                                 # C2: no subject binding at all
                else:
                    keep = c["_status"] not in EXCLUDED          # subtractive; "mask" selects as M2b
                if not keep:
                    continue
                if pool_key and not c[pool_key]:
                    continue
                by_attr[t].append({"surface": c["surface"], "start": s0, "end": e0})
            if mode == "mask":
                # C1 must edit EXACTLY the spans M2b edits, not merely the same selection: M2b's
                # replacer leaves some selected cues unedited (KEEP), and masking those would make
                # C1 delete more than the ladder ever touched. So M2b's plan is computed first and
                # its non-KEEP spans are the ones masked.
                _t, m2b_plan = R.repair(base_txt, dict(by_attr))
                planned = [dict(start=(q.get("applied") or [q["start"], q["end"]])[0],
                                end=(q.get("applied") or [q["start"], q["end"]])[1],
                                surface=q["surface"], attribute=q["attribute"], companion=[],
                                rule="C1_fixed_placeholder", action="REPLACE",
                                replacement=R.MASK)
                           for q in m2b_plan if q["action"] != "KEEP"]
                txt = R.apply_edits(base_txt, planned)
            else:
                txt, planned = R.repair(base_txt, dict(by_attr))
            st[f"{arm}_edits"] += sum(1 for p in planned if p["action"] != "KEEP")
            st[f"{arm}_chars"] += len(base_txt) - len(txt)
            sink.append({"corpus": a.ds, "doc_id": did, "arm": arm, "text": txt})

    def append(path, rows, key="corpus"):
        old = [l for l in open(path)] if os.path.exists(path) else []
        old = [l for l in old if json.loads(l).get(key) != a.ds]
        with open(path + ".tmp", "w") as fh:
            fh.writelines(old)
            for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
    append(f"{D}/traces/candidate_inventory.jsonl", inv)
    append(f"{D}/traces/binding_trace.jsonl", trace)
    append(f"{D}/outputs/M2_local.jsonl", out_local)
    append(f"{D}/outputs/M2.jsonl", out_m2)
    append(f"{D}/outputs/M1b.jsonl", out_m1b)
    append(f"{D}/outputs/M2b.jsonl", out_m2b)
    if ARM_SUFFIX:
        name = "M3lex" if a.lexext == "x1x2" else "M3lexX1"
        append(f"{D}/outputs/{name}.jsonl",
               [dict(r, arm=name) for r in out_c1])
        print(f"        extension arm {name}: {sum(1 for _ in out_c1)} documents")
        return 0
    append(f"{D}/outputs/C1.jsonl", out_c1)
    append(f"{D}/outputs/C2.jsonl", out_c2)
    print(f"[{a.ds}] docs {st['docs']} (dev {st['dev']} / untouched {st['untouched']})  candidates {st['cand']}  "
          f"DIRECT {st['DIRECT_TARGET']}  RELATIONAL {st['RELATIONAL_SHARED']}  "
          f"OTHER {st['OTHER_ENTITY']}  NONPERSON {st['NONPERSON_USE']}  AMBIG {st['AMBIGUOUS']}")
    print(f"        edits applied: M2_local {st['M2_local_edits']}  M2 {st['M2_edits']}"
          f"  |  M1b {st['M1b_edits']}  M2b {st['M2b_edits']}  C1 {st['C1_edits']}  C2 {st['C2_edits']}"
          f"  |  chars removed: M2b {st['M2b_chars']}  C1 {st['C1_chars']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
