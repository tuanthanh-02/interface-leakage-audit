#!/usr/bin/env python
"""Phase 10 P1 -- emit M1 for the DEV_REGRESSION_238 documents, whole-document.

A packet is one (subject, attribute) pair, but base_text and sacc_text are whole-document arms,
so a per-packet edit set can never reproduce them. M1 is therefore built the way the M0 arm was:
every subject, every tag of a document at once.

The method contract has the parser read the BASE text the method receives. So the base spans are
applied first, the resulting base text is parsed, cue offsets are mapped source -> base, and the
repair is applied to the base text. Two reproductions are checked on the same path:

    rebuilt base   == the reported base arm            (the span layer is being read correctly)
    v1 on base     == the reported sacc arm (M0)       (the executor path is equivalent to v1's)

Writes outputs/{M0_basepath,M1}.jsonl, traces/edit_trace_M1.jsonl, results/coverage.csv.
"""
import argparse, collections, csv, difflib, json, os, re, sys
import importlib.util as il
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, "scripts")
from src.relaware import registry
from src.relaware.method import edits as V1
import dev_phase10_repair as R
D = "runs/dev/sacc_improvement"; assert D.startswith("runs/dev/")
TAGS = ("SEX", "OCCUPATION", "RELATIONSHIP")
CFG = {"synthpai": ("runs/relaware_synthpai", "data/spia/spia_synthpai.jsonl", "lf", "lf_only", "sacc_lf"),
       "tab": ("runs/relaware", "data/spia/spia_tab_144.jsonl", "lf", "lf_only", "sacc_lf"),
       "medqa": ("runs/relaware_medqa", "data/spia/spia_medqa.jsonl", "s1", "s1_only", "sacc_s1"),
       "panorama": ("runs/relaware_panorama", "data/spia/spia_panorama_151.jsonl", "lf", "lf_only", "sacc_lf")}


def _load(n, p):
    s = il.spec_from_file_location(n, p); m = il.module_from_spec(s); s.loader.exec_module(m); return m


def v1_sweep(text, base, cue_edits):
    """the frozen executor's behaviour: base first, then attribute priority, longest first."""
    ed = [(s, e, r, -1, e - s) for s, e, r in base]
    ed += [(s, e, r, V1._ATTR_PRIORITY.get(a, 9), e - s) for s, e, r, a in cue_edits]
    acc = []
    for s, e, r, pri, ln in sorted(ed, key=lambda x: (x[3], -x[4])):
        if all(e <= a or s >= b for a, b, _ in acc):
            acc.append((s, e, r))
    out = text
    for s, e, r in sorted(acc, key=lambda x: -x[0]):
        if r == "" and e < len(out) and out[e:e + 1] == " ":
            e += 1
        if r[:2].lower() in ("a ", "an") and re.search(r"\b(a|an|the)\s+$", out[:s], re.I):
            r = r.split(" ", 1)[1]
        out = out[:s] + r + out[e:]
    return out


def offset_map(src, base):
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src, base, autojunk=False).get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                m[i1 + k] = j1 + k
    return m


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ds", required=True, choices=sorted(CFG)); a = ap.parse_args()
    tree, corp, layer, basearm, saccarm = CFG[a.ds]
    want = {json.loads(l)["doc_id"] for l in open("runs/dev/phase8/e1/packets_v1.jsonl")
            if json.loads(l)["corpus"] == a.ds}
    reg = registry.load(f"{tree}/registry.json"); det = json.load(open(f"{tree}/detections_d1.json"))
    assert det["registry_hash"] == reg["_freeze_hash"], "detections do not match the frozen registry"
    cues_of = {tuple(k.split("||")): v for k, v in det["cues"].items()}
    subs = collections.defaultdict(list)
    for s in reg["subjects"]:
        subs[s["doc_id"]].append(s)
    if layer == "lf":
        ai = _load("ai", "src/anonymize/anonymize_intact.py"); spans_of = ai.load_spans(a.ds, "longformer", None)
        def base_spans(did, text):
            bs = []
            for e in spans_of.get(did) or []:
                s0, e0 = e["start_offset"], e["end_offset"]
                if 0 <= s0 < e0 <= len(text):
                    bs.append((s0, e0, f"[{e.get('entity_type', 'MISC')}]"))
            bs.sort(key=lambda x: (x[0], -(x[1] - x[0]))); kept, last = [], -1
            for x in bs:
                if x[0] >= last:
                    kept.append(x); last = x[1]
            return kept
    else:
        s1 = _load("s1", "scripts/dev_simple_spans.py"); import spacy
        nlp0 = spacy.load("en_core_web_sm", disable=["lemmatizer"]); occ = s1.OccLex("tight")
        def base_spans(did, text):
            return [(a0, b0, f"[{t}]") for a0, b0, t in s1.s1_spans(text, nlp0, occ)]
    rep_base = {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"]
                for l in open(f"runs/dev/arms/{a.ds}_{basearm}/anonymized.jsonl")}
    rep_sacc = {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"]
                for l in open(f"runs/dev/arms/{a.ds}_{saccarm}/anonymized.jsonl")}

    st = collections.Counter(); m0_rows, m1_rows, trace = [], [], []
    for line in open(corp):
        rec = json.loads(line); did, text = rec["metadata"]["data_id"], rec["text"]
        if did not in want:
            continue
        bs = base_spans(did, text)
        base_txt = v1_sweep(text, bs, [])
        st["docs"] += 1; st["base_reproduced"] += (base_txt == rep_base.get(did))
        m = offset_map(text, base_txt)
        seen, by_attr, dropped = set(), collections.defaultdict(list), 0
        for s in subs.get(did, []):
            for t in TAGS:
                for c in cues_of.get((s["eval_subject_id"], t), []):
                    k = (c["start"], c["end"], t)
                    if k in seen:
                        continue
                    seen.add(k)
                    s0, e0 = c["start"], c["end"]
                    if s0 not in m or (e0 - 1) not in m or m[e0 - 1] + 1 - m[s0] != e0 - s0:
                        dropped += 1; continue
                    by_attr[t].append({"surface": c["surface"], "start": m[s0], "end": m[e0 - 1] + 1})
        st["cues_mapped"] += sum(len(v) for v in by_attr.values()); st["cues_inside_mask"] += dropped
        # --- M0 on the same base-text path, as the reference for P1
        v1_edits = [(c["start"], c["end"],
                     V1.replacement(t, c, 1, V1._next_word(base_txt, c["end"])), t)
                    for t, cs in by_attr.items() for c in cs]
        m0_txt = v1_sweep(base_txt, [], v1_edits)
        st["m0_reproduced"] += (m0_txt == rep_sacc.get(did))
        # --- M1
        m1_txt, planned = R.repair(base_txt, dict(by_attr))
        for q in planned:
            st[q["action"]] += 1
            trace.append({"corpus": a.ds, "doc_id": did, "attribute": q["attribute"],
                          "surface": q["surface"], "base_start": q["start"], "base_end": q["end"],
                          "action": q["action"], "rule": q["rule"], "replacement": q["replacement"],
                          "applied_start": (q.get("applied") or [None, None])[0],
                          "applied_end": (q.get("applied") or [None, None])[1],
                          "companion_edits": [[x, y, r, w] for x, y, r, w in q["companion"]]})
            st["companion_edits"] += len(q["companion"])
        m0_rows.append({"corpus": a.ds, "doc_id": did, "arm": "M0_basepath", "text": m0_txt})
        m1_rows.append({"corpus": a.ds, "doc_id": did, "arm": "M1", "text": m1_txt})

    def append(path, rows):
        old = [l for l in open(path)] if os.path.exists(path) else []
        old = [l for l in old if json.loads(l).get("corpus") != a.ds]
        with open(path + ".tmp", "w") as fh:
            fh.writelines(old)
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
    append(f"{D}/outputs/M0_basepath.jsonl", m0_rows)
    append(f"{D}/outputs/M1.jsonl", m1_rows)
    append(f"{D}/traces/edit_trace_M1.jsonl", trace)
    print(f"[{a.ds}] docs {st['docs']}  base reproduced {st['base_reproduced']}/{st['docs']}"
          f"  M0 reproduced {st['m0_reproduced']}/{st['docs']}")
    print(f"        cues mapped {st['cues_mapped']}  inside a mask {st['cues_inside_mask']}"
          f"  -> REPLACE {st['REPLACE']}  MASK {st['MASK']}  KEEP {st['KEEP']}"
          f"  companion verb edits {st['companion_edits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
