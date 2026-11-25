#!/usr/bin/env python
"""Phase 8 §7 -- scope control arms. Prereg: runs/dev/preregs/scope_control_prereg.md

  --ds synthpai   byte-identity check: global lexicon (no scope) vs sacc_lf
  --ds medqa      EMIT runs/dev/arms/medqa_global_lex_l1 over the S1 base
  --ds tab        verify the stored arm reproduces; no re-emit
  --results       write scope_control_manifest.json + scope_control_results.csv

Frozen modules imported, never modified: profiles, cues._find_all,
edits.replacement / apply_bundle, the S1 span layer, load_spans.
"""
import argparse, collections, csv, hashlib, importlib.util as il, json, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO); sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.relaware import profiles, registry
from src.relaware.method import cues, edits

TAGS = ("SEX", "OCCUPATION", "RELATIONSHIP")
CFG = {"tab": ("runs/relaware", "data/spia/spia_tab_144.jsonl", "lf"),
       "synthpai": ("runs/relaware_synthpai", "data/spia/spia_synthpai.jsonl", "lf"),
       "medqa": ("runs/relaware_medqa", "data/spia/spia_medqa.jsonl", "s1")}

def _load(n, p):
    s = il.spec_from_file_location(n, p); m = il.module_from_spec(s); s.loader.exec_module(m); return m
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
def load_arm(p): return {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"] for l in open(p)}

def global_groups(text, protected, prof):
    """detect_groups with the scope removed: the lexicon over the whole text.
    Byte-identical to scripts/dev_e3_control_arms.py:global_groups."""
    lex = prof.lex; g = {a: [] for a in protected}; a, b = 0, len(text)
    if "SEX" in protected:
        for pat, kind in ((lex["pronoun"], "pronoun"), (lex["title"], "title"), (lex["gendered_kin"], "gendered_kin")):
            g["SEX"] += cues._find_all(text, a, b, pat, kind)
    if "RELATIONSHIP" in protected: g["RELATIONSHIP"] += cues._find_all(text, a, b, lex["kinship"], "kinship")
    if "OCCUPATION" in protected: g["OCCUPATION"] += cues._find_all(text, a, b, lex["occ_role"], "role")
    for k, cs in g.items():
        cs.sort(key=lambda c: (c["start"], -(c["end"] - c["start"]))); kept, last = [], -1
        for c in cs:
            if c["start"] >= last: kept.append(c); last = c["end"]
        g[k] = kept
    return g

def base_spans(ds, did, text, spans_of, s1, nlp, occ):
    if CFG[ds][2] == "s1":
        return [(a0, b0, f"[{t}]") for a0, b0, t in s1.s1_spans(text, nlp, occ)]
    bs = []
    for e in spans_of.get(did) or []:
        s0, e0 = e["start_offset"], e["end_offset"]
        if 0 <= s0 < e0 <= len(text): bs.append((s0, e0, f"[{e.get('entity_type', 'MISC')}]"))
    bs.sort(key=lambda x: (x[0], -(x[1] - x[0]))); kept, last = [], -1
    for x in bs:
        if x[0] >= last: kept.append(x); last = x[1]
    return kept

def emit(ds, write):
    tree, corp, layer = CFG[ds]
    reg = registry.load(f"{tree}/registry.json"); prof = profiles.for_registry(reg)
    det = json.load(open(f"{tree}/detections_d1.json")); assert det["registry_hash"] == reg["_freeze_hash"]
    cues_of = {tuple(k.split("||")): v for k, v in det["cues"].items()}
    subs = collections.defaultdict(list)
    for s in reg["subjects"]: subs[s["doc_id"]].append(s)
    prot = collections.defaultdict(set)
    for e in reg["entries"]: prot[e["eval_subject_id"]].add(e["attribute"])
    spans_of = s1 = nlp = occ = None
    if layer == "lf":
        ai = _load("ai", "src/anonymize/anonymize_intact.py"); spans_of = ai.load_spans(ds, "longformer", None)
    else:
        s1 = _load("s1", "scripts/dev_simple_spans.py"); import spacy
        nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"]); occ = s1.OccLex("tight")
    sacc_arm = f"runs/dev/arms/{ds}_sacc_{layer}/anonymized.jsonl"; sacc = load_arm(sacc_arm)
    out_dir = f"runs/dev/arms/{ds}_global_lex_l1"; assert out_dir.startswith("runs/dev/")
    st = collections.Counter(); texts = {}; n_docs = 0; identical = 0
    d1_spans = collections.defaultdict(set)
    for (sid, t), cs in cues_of.items():
        for c in cs: d1_spans[sid.split(":")[0]].add((c["start"], c["end"]))
    for line in open(corp):
        rec = json.loads(line); did, text = rec["metadata"]["data_id"], rec["text"]; n_docs += 1
        bs = base_spans(ds, did, text, spans_of, s1, nlp, occ)
        P = set().union(*[prot[s["eval_subject_id"]] for s in subs.get(did, [])]) if subs.get(did) else set()
        g = global_groups(text, P, prof) if P else {}
        g = {k: v for k, v in g.items() if v}
        gl = {(c["start"], c["end"]) for v in g.values() for c in v}
        st["global_edits"] += len(gl); st["d1_edits"] += len(d1_spans[did])
        st["global_not_in_d1"] += len(gl - d1_spans[did]); st["d1_not_in_global"] += len(d1_spans[did] - gl)
        anon = edits.apply_bundle(text, g, {t: 1 for t in g}, bs)      # FROZEN executor
        texts[did] = {**rec, "anonymized_text": anon}
        identical += (anon == sacc.get(did))
    st["docs"] = n_docs; st["docs_identical_to_sacc"] = identical
    print(ds, dict(st))
    if write:
        os.makedirs(out_dir, exist_ok=True)
        with open(f"{out_dir}/anonymized.jsonl.tmp", "w") as fh:
            for did, r in texts.items(): fh.write(json.dumps(r) + "\n")
        os.replace(f"{out_dir}/anonymized.jsonl.tmp", f"{out_dir}/anonymized.jsonl")
        json.dump({"arm": "global_lex_l1", "ds": ds, "base": f"{ds}_{layer}_only", "scope": "OFF (whole document)",
                   "lexicon": f"profiles.PROFILES['{prof.name}'].lex (frozen)", "replacement": "edits.replacement L1 (frozen)",
                   "executor": "edits.apply_bundle (frozen)", "registry_hash": reg["_freeze_hash"],
                   "prereg": "runs/dev/preregs/scope_control_prereg.md", "stats": dict(st)},
                  open(f"{out_dir}/emit_meta.json", "w"), indent=1)
        print("->", out_dir)
    else:
        # TAB: compare with the stored arm instead of overwriting it
        if os.path.exists(f"{out_dir}/anonymized.jsonl"):
            stored = load_arm(f"{out_dir}/anonymized.jsonl")
            print(ds, "stored arm reproduced on", sum(1 for d in texts if texts[d]["anonymized_text"] == stored.get(d)), "/", n_docs, "documents")
    return st

def results():
    rows = list(csv.DictReader(open("runs/dev/phase8/corrected_contrasts.csv")))
    def get(cid, ds):
        for r in rows:
            if r["contrast_id"] == cid and r["corpus"] == ds: return r
    out = []
    for ds in ("tab", "medqa", "synthpai"):
        gm = f"runs/dev/arms/{ds}_global_lex_l1/emit_meta.json"
        meta = json.load(open(gm)) if os.path.exists(gm) else {}
        stt = meta.get("stats", {})
        base = {"corpus": ds, "family": "llama-3.1-8b-instruct", "sacc_version": "SACC-Core v1",
                "global_edits": stt.get("global_edits") or stt.get("global_lex_edits"), "sacc_edits": stt.get("d1_edits") or stt.get("same_cue_mask_edits"),
                "global_edits_outside_sacc_scope": stt.get("global_not_in_d1"), "equal_budget": False}
        if ds == "synthpai":
            out.append(dict(base, contrast="sacc_lf:global_lex_l1", status="STRUCTURAL_IDENTITY_CHECK", docs_identical=stt.get("docs_identical_to_sacc"), docs=stt.get("docs")))
            continue
        for cid, label in (("scope_sacc_to_global", "sacc:global_lex_l1"), ("mask_to_global", "mask:global_lex_l1"), ("samecue_to_global", "same_cue_mask:global_lex_l1"), ("mask_to_sacc_e3pass", "mask:sacc (same pass)")):
            r = get(cid, ds)
            if r is None:
                out.append(dict(base, contrast=label, status="PENDING_PROBE" if ds == "medqa" else "MISSING")); continue
            out.append(dict(base, contrast=label, status="OBSERVED (scored before prereg)" if ds == "tab" else "OK",
                            n_entries=r["n_sa"], L_A=r["L_A"], L_B=r["L_B"], est_sa=r["est_sa"], lower99=r["lower99_sa"], upper99=r["upper99_sa"],
                            lower995=r["lower995_sa"], upper995=r["upper995_sa"], est_bb=r["est_bb"], bb_match_A=r["bb_match_A"], bb_match_B=r["bb_match_B"],
                            utility_A=r["utility_A"], utility_B=r["utility_B"], dU_B_minus_A=r["dU_B_minus_A"], dU_ci99_lo=r["dU_ci99_lo"], dU_ci99_hi=r["dU_ci99_hi"],
                            artifact_path=r["artifact_path"]))
    cols = ["corpus", "contrast", "status", "family", "sacc_version", "n_entries", "L_A", "L_B", "est_sa", "lower99", "upper99", "lower995", "upper995",
            "est_bb", "bb_match_A", "bb_match_B", "utility_A", "utility_B", "dU_B_minus_A", "dU_ci99_lo", "dU_ci99_hi",
            "global_edits", "sacc_edits", "global_edits_outside_sacc_scope", "equal_budget", "docs_identical", "docs", "artifact_path"]
    with open("runs/dev/phase8/scope_control_results.csv.tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); [w.writerow({k: r.get(k) for k in cols}) for r in out]
    os.replace("runs/dev/phase8/scope_control_results.csv.tmp", "runs/dev/phase8/scope_control_results.csv")
    man = {"prereg": "runs/dev/preregs/scope_control_prereg.md", "family": "llama-3.1-8b-instruct (priors_llama.json, N_PERM 8)",
           "executor": "edits.apply_bundle (SACC-Core v1, sha256_16 " + sha("src/relaware/method/edits.py") + ")",
           "lexicon": "src/relaware/profiles.py sha256_16 " + sha("src/relaware/profiles.py"), "arms": {}}
    for ds in ("tab", "medqa", "synthpai"):
        d = {}
        for arm in (f"{ds}_{'s1' if ds == 'medqa' else 'lf'}_only", f"{ds}_sacc_{'s1' if ds == 'medqa' else 'lf'}", f"{ds}_same_cue_mask", f"{ds}_global_lex_l1"):
            p = f"runs/dev/arms/{arm}/anonymized.jsonl"
            d[arm] = {"path": p, "sha256_16": sha(p) if os.path.exists(p) else "MISSING"}
        man["arms"][ds] = d
    man["scoring"] = {"tab": "runs/dev/phase7/e3_control_arms_tab_llama.json (one pass, 4 arms, 2026-09-08)",
                      "medqa": "PENDING -- runs/dev/phase8/scope_control_medqa_llama.json (see COMMANDS.md)",
                      "synthpai": "not scored: structural identity check"}
    json.dump(man, open("runs/dev/phase8/scope_control_manifest.json", "w"), indent=1)
    for r in out: print({k: v for k, v in r.items() if k in ("corpus", "contrast", "status", "est_sa", "lower99", "upper99", "dU_B_minus_A", "dU_ci99_lo", "dU_ci99_hi", "global_edits", "sacc_edits", "docs_identical")})
    print("-> runs/dev/phase8/scope_control_results.csv, scope_control_manifest.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds"); ap.add_argument("--results", action="store_true"); a = ap.parse_args()
    if a.ds: emit(a.ds, write=(a.ds != "tab"))
    if a.results: results()
