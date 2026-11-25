#!/usr/bin/env python3
"""Phase 24 D0 + emit — surface sites, the four arms and their mechanical checks. CPU only, no model call.

Prereg: runs/dev/preregs/phase24_sacc_d_prereg.md. Sites are the frozen D1 SEX cues attributed to the target subject,
projected to x_base offsets exactly as dev_hitset_r1.build projects them, restricted to surfaces in the closed table.

    $SPIA_PY scripts/dev_p24_d0.py --stage d0      cohort, split, sites, contract, request plan
    $SPIA_PY scripts/dev_p24_d0.py --stage emit    the four arms + K-2 check
Writes runs/dev/sacc_phase24/{cohort.csv, sites.jsonl, contract.yaml, request_plan.json, arms/*.jsonl, emit_run.json}.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import sys

import yaml

SP = __import__("os").environ["SACC_ROOT"]
os.chdir(SP)
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(SP, "scripts"))
from src.relaware import registry                                  # noqa: E402
import dev_phase10_emit_m1 as M1E                                  # noqa: E402
import dev_hitset_r1 as H                                          # noqa: E402

OUT = "runs/dev/sacc_phase24"
assert OUT.startswith("runs/dev/")
PREREG = "runs/dev/preregs/phase24_sacc_d_prereg.md"
P22 = "runs/dev/sacc_phase22"
DOCSET = "cohort"
MASK = "[REDACTED]"
STRONG = 0.75
SEED = 0
sha16f = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
h16 = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]

# closed, corpus-independent tables. M = male-marked, F = female-marked; NEUTRAL is the fluent neutral form.
PAIRS = [
    ("he", "she"), ("him", "her"), ("his", "her"), ("himself", "herself"),
    ("mr", "ms"), ("mr.", "ms."), ("sir", "madam"), ("gentleman", "lady"), ("gentlemen", "ladies"),
    ("man", "woman"), ("men", "women"), ("boy", "girl"), ("boys", "girls"), ("male", "female"), ("males", "females"),
    ("father", "mother"), ("dad", "mom"), ("son", "daughter"), ("sons", "daughters"),
    ("brother", "sister"), ("brothers", "sisters"), ("husband", "wife"), ("uncle", "aunt"),
    ("nephew", "niece"), ("grandfather", "grandmother"), ("grandson", "granddaughter"),
    ("widower", "widow"), ("king", "queen"), ("prince", "princess"), ("waiter", "waitress"),
    ("actor", "actress"), ("businessman", "businesswoman"), ("chairman", "chairwoman"), ("spokesman", "spokeswoman"),
    ("boyfriend", "girlfriend"), ("fiance", "fiancee"), ("godfather", "godmother"), ("stepfather", "stepmother"),
    ("stepson", "stepdaughter"), ("lad", "lass"), ("guy", "gal"), ("paternal", "maternal"),
]
NEUTRAL_OF = {
    "he": "they", "she": "they", "him": "them", "her": "them", "his": "their", "himself": "themselves",
    "herself": "themselves", "mr": "the person", "ms": "the person", "mr.": "the person", "ms.": "the person",
    "mrs": "the person", "mrs.": "the person", "sir": "the person", "madam": "the person",
    "gentleman": "person", "lady": "person", "gentlemen": "people", "ladies": "people",
    "man": "person", "woman": "person", "men": "people", "women": "people", "boy": "child", "girl": "child",
    "boys": "children", "girls": "children", "male": "person", "female": "person", "males": "people", "females": "people",
    "father": "parent", "mother": "parent", "dad": "parent", "mom": "parent", "son": "child", "daughter": "child",
    "sons": "children", "daughters": "children", "brother": "sibling", "sister": "sibling",
    "brothers": "siblings", "sisters": "siblings", "husband": "spouse", "wife": "spouse",
    "uncle": "relative", "aunt": "relative", "nephew": "relative", "niece": "relative",
    "grandfather": "grandparent", "grandmother": "grandparent", "grandson": "grandchild",
    "granddaughter": "grandchild", "widower": "surviving spouse", "widow": "surviving spouse",
    "king": "monarch", "queen": "monarch", "prince": "royal", "princess": "royal",
    "waiter": "server", "waitress": "server", "actor": "performer", "actress": "performer",
    "businessman": "businessperson", "businesswoman": "businessperson", "chairman": "chair", "chairwoman": "chair",
    "spokesman": "spokesperson", "spokeswoman": "spokesperson", "boyfriend": "partner", "girlfriend": "partner",
    "fiance": "partner", "fiancee": "partner", "godfather": "godparent", "godmother": "godparent",
    "stepfather": "stepparent", "stepmother": "stepparent", "stepson": "stepchild", "stepdaughter": "stepchild",
    "lad": "youth", "lass": "youth", "guy": "person", "gal": "person", "paternal": "parental", "maternal": "parental",
}
FLIP_OF, MARK = {}, {}
for a, b in PAIRS:
    FLIP_OF[a], FLIP_OF[b] = b, a
    MARK[a], MARK[b] = "M", "F"
FLIP_OF["mrs"], FLIP_OF["mrs."] = "mr", "mr."
MARK["mrs"], MARK["mrs."] = "F", "F"
TOKEN = re.compile(r"(?<![\w\[])[A-Za-z]+\.?(?![\w\]])")
PRON_TITLE = {"he", "him", "his", "himself", "she", "her", "hers", "herself",
              "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "sir", "madam"}


def case_like(src, rep):
    if src.isupper() and len(src) > 1:
        return rep.upper()
    if src[:1].isupper():
        return rep[:1].upper() + rep[1:]
    return rep


def her_is_possessive(text, end):
    """deterministic rule: "her" before a word character run that is not a verb-ish suffix -> possessive ("his"), else "him" """
    m = re.match(r"\s+([A-Za-z]+)", text[end:])
    return bool(m and m.group(1).lower() not in ("and", "or", "to", "in", "on", "at", "with", "for", "from", "of", "by",
                                                 "as", "that", "who", "she", "he", "it", "was", "is", "were", "are",
                                                 "had", "has", "have", "the", "a", "an", "after", "before", "about"))


def flip_surface(word, text, end):
    w = word.lower()
    if w not in FLIP_OF:
        return None
    if w == "her":
        return case_like(word, "his" if her_is_possessive(text, end) else "him")
    return case_like(word, FLIP_OF[w])


def neutral_surface(word, text, end):
    w = word.lower()
    if w not in NEUTRAL_OF:
        return None
    if w == "her":
        return case_like(word, "their" if her_is_possessive(text, end) else "them")
    if w == "his":
        return case_like(word, "their" if her_is_possessive(text, end) else "theirs")
    return case_like(word, NEUTRAL_OF[w])


def build_sites():
    """per cohort document: base text, the target subject's mappable SEX surface sites (x_base offsets)"""
    srcfile = f"{P22}/cohort.csv" if DOCSET == "cohort" else f"{P22}/dev24.csv"
    coh = {(r["corpus"], r["document_id"]): r for r in csv.DictReader(open(srcfile))}
    rows = []
    for ds in ("medqa", "tab"):
        tree, corp, *_ = M1E.CFG[ds]
        reg = registry.load(f"{tree}/registry.json")
        det = json.load(open(f"{tree}/detections_d1.json"))
        cues_of = {tuple(k.split("||")): v for k, v in det["cues"].items()}
        _c, base_spans = __import__("dev_saccv2_s0_census").make_base(ds, M1E.CFG)
        subs = collections.defaultdict(list)
        for s in reg["subjects"]:
            subs[s["doc_id"]].append(s)
        ent = {}
        for e in reg["entries"]:
            if e["attribute"] == "SEX" and str(e.get("gold_value", "")).lower() in ("male", "female"):
                ent.setdefault(e["doc_id"], e)
        for line in open(corp):
            rec = json.loads(line)
            did, src = rec["metadata"]["data_id"], rec["text"]
            if (ds, did) not in coh or did not in ent:
                continue
            bsp = base_spans(did, src)
            base_txt = M1E.v1_sweep(src, bsp, [])
            m = M1E.offset_map(src, base_txt)
            e = ent[did]
            sid = e["eval_subject_id"]
            other_ids = [s["eval_subject_id"] for s in subs[did] if s["eval_subject_id"] != sid]
            sites, unmappable, other_sites = [], [], 0
            s2_added = s2_minority = 0
            for who, tgt in [(sid, True)] + [(o, False) for o in other_ids]:
                for c in cues_of.get((who, "SEX"), []):
                    s0, e0 = c["start"], c["end"]
                    if not (s0 in m and (e0 - 1) in m and m[e0 - 1] + 1 - m[s0] == e0 - s0):
                        continue
                    a, z = m[s0], m[e0 - 1] + 1
                    surf = base_txt[a:z]
                    mappable = surf.lower() in FLIP_OF and surf.lower() in NEUTRAL_OF
                    if not tgt:
                        other_sites += mappable
                        continue
                    (sites if mappable else unmappable).append({"start": a, "end": z, "surface": surf,
                                                               "mark": MARK.get(surf.lower())})
            # S2 (amendment): in a single-subject document, closed-table pronouns/titles of the majority mark
            s2_added, s2_minority = 0, 0
            if len(subs[did]) == 1:
                toks = [(m.start(), m.end(), m.group(0)) for m in TOKEN.finditer(base_txt)
                        if m.group(0).lower().rstrip(".") in PRON_TITLE or m.group(0).lower() in PRON_TITLE]
                mk = collections.Counter(MARK.get(t[2].lower().rstrip("."), MARK.get(t[2].lower())) for t in toks)
                major = "M" if mk.get("M", 0) >= mk.get("F", 0) else "F"
                have = {(s["start"], s["end"]) for s in sites}
                for a, z, w in toks:
                    mark = MARK.get(w.lower().rstrip("."), MARK.get(w.lower()))
                    if mark != major:
                        s2_minority += 1
                        continue
                    if (a, z) in have or any(x["start"] < z and a < x["end"] for x in sites):
                        continue
                    sites.append({"start": a, "end": z, "surface": w, "mark": mark, "rule": "S2"})
                    s2_added += 1
            for s in sites:
                s.setdefault("rule", "S1")
            sites = sorted({(s["start"], s["end"]): s for s in sites}.values(), key=lambda s: s["start"])
            marks = collections.Counter(s["mark"] for s in sites)
            rows.append({"corpus": ds, "document_id": did, "gold": e["gold_value"], "n_subjects": len(subs[did]),
                         "n_sites_S1": sum(1 for s in sites if s["rule"] == "S1"),
                         "n_sites_S2": sum(1 for s in sites if s["rule"] == "S2"), "s2_minority_left": s2_minority,
                         "p0_A": float(coh[(ds, did)]["p0_A"]) if "p0_A" in coh[(ds, did)] else None,
                         "chars": len(base_txt),
                         "sites": sites, "n_sites": len(sites), "n_unmappable": len(unmappable),
                         "unmappable": [u["surface"] for u in unmappable][:20],
                         "other_subject_mappable_sites": other_sites,
                         "surface_value": ("male" if marks.get("M", 0) >= marks.get("F", 0) else "female") if sites else None,
                         "surface_mixed": int(bool(marks.get("M", 0) and marks.get("F", 0))),
                         "base_sha16": h16(base_txt)})
    return rows


def stage_d0():
    if "P24-1" not in open(PREREG).read():
        sys.exit("the Phase 24 prereg must be registered first")
    os.makedirs(OUT, exist_ok=True)
    rows = build_sites()
    with open(f"{OUT}/sites.jsonl.tmp", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f"{OUT}/sites.jsonl.tmp", f"{OUT}/sites.jsonl")
    fields = ["corpus", "document_id", "gold", "n_subjects", "p0_A", "chars", "n_sites", "n_sites_S1",
              "n_sites_S2", "s2_minority_left", "n_unmappable",
              "other_subject_mappable_sites", "surface_value", "surface_mixed", "base_sha16"]
    with open(f"{OUT}/cohort.csv.tmp", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields + ["group"])
        w.writeheader()
        for r in rows:
            w.writerow({**{k: r[k] for k in fields}, "group": ("UNKNOWN" if r["p0_A"] is None else
                       ("CONTENT_STRONG" if r["p0_A"] >= STRONG else "CONTENT_WEAK"))})
    os.replace(f"{OUT}/cohort.csv.tmp", f"{OUT}/cohort.csv")
    per = {}
    for ds in ("medqa", "tab"):
        v = [r for r in rows if r["corpus"] == ds]
        st = [r for r in v if r["p0_A"] is not None and r["p0_A"] >= STRONG]
        per[ds] = {"documents": len(v), "content_strong": len(st),
                   "docs_with_sites": sum(1 for r in v if r["n_sites"]),
                   "median_sites": sorted(r["n_sites"] for r in v)[len(v) // 2] if v else 0,
                   "docs_with_unmappable": sum(1 for r in v if r["n_unmappable"]),
                   "docs_touching_other_subject": sum(1 for r in v if r["other_subject_mappable_sites"]),
                   "surface_value_equals_gold": sum(1 for r in v if r["surface_value"] == r["gold"]),
                   "surface_mixed": sum(r["surface_mixed"] for r in v),
                   "sites_by_rule": {"S1": sum(r["n_sites_S1"] for r in v), "S2": sum(r["n_sites_S2"] for r in v)},
                   "s2_minority_tokens_left": sum(r["s2_minority_left"] for r in v),
                   "single_subject_documents": sum(1 for r in v if r["n_subjects"] == 1)}
    n_arms_probed = 4                                   # HOLE, NEUTRAL, CONFLICT, RANDOM (B0 is cached from R1b/Phase 23)
    docs = sum(len([r for r in rows if r["corpus"] == ds]) for ds in ("medqa", "tab"))
    chars = sum(r["chars"] for r in rows)
    ptok = (chars / 3.7 + 120 * docs) * 2 * n_arms_probed
    plan = {"documents": docs, "arms_probed": n_arms_probed, "probe_calls": 2 * n_arms_probed * docs,
            "model": "gpt-4o-mini-2024-07-18", "orders": 2,
            "est_usd": round(ptok * 0.15 / 1e6 + 2 * n_arms_probed * docs * 0.60 / 1e6, 4),
            "b0_reuse": "p_U on x_base is already in the shared probe cache from Phase 23 d1b",
            "gpu": "2 E2-CF-FULL passes (Llama :8032, Qwen :8031) run by the author, only after the internal gate"}
    json.dump(plan, open(f"{OUT}/request_plan.json", "w"), indent=1)
    contract = {"prereg": PREREG, "plan_sha16": "89895c31f10fddb8", "technique_note_sha16": "c9eb7e51c6adba97",
                "content_strong_threshold": STRONG, "delta": 0.05, "content_decisive": 0.25, "seed": SEED,
                "arms": ["B0", "HOLE", "NEUTRAL", "CONFLICT", "RANDOM"],
                "site_rule": "frozen D1 SEX cues of the target subject, projected to x_base, surface in the closed table",
                "tables": {"pairs": len(PAIRS), "neutral_entries": len(NEUTRAL_OF)},
                "per_corpus": per, "request_plan": plan,
                "inputs_sha16": {p: sha16f(p) for p in (f"{P22}/cohort.csv", "runs/relaware_medqa/detections_d1.json",
                                                        "runs/relaware/detections_d1.json")},
                "code_sha16": sha16f("scripts/dev_p24_d0.py")}
    with open(f"{OUT}/contract.yaml.tmp", "w") as fh:
        yaml.safe_dump(contract, fh, sort_keys=False, allow_unicode=True, width=140)
    os.replace(f"{OUT}/contract.yaml.tmp", f"{OUT}/contract.yaml")
    print(json.dumps({"per_corpus": per, "request_plan": plan}, indent=1))
    return 0


AGREE = [(r"(?<![\w])([Tt]hey)(\s+)is(?![\w])", r"\1\2are"), (r"(?<![\w])([Tt]hey)(\s+)was(?![\w])", r"\1\2were"),
         (r"(?<![\w])([Tt]hey)(\s+)has(?![\w])", r"\1\2have"), (r"(?<![\w])([Tt]hey)(\s+)does(?![\w])", r"\1\2do"),
         (r"(?<![\w])([Tt]hey)(\s+)(?:'s|\u2019s)(?![\w])", r"\1\2are")]


def agree(text):
    """subject-verb agreement after a he/she -> they rewrite; deterministic, only next to the inserted pronoun"""
    n = 0
    for pat, rep in AGREE:
        text, k = re.subn(pat, rep, text)
        n += k
    return text, n


def apply_spans(text, edits):
    out = text
    for a, z, rep in sorted(edits, key=lambda x: -x[0]):
        out = out[:a] + rep + out[z:]
    return out


def stage_emit():
    rows = [json.loads(l) for l in open(f"{OUT}/sites.jsonl")]
    texts = {}
    for ds in ("medqa", "tab"):
        tree, corp, *_ = M1E.CFG[ds]
        _c, base_spans = __import__("dev_saccv2_s0_census").make_base(ds, M1E.CFG)
        for line in open(corp):
            rec = json.loads(line)
            did, src = rec["metadata"]["data_id"], rec["text"]
            if any(r["corpus"] == ds and r["document_id"] == did for r in rows):
                texts[(ds, did)] = M1E.v1_sweep(src, base_spans(did, src), [])
    os.makedirs(f"{OUT}/arms", exist_ok=True)
    arms = {a: [] for a in ("B0", "HOLE", "NEUTRAL", "CONFLICT", "RANDOM")}
    stats = collections.Counter()
    checks = []
    for r in rows:
        k = (r["corpus"], r["document_id"])
        base = texts[k]
        assert h16(base) == r["base_sha16"], f"{k} base text changed"
        sites = r["sites"]
        hole = [(s["start"], s["end"], MASK) for s in sites]
        neu, con = [], []
        for s in sites:
            n = neutral_surface(s["surface"], base, s["end"])
            f = flip_surface(s["surface"], base, s["end"])
            stats["neutral_missing"] += n is None
            stats["flip_missing"] += f is None
            neu.append((s["start"], s["end"], n if n is not None else MASK))
            con.append((s["start"], s["end"], f if f is not None else MASK))
        # RANDOM: equally many content words elsewhere, masked; never overlapping a site
        taken = [(s["start"], s["end"]) for s in sites]
        cands = [(m.start(), m.end()) for m in re.finditer(r"(?<![\w\[])[A-Za-z]{4,}(?![\w\]])", base)
                 if not any(a < m.end() and m.start() < z for a, z in taken)]
        rng = random.Random(f"p24|{r['corpus']}|{r['document_id']}|{SEED}")
        pick = sorted(rng.sample(cands, min(len(sites), len(cands))))
        rnd = [(a, z, MASK) for a, z in pick]
        neutral_text, n_agree = agree(apply_spans(base, neu))
        stats["agreement_fixes"] += n_agree
        made = {"B0": base, "HOLE": apply_spans(base, hole), "NEUTRAL": neutral_text,
                "CONFLICT": apply_spans(base, con), "RANDOM": apply_spans(base, rnd)}
        for a, t in made.items():
            arms[a].append({"metadata": {"data_id": r["document_id"]}, "anonymized_text": t})
        # mechanical check: the text between registered sites must reappear, in order, unchanged
        for a_ in ("HOLE", "NEUTRAL", "CONFLICT", "RANDOM"):
            spans = sorted([(s["start"], s["end"]) for s in sites] if a_ != "RANDOM" else [(x, y) for x, y, _ in rnd])
            gaps, pos = [], 0
            for x, y in spans:
                gaps.append(base[pos:x])
                pos = y
            gaps.append(base[pos:])
            t, cur, bad = made[a_], 0, 0
            for g in gaps:
                if not g:
                    continue
                cands = [g]
                if a_ == "NEUTRAL":          # an agreement fix can straddle the site boundary: They|_is -> They|_are
                    lead = {"is": "are", "was": "were", "has": "have", "does": "do", "'s": "are", "\u2019s": "are"}
                    m_ = re.match(r"(\s*)(is|was|has|does|'s|\u2019s)(?![\w])", g)
                    if m_:
                        cands.append(m_.group(1) + lead[m_.group(2)] + g[m_.end():])
                i = -1
                for cand in cands:
                    i = t.find(cand, cur)
                    if i >= 0:
                        cur = i + len(cand)
                        break
                bad += i < 0
            checks.append({"corpus": r["corpus"], "document_id": r["document_id"], "arm": a_, "out_of_site_changes": bad,
                           "other_subject_sites": r["other_subject_mappable_sites"]})
        stats["docs"] += 1
        stats["sites"] += len(sites)
        stats["random_short"] += len(pick) < len(sites)
    for ds in ("medqa", "tab"):
        ids = {r["document_id"] for r in rows if r["corpus"] == ds}
        for a, v in arms.items():
            q = f"{OUT}/arms/{ds}_{a}.jsonl"
            with open(q + ".tmp", "w") as fh:
                for x in v:
                    if x["metadata"]["data_id"] in ids:
                        fh.write(json.dumps(x, ensure_ascii=False) + "\n")
            os.replace(q + ".tmp", q)
    for a, v in arms.items():
        p = f"{OUT}/arms/{a}.jsonl"
        with open(p + ".tmp", "w") as fh:
            for x in v:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
    k2 = {a: round(sum(1 for c in checks if c["arm"] == a and c["out_of_site_changes"]) /
                   max(1, sum(1 for c in checks if c["arm"] == a)), 4) for a in ("HOLE", "NEUTRAL", "CONFLICT", "RANDOM")}
    touch = sum(1 for r in rows if r["other_subject_mappable_sites"]) / max(1, len(rows))
    run = {"documents": stats["docs"], "sites_total": stats["sites"], "neutral_missing": stats["neutral_missing"],
           "flip_missing": stats["flip_missing"], "random_short_documents": stats["random_short"],
           "agreement_fixes": stats["agreement_fixes"],
           "out_of_site_change_rate_by_arm": k2,
           "documents_with_other_subject_sites_left_untouched": round(touch, 4),
           "arm_files_sha16": {a: sha16f(f"{OUT}/arms/{a}.jsonl") for a in arms},
           "code_sha16": sha16f("scripts/dev_p24_d0.py")}
    json.dump(run, open(f"{OUT}/emit_run.json", "w"), indent=1)
    print(json.dumps(run, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("d0", "emit"))
    ap.add_argument("--set", dest="docset", default="cohort", choices=("cohort", "dev24"))
    a = ap.parse_args()
    DOCSET = a.docset
    if DOCSET == "dev24":
        OUT = OUT + "_facts"
        os.makedirs(OUT, exist_ok=True)
    sys.exit(stage_d0() if a.stage == "d0" else stage_emit())
