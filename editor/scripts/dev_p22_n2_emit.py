#!/usr/bin/env python3
"""SACC-N2 — the revised neutral realizer. CPU only, deterministic, no model call.

Prereg: runs/dev/preregs/phase22_n2_prereg.md. Three changes over SACC_N, each tied to a measured defect:
  D2  every [REDACTED] in the C1 text is aligned to its own base span, not one per diff block;
  D1  agreement is repaired over the whole clause with a dependency parse, not only next to the pronoun;
  D3  after writing, the document is re-parsed and any sentence that still carries `they + VBZ` falls back to the mask.

    $SPIA_PY scripts/dev_p22_n2_emit.py [--ds medqa,tab,synthpai]
Writes runs/dev/sacc_phase22/arms/<ds>_SACC_N2.jsonl and n2_emit_run.json
"""
import argparse
import collections
import difflib
import hashlib
import importlib.util as il
import json
import os
import re
import sys

SP = __import__("os").environ["SACC_ROOT"]
os.chdir(SP)
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(SP, "scripts"))
import dev_hitset_r1 as H                                           # noqa: E402
import dev_phase10_repair as R                                      # noqa: E402
_s = il.spec_from_file_location("p24", "scripts/dev_p24_d0.py")
P24 = il.module_from_spec(_s); _s.loader.exec_module(P24)

OUT = "runs/dev/sacc_phase22"
PREREG = "runs/dev/preregs/phase22_n2_prereg.md"
C1SRC = "runs/dev/sacc_improvement/outputs/arms/{ds}_C1/anonymized.jsonl"
MASK = "[REDACTED]"
PRON = {"they", "them", "their", "theirs", "themselves"}
# The gate must use the SAME criterion as the fixer: a VBZ whose subject is an inserted pronoun. A regex over
# "they ... <word ending in s>" also matches plural nouns ("they have several cases") and, used as a gate, threw
# away 403 correct TAB sites on false positives in the first trial.
sha16f = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def texts(p):
    return {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"] for l in open(p)}


def holes(base, c1):
    """(mask_start_in_c1, base_span) for EVERY [REDACTED] in the C1 text — D2"""
    ops = difflib.SequenceMatcher(None, base, c1, autojunk=False).get_opcodes()
    out = []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        seg = c1[j1:j2]
        k = seg.find(MASK)
        if k < 0:
            continue
        n = seg.count(MASK)
        if n == 1:
            out.append((j1 + k, (i1, i2)))
            continue
        # several masks inside one block: split the base span on the block's own word boundaries
        words = [m for m in re.finditer(r"\S+", base[i1:i2])]
        pos, taken = 0, 0
        for m in re.finditer(re.escape(MASK), seg):
            w = words[taken] if taken < len(words) else None
            out.append((j1 + m.start(), (i1 + w.start(), i1 + w.end()) if w else (i1, i2)))
            taken += 1
            pos = m.end()
    return out


AUXFIX = {"has": "have", "is": "are", "was": "were", "does": "do", "isn't": "aren't", "wasn't": "weren't",
          "hasn't": "haven't", "doesn't": "don't"}


def subject_of(t):
    """the subject governing t, following aux/auxpass up to the verb that carries it"""
    head = t.head if t.dep_ in ("aux", "auxpass") else t
    return [c for c in head.children if c.dep_ in ("nsubj", "nsubjpass")]


def bad_pairs(doc):
    """every singular-agreeing verb form whose subject is an inserted they/them/their.

    The first version only looked at tokens tagged VBZ that carried the subject themselves. That misses exactly the
    auxiliaries — `they has had`, `They is admitted`, `they was born` — because there the subject hangs off the main
    verb and `was` is tagged VBD. The gate shared that blind spot and passed 1,149 real errors on MedQA."""
    out = []
    for t in doc:
        w = t.text.lower()
        if not (t.tag_ == "VBZ" or w in AUXFIX):
            continue
        if t.pos_ not in ("VERB", "AUX"):
            continue
        if any(c.text.lower() in PRON for c in subject_of(t)):
            out.append(t)
    return out


def surface_bad(doc):
    """an INDEPENDENT check: `they` followed within three tokens by a singular-agreeing verb form, judged by the tag,
    not by the dependency link. It exists so the gate cannot inherit the fixer's blind spot."""
    out = []
    for i, t in enumerate(doc):
        if t.text.lower() != "they":
            continue
        for u in doc[i + 1:i + 5]:
            if u.pos_ in ("VERB", "AUX") and (u.tag_ == "VBZ" or u.text.lower() in AUXFIX):
                out.append(u)
                break
            if u.pos_ in ("VERB", "AUX"):
                break
    return out


def fix_agreement(nlp, text, log=None):
    """every verb whose subject is an inserted they/them/their becomes its lemma — D1"""
    doc = nlp(text)
    edits, n = [], 0
    for t in doc:
        w = t.text.lower()
        if not ((t.tag_ == "VBZ" or w in AUXFIX) and t.pos_ in ("VERB", "AUX")):
            continue
        ok = any(c.text.lower() in PRON for c in subject_of(t))
        if not ok and t.dep_ == "conj":                      # "they reports X and denies Y"
            ok = any(c.text.lower() in PRON for c in subject_of(t.head))
        if not ok:
            continue
        lem = AUXFIX.get(t.text.lower(), t.lemma_.lower())
        if lem == t.text.lower():
            continue
        rep = lem.capitalize() if t.text[:1].isupper() else lem
        edits.append((t.idx, t.idx + len(t.text), rep))
        n += 1
    for a, b, rep in sorted(edits, key=lambda e: -e[0]):
        if log is not None:
            log.append([a, text[a:b], rep])
        text = text[:a] + rep + text[b:]
    return text, n


def main(only):
    if "PN2-1" not in open(PREREG).read():
        sys.exit("the SACC-N2 prereg must be registered first")
    nlp = R.nlp()
    os.makedirs(f"{OUT}/arms", exist_ok=True)
    run = {"prereg": PREREG, "corpora": {}}
    for ds in only:
        c1 = texts(C1SRC.format(ds=ds))
        H.COHORT[ds] = 10 ** 6
        items, _n = H.build(ds, require_sex=False, require_pool=False)
        base = {it["doc_id"]: it["text"] for it in items}
        st = collections.Counter()
        recs, plan = [], []
        for did, t in c1.items():
            b = base.get(did)
            if b is None:
                recs.append({"metadata": {"data_id": did}, "anonymized_text": t}); st["no_base"] += 1; continue
            hs = holes(b, t)
            st["holes"] += len(hs)
            edits = []
            for mpos, (i1, i2) in hs:
                orig = b[i1:i2].strip()
                rep = P24.neutral_surface(orig, b, i2) if orig else None
                if rep is None:
                    st["kept_mask"] += 1
                    continue
                edits.append((mpos, mpos + len(MASK), rep, mpos))
                st["rewritten"] += 1
            new = t
            for a, z, rep, _k in sorted(edits, key=lambda e: -e[0]):
                new = new[:a] + rep + new[z:]
            alog = []
            new, nfix = fix_agreement(nlp, new, alog)
            st["agreement_fixes"] += nfix
            # D3: the gate — a sentence that still carries they + VBZ gives its sites back to the mask
            doc = nlp(new)
            bad = bad_pairs(doc) + surface_bad(doc)
            if bad:
                spans = [(s.start_char, s.end_char) for s in doc.sents
                         if any(s.start_char <= t.idx < s.end_char for t in bad)]
                keep = [e for e in edits if not any(a <= e[0] < z for a, z in spans)]
                if len(keep) != len(edits):
                    st["gate_docs"] += 1
                    st["gate_sites"] += len(edits) - len(keep)
                    st["rewritten"] -= len(edits) - len(keep)
                    st["kept_mask"] += len(edits) - len(keep)
                    new = t
                    for a, z, rep, _k in sorted(keep, key=lambda e: -e[0]):
                        new = new[:a] + rep + new[z:]
                    alog = []
                    new, n2 = fix_agreement(nlp, new, alog)
                    st["agreement_fixes"] += n2
                    edits = keep
            _d = nlp(new)
            st["residual_bad"] += len(set(bad_pairs(_d)) | set(surface_bad(_d)))
            st["docs_changed"] += new != t
            recs.append({"metadata": {"data_id": did}, "anonymized_text": new})
            plan.append({"document_id": did, "holes": [[e[0], e[2]] for e in edits], "agreement": alog})
        p = f"{OUT}/arms/{ds}_SACC_N2.jsonl"
        with open(p + ".tmp", "w") as fh:
            for x in recs:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        pp = f"{OUT}/n2_edit_plan_{ds}.jsonl"
        with open(pp + ".tmp", "w") as fh:
            for x in plan:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        os.replace(pp + ".tmp", pp)
        n2 = texts(p)
        cell = {"documents": len(recs), "path": p, "sha16": sha16f(p), **dict(st),
                "coverage": round(st["rewritten"] / max(1, st["holes"]), 4),
                "masks_C1": sum(t.count(MASK) for t in c1.values()),
                "masks_N2": sum(t.count(MASK) for t in n2.values()),
                "masks_never_above_C1": int(all(n2[d].count(MASK) <= c1[d].count(MASK) for d in c1))}
        run["corpora"][ds] = cell
        print(f"{ds}: holes {st['holes']} | rewritten {st['rewritten']} (coverage {cell['coverage']}) | "
              f"kept mask {st['kept_mask']} | agreement fixes {st['agreement_fixes']} | "
              f"gate {st['gate_docs']} documents/{st['gate_sites']} sites | residual errors {st['residual_bad']}")
    json.dump(run, open(f"{OUT}/n2_emit_run.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="medqa,tab,synthpai")
    a = ap.parse_args()
    sys.exit(main(tuple(a.ds.split(","))))
