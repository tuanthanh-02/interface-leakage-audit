#!/usr/bin/env python
"""Reviewer W7: add SACC's rewrite step to SACC's selection on other upstream anonymisers' outputs. CPU only, 0 USD.

Prereg: runs/dev/preregs/sacc_review_fixedref_upstream_prereg.md, Part U.

The rewrite step is the light setting's algorithm (dev_p29n_emit.py, which produced TAB's released SACC): a character diff
of the base text against the masked text, each redacted region rewritten through the table nu when the region is exactly
one table entry, then agreement repair -- with both defect fixes of preregs/sacc_rewrite_defect_fix_prereg.md (Addendum 1):
possessive his/her before a placeholder, a digit or an opening quote; no table output that names a relationship status.

Gate U-0: on the detector base and SACC-mask, this code must reproduce the fixed TAB SACC
(runs/dev/sacc_fix/arms/tab_SACC_Nfix2.jsonl) byte for byte; otherwise nothing is written.

    $SPIA_PY scripts/dev_sacc_stack_rewrite_emit.py
Writes runs/dev/sacc_review/arms/tab_STK_{INTACT,ADAPTIVE}_SACCN.jsonl and stack_rewrite.json.
"""
import difflib, hashlib, importlib.util as il, json, os, re, sys

SP = __import__("os").environ["SACC_ROOT"]
os.chdir(SP)
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(SP, "scripts"))
_s = il.spec_from_file_location("p24", "scripts/dev_p24_d0.py")
P24 = il.module_from_spec(_s); _s.loader.exec_module(P24)

OUT = "runs/dev/sacc_review"
assert OUT.startswith("runs/dev/")
PREREG = "runs/dev/preregs/sacc_review_fixedref_upstream_prereg.md"
MASK = "[REDACTED]"
NOT_ADMISSIBLE = {"spouse", "surviving spouse", "partner"}     # name a relationship status
POSS_NEXT = re.compile(r"\s+[\[0-9“\"']")
BASES = {"INTACT": "runs/tab_intact_llama-3.1-8b-instruct_longformerspans/anonymized.jsonl",
         "ADAPTIVE": "runs/tab_adaptive_qwen3-30b-a3b-instruct-2507/anonymized.jsonl"}
STK = "runs/dev/main_upgrade/p5/arms/tab_STK_{b}_SACC/anonymized.jsonl"
sha16f = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def load(p):
    return {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"] for l in open(p)}


def neutral(word, text, end):
    w = word.lower()
    if w not in P24.NEUTRAL_OF:
        return None
    if w in ("her", "his"):
        poss = P24.her_is_possessive(text, end) or bool(POSS_NEXT.match(text[end:]))
        rep = "their" if poss else ("them" if w == "her" else "theirs")
        return P24.case_like(word, rep)
    if P24.NEUTRAL_OF[w] in NOT_ADMISSIBLE:
        return None
    return P24.case_like(word, P24.NEUTRAL_OF[w])


def realise(b, t):
    """dev_p29n_emit.py's loop with the fixed neutral() in place of P24.neutral_surface"""
    out, pos = [], 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, t).get_opcodes():
        out.append(t[pos:j1])
        pos = j1
        seg = t[j1:j2]
        if tag != "equal" and MASK in seg:
            orig = b[i1:i2]
            rep = neutral(orig.strip(), b, i2) if orig.strip() else None
            if rep is not None:
                lead = orig[:len(orig) - len(orig.lstrip())]
                trail = orig[len(orig.rstrip()):]
                out.append(seg.replace(MASK, lead + rep + trail, 1) if seg != MASK else lead + rep + trail)
                pos = j2
                continue
        out.append(seg)
        pos = j2
    out.append(t[pos:])
    new, _n = P24.agree("".join(out))
    return new


def main():
    if "U-0" not in open(PREREG).read():
        sys.exit("register Part U first")
    import dev_hitset_r1 as H                                   # the base dev_p29n_emit.py used
    H.COHORT["tab"] = 10 ** 6
    items, _n = H.build("tab", require_sex=False, require_pool=False)
    base = {it["doc_id"]: it["text"] for it in items}
    c1 = load("runs/dev/sacc_improvement/outputs/arms/tab_C1/anonymized.jsonl")
    ref = load("runs/dev/sacc_fix/arms/tab_SACC_Nfix2.jsonl")
    diff = [d for d, t in c1.items() if (realise(base[d], t) if d in base else t) != ref[d]]
    gate = {"documents": len(c1), "mismatches": len(diff), "examples": diff[:5]}
    print("U-0:", "HELD" if not diff else f"FAILED on {len(diff)} documents {diff[:5]}")
    if diff:
        json.dump({"U-0": gate}, open(f"{OUT}/stack_rewrite_gate_failed.json", "w"), indent=1)
        return 2
    os.makedirs(f"{OUT}/arms", exist_ok=True)
    log = {"prereg": PREREG, "U-0": "HELD", "gate": gate, "bases": {}}
    for b, bp in BASES.items():
        bt, mt = load(bp), load(STK.format(b=b))
        recs, changed = [], 0
        for d, t in mt.items():
            new = realise(bt[d], t) if d in bt else t
            changed += new != t
            recs.append({"metadata": {"data_id": d, "arm": f"STK_{b}_SACCN"}, "anonymized_text": new})
        p = f"{OUT}/arms/tab_STK_{b}_SACCN.jsonl"
        with open(p + ".tmp", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        log["bases"][b] = {"base": bp, "base_sha16": sha16f(bp), "masked": STK.format(b=b),
                           "masked_sha16": sha16f(STK.format(b=b)), "output": p, "output_sha16": sha16f(p),
                           "documents": len(recs), "documents_rewritten": changed}
        print(f"STK_{b}_SACCN: {len(recs)} documents, {changed} differ from the masked arm -> {p}")
    json.dump(log, open(f"{OUT}/stack_rewrite.json.tmp", "w"), indent=1)
    os.replace(f"{OUT}/stack_rewrite.json.tmp", f"{OUT}/stack_rewrite.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
