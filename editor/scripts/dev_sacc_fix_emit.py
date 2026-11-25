#!/usr/bin/env python
"""Fix SACC's two rewrite-step defects on the released texts. CPU only, 0 USD, deterministic.

Prereg: runs/dev/preregs/sacc_rewrite_defect_fix_prereg.md (F-1 is checked here and stops the run on failure).

D1  'theirs' written for possessive 'his' before a placeholder, a number or an opening quote  ->  'their'
D2  'spouse(s)' / 'surviving spouse' / 'partner(s)' written in place of an input marital term  ->  '[REDACTED]'
    (a table output that names a relationship status is not admissible while RELATIONSHIP is protected)

Every change must align (word-level difflib against the anonymised input) to the stated input word; a change that does
not align is not made and is counted as a mismatch. Text outside the changed spans is copied unchanged.

    $SPIA_PY scripts/dev_sacc_fix_emit.py

Writes runs/dev/sacc_fix/arms/{medqa,tab,synthpai}_SACC_Nfix.jsonl, medqa_SACC_Nfix_light.jsonl and changes.json.
"""
import difflib, hashlib, json, os, re, sys

SP = __import__("os").environ["SACC_ROOT"]
os.chdir(SP)
OUT = "runs/dev/sacc_fix"
assert OUT.startswith("runs/dev/")
PREREG = "runs/dev/preregs/sacc_rewrite_defect_fix_prereg.md"
MASK = "[REDACTED]"
INPUT = {"medqa": "runs/dev/arms/medqa_s1_only/anonymized.jsonl", "tab": "runs/dev/arms/tab_lf_only/anonymized.jsonl",
         "synthpai": "runs/dev/arms/synthpai_lf_only/anonymized.jsonl"}
JOBS = (("medqa", "SACC_Nfix", "runs/dev/sacc_nstar/arms/medqa_SACC_Nstar.jsonl"),
        ("tab", "SACC_Nfix", "runs/dev/sacc_nstar/arms/tab_SACC_Nstar.jsonl"),
        ("synthpai", "SACC_Nfix", "runs/dev/sacc_nstar/arms/synthpai_SACC_Nstar.jsonl"),
        ("medqa", "SACC_Nfix_light", "runs/dev/sacc_neutral/arms/medqa_SACC_N.jsonl"))
EXPECT = {("medqa", "SACC_Nfix"): {"D1": 136}, ("tab", "SACC_Nfix"): {"D1": 7, "D2": 21},
          ("synthpai", "SACC_Nfix"): {"D1": 0, "D2": 0}}      # F-1 recounts on the released arm
MARITAL = {"husband", "wife", "widow", "widower", "boyfriend", "girlfriend", "fiance", "fiancee", "fiancé", "fiancée",
           "husbands", "wives", "widows", "widowers", "boyfriends", "girlfriends"}
D1_RE = re.compile(r"\b(theirs|Theirs|THEIRS)\b(?=\s+[\[0-9“\"'])")
D1B_RE = re.compile(r"\b(them|Them|THEM)\b(?=\s+[\[0-9“\"'])")   # Addendum 1: possessive 'her' written as 'them'
D2_RE = re.compile(r"\b(surviving spouses?|spouses?|partners?)\b", re.I)
TOK = re.compile(r"\S+")
V2 = "--v2" in sys.argv                                            # Addendum 1: v1 rules + D1b, arms SACC_Nfix2*
EXPECT2 = {("medqa", "SACC_Nfix2"): {"D1": 136, "D1b": 125, "D2": 0},
           ("medqa", "SACC_Nfix2_light"): {"D1": 119, "D1b": 118},
           ("tab", "SACC_Nfix2"): {"D1": 7, "D1b": 0, "D2": 21},
           ("synthpai", "SACC_Nfix2"): {"D1": 0, "D1b": 0, "D2": 0}}
sha16f = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def load(p):
    return {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"] for l in open(p)}


def norm(w):
    return re.sub(r"(['’]s)$", "", w.strip(".,;:!?()\"'“”‘’[]").lower())


def aligned_input_words(inp, out, start, end):
    """the input words aligned to the output characters [start, end): word-level diff of input vs output"""
    ti, to = list(TOK.finditer(inp)), list(TOK.finditer(out))
    js = [k for k, m in enumerate(to) if m.start() < end and m.end() > start]
    if not js:
        return None, None
    sm = difflib.SequenceMatcher(None, [m.group() for m in ti], [m.group() for m in to], autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if j1 <= js[0] < j2 or (j1 == j2 == js[0]):
            return tag, [ti[i].group() for i in range(i1, i2)]
    return None, None


def fix(inp, out):
    edits, miss = [], {"D1": 0, "D1b": 0, "D2": 0}
    for m in (D1B_RE.finditer(out) if V2 else ()):
        tag, words = aligned_input_words(inp, out, m.start(), m.end())
        if tag == "replace" and words and any(norm(w) == "her" for w in words):
            rep = {"them": "their", "Them": "Their", "THEM": "THEIR"}[m.group(1)]
            edits.append((m.start(), m.end(), rep, "D1b"))
        else:
            miss["D1b"] += 1                     # an original 'them' before such a token: left alone
    for m in D1_RE.finditer(out):
        tag, words = aligned_input_words(inp, out, m.start(), m.end())
        if tag in ("replace",) and words and any(norm(w) == "his" for w in words):
            rep = {"theirs": "their", "Theirs": "Their", "THEIRS": "THEIR"}[m.group(1)]
            edits.append((m.start(), m.end(), rep, "D1"))
        else:
            miss["D1"] += 1
    for m in D2_RE.finditer(out):
        tag, words = aligned_input_words(inp, out, m.start(), m.end())
        if tag in ("replace", "insert") and words and any(norm(w) in MARITAL for w in words):
            edits.append((m.start(), m.end(), MASK, "D2"))
        elif tag != "equal":
            miss["D2"] += 1                      # a rewrite output that does not align to a marital term
    edits.sort()
    for a, b in zip(edits, edits[1:]):
        assert a[1] <= b[0], f"overlapping edits {a} {b}"
    new, pos = [], 0
    for s, e, rep, _k in edits:
        new.append(out[pos:s]); new.append(rep); pos = e
    new.append(out[pos:])
    return "".join(new), edits, miss


def unchanged_outside(out, new, edits):
    """walk both texts edit by edit: every character outside an edited span must be identical"""
    po, pn = 0, 0
    for s, e, rep, _k in edits:
        if out[po:s] != new[pn:pn + (s - po)]:
            return False
        pn += s - po
        if new[pn:pn + len(rep)] != rep:
            return False
        pn += len(rep); po = e
    return out[po:] == new[pn:]


def main():
    if "F-1" not in open(PREREG).read():
        sys.exit("register the prereg first")
    os.makedirs(f"{OUT}/arms", exist_ok=True)
    log = {"prereg": PREREG, "jobs": {}}
    failed = []
    for ds, arm, src in JOBS:
        inp, out = load(INPUT[ds]), load(src)
        if V2:
            arm = arm.replace("SACC_Nfix", "SACC_Nfix2")
        recs, n = [], {"D1": 0, "D1b": 0, "D2": 0}
        miss = {"D1": 0, "D1b": 0, "D2": 0}
        docs = {"D1": set(), "D1b": set(), "D2": set()}
        examples = []
        for did, t in out.items():
            if did not in inp:
                recs.append({"metadata": {"data_id": did, "arm": arm}, "anonymized_text": t}); continue
            new, edits, ms = fix(inp[did], t)
            for k in miss:
                miss[k] += ms[k]
            for s, e, rep, kind in edits:
                n[kind] += 1; docs[kind].add(did)
                if len(examples) < 6:
                    examples.append({"doc": did, "kind": kind, "was": t[max(0, s - 30):e + 25], "now": rep})
            assert unchanged_outside(t, new, edits), f"{ds}/{did}: text changed outside the edited spans"
            recs.append({"metadata": {"data_id": did, "arm": arm, "source": src}, "anonymized_text": new})
        p = f"{OUT}/arms/{ds}_{arm}.jsonl"
        with open(p + ".tmp", "w") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(p + ".tmp", p)
        cell = {"source": src, "source_sha16": sha16f(src), "output": p, "output_sha16": sha16f(p), "documents": len(recs),
                "D1_changes": n["D1"], "D1_documents": len(docs["D1"]), "D1b_changes": n["D1b"],
                "D1b_documents": len(docs["D1b"]), "D2_changes": n["D2"],
                "D2_documents": len(docs["D2"]), "unaligned": miss, "examples": examples}
        exp = (EXPECT2 if V2 else EXPECT).get((ds, arm), {})
        for k, v in exp.items():
            if cell[f"{k}_changes"] != v:
                failed.append(f"{ds}/{arm} {k}: {cell[f'{k}_changes']} changes, recount {v}")
        log["jobs"][f"{ds}_{arm}"] = cell
        print(f"{ds:8s} {arm:17s} D1 {n['D1']:4d} | D1b {n['D1b']:4d} in {len(docs['D1b']):4d} docs | D2 {n['D2']:3d}"
              f" in {len(docs['D2']):3d} docs | unaligned {miss}", flush=True)
    log["F-1"] = "HELD" if not failed else "FAILED: " + "; ".join(failed)
    cj = f"{OUT}/changes{'_v2' if V2 else ''}.json"
    json.dump(log, open(cj + ".tmp", "w"), indent=1, ensure_ascii=False)
    os.replace(cj + ".tmp", cj)
    print("F-1:", log["F-1"])
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
