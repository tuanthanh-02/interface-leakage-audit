#!/usr/bin/env python3
"""Build SWAP, the misdirection construct of Section 4, from SACC-mask. Python standard library only.

SWAP starts from the SACC-mask release. Wherever SACC-mask redacted a SEX cue whose surface form has an
opposite-sex counterpart, SWAP writes that counterpart (he -> she, wife -> husband); every other
selected cue stays [REDACTED]. With --gate, the flipped words are written as [REDACTED] instead,
which must give back SACC-mask (the paper reports 99.7% of MedQA and all TAB documents).

    python build_swap.py --corpus medqa --out medqa_SWAP.jsonl
    python build_swap.py --corpus medqa --gate          # reproduction check against SACC-mask
    python build_swap.py --corpus medqa --check released/medqa_SWAP.jsonl

Inputs in data/: {corpus}_base.jsonl (the text before cue editing), {corpus}_sacc_mask.jsonl (the
released SACC-mask text) and {corpus}_cue_trace.jsonl (the editor's trace: one row per selected cue
with its offsets in the base text, surface form, syntactic role and attribute).

SWAP attributes false sex values to the people its texts describe. It is an evaluation construct,
not an anonymiser; see the paper's Ethical Considerations.
"""
import argparse, collections, difflib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
SWAP = {"he": "she", "she": "he", "him": "her", "himself": "herself", "herself": "himself", "hers": "his",
        "mother": "father", "father": "mother", "daughter": "son", "son": "daughter", "wife": "husband",
        "husband": "wife", "brother": "sister", "sister": "brother", "widow": "widower", "widower": "widow",
        "miss": "Mr"}


def swap_form(text, role):
    low = text.lower()
    if low == "her":
        out = "his" if role == "poss" else "him"
    elif low == "his":
        out = "her" if role == "poss" else "hers"
    else:
        out = SWAP[low]
    if out == "Mr":
        return out
    return out[0].upper() + out[1:] if text[:1].isupper() else out


def load(path):
    return {r["data_id"]: r["text"] for r in map(json.loads, open(path, encoding="utf-8"))}


def build(corpus, flip=True):
    base = load(f"{HERE}/data/{corpus}_base.jsonl")
    c1 = load(f"{HERE}/data/{corpus}_sacc_mask.jsonl")
    cues = collections.defaultdict(dict)
    for r in map(json.loads, open(f"{HERE}/data/{corpus}_cue_trace.jsonl", encoding="utf-8")):
        s0, e0 = r["base_offsets"]
        t, role, sx = cues[r["doc_id"]].get((s0, e0), (r["cue_text"], r.get("syntax_role"), False))
        cues[r["doc_id"]][(s0, e0)] = (t, role, sx or r["attribute"] == "SEX")
    out, stat = {}, collections.Counter()
    for d, t1 in c1.items():
        b = base.get(d)
        if b is None:
            out[d] = t1; stat["no_base"] += 1; continue
        pieces, flipped = [], 0
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, b, t1, autojunk=False).get_opcodes():
            if op == "equal":
                pieces.append(b[i1:i2]); continue
            inside = sorted((k, v) for k, v in cues[d].items() if k[0] >= i1 and k[1] <= i2)
            if not any(v[2] for _k, v in inside):
                pieces.append(t1[j1:j2]); continue
            pos = i1
            for (s0, e0), (txt, role, sx) in inside:
                if s0 < pos:
                    continue
                pieces.append(b[pos:s0])
                if sx and flip and txt.lower() in set(SWAP) | {"her", "his"}:
                    pieces.append(swap_form(b[s0:e0], role)); flipped += 1
                else:
                    pieces.append("[REDACTED]")
                pos = e0
            pieces.append(b[pos:i2])
        out[d] = "".join(pieces)
        stat["flipped"] += flipped
        stat["documents_flipped"] += flipped > 0
    return out, c1, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=["medqa", "tab"])
    ap.add_argument("--out")
    ap.add_argument("--gate", action="store_true", help="write flips as [REDACTED]; compare with SACC-mask")
    ap.add_argument("--check", help="a released SWAP jsonl to compare with, byte for byte per document")
    a = ap.parse_args()
    if a.gate:
        txt, c1, _ = build(a.corpus, flip=False)
        same = sum(txt[d] == c1[d] for d in c1)
        print(f"{a.corpus}: gate reproduces SACC-mask on {same}/{len(c1)} documents ({same / len(c1):.3%})")
        return
    txt, _, stat = build(a.corpus, flip=True)
    print(f"{a.corpus}: {len(txt)} documents, {dict(stat)}")
    if a.check:
        ref = {}
        for r in map(json.loads, open(a.check, encoding="utf-8")):
            ref[r["metadata"]["data_id"]] = r["anonymized_text"]
        same = sum(txt.get(d) == t for d, t in ref.items())
        print(f"identical to {os.path.basename(a.check)}: {same}/{len(ref)}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            for d in sorted(txt):
                fh.write(json.dumps({"metadata": {"data_id": d, "arm": "SWAP"}, "anonymized_text": txt[d]},
                                    ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
