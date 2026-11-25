#!/usr/bin/env python
"""Reproduce SACC's released texts from the anonymised inputs. CPU only; no model call.

    python reproduce_sacc.py                 # writes outputs/arms/{corpus}_{arm}.jsonl
    python reproduce_sacc.py --verify        # and compares every file byte for byte with released/arms/

Arms: C1v3 = SACC-mask (selection, every selected cue redacted); SACC_Nfix3 = SACC (MedQA: full agreement setting, the
setting used under the development adversary; TAB: light setting; SynthPAI: identical to SACC-mask, nothing is
rewritable); medqa SACC_Nfix3_light = SACC on MedQA in the light setting.
"""
import argparse, hashlib, importlib.util as il, json, os, re, sys, time

ROOT = os.environ.get("SACC_ROOT") or os.path.dirname(os.path.abspath(__file__))
os.environ["SACC_ROOT"] = ROOT
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
DSS = ("medqa", "tab", "synthpai")
INPUT = {"medqa": "runs/dev/arms/medqa_s1_only/anonymized.jsonl", "tab": "runs/dev/arms/tab_lf_only/anonymized.jsonl"}
WORD = re.compile(r"[a-z]+")


def _load(name, path):
    s = il.spec_from_file_location(name, path)
    m = il.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def load_arm(p):
    return {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"] for l in open(p)}


def write_arm(p, texts, arm):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as fh:
        for d in sorted(texts):
            fh.write(json.dumps({"metadata": {"data_id": d, "arm": arm}, "anonymized_text": texts[d]},
                                ensure_ascii=False) + "\n")
    os.replace(p + ".tmp", p)


def r9(orig):
    """binder rule R9: a sex or relationship noun whose words all occur in a protected subject's mention form
    describes that subject directly (applied around the base binder's OTHER_ENTITY decision)"""
    import dev_phase10_binding as B

    def bind_v3(doc, s0, e0, attribute, mentions, corpus, **kw):
        r = orig(doc, s0, e0, attribute, mentions, corpus, **kw)
        if r["status"] != "OTHER_ENTITY" or attribute not in ("SEX", "RELATIONSHIP"):
            return r
        surf = doc.text[s0:e0].strip().lower()
        if not surf or surf in B.PRONOUNS:
            return r
        words = WORD.findall(surf)
        for m in mentions:
            mw = set(WORD.findall(m.lower()))
            if words and all(w in mw for w in words):
                return dict(r, status="DIRECT_TARGET", rule_id="R9_cue_names_protected_subject",
                            direct_referent=m, supported_pairs=[["SUBJECT", attribute]])
        return r
    return bind_v3


def emit_sacc_mask(out):
    """sweep, bind (with R9), scope, redact: the SACC-mask texts of every corpus"""
    M = _load("emit_m2", "scripts/dev_phase10_emit_m2.py")
    M.D = f"{out}/emit"
    for sub in ("traces", "outputs"):
        os.makedirs(f"{M.D}/{sub}", exist_ok=True)
    orig = M.B.bind
    M.B.bind = r9(orig)
    try:
        for ds in DSS:
            sys.argv = ["emit", "--ds", ds, "--all"]
            M.main()
    finally:
        M.B.bind = orig
    rows = [json.loads(l) for l in open(f"{M.D}/outputs/C1.jsonl")]
    return {ds: {r["doc_id"]: r["text"] for r in rows if r["corpus"] == ds} for ds in DSS}


def base_of(ds):
    import dev_hitset_r1 as H
    H.COHORT[ds] = 10 ** 6
    items, _n = H.build(ds, require_sex=False, require_pool=False)
    return {it["doc_id"]: it["text"] for it in items}


def fixer():
    FX = _load("fx", "scripts/dev_sacc_fix_emit.py")
    FX.V2 = True
    return FX


def light(base, c1):
    ST = _load("st", "scripts/dev_sacc_stack_rewrite_emit.py")
    return {d: (ST.realise(base[d], t) if d in base else t) for d, t in c1.items()}


def light_v2path(base, c1, inp):
    ST = _load("st_orig", "scripts/dev_sacc_stack_rewrite_emit.py")
    ST.neutral = ST.P24.neutral_surface
    FX = fixer()
    out = {}
    for d, t in c1.items():
        r = ST.realise(base[d], t) if d in base else t
        out[d] = FX.fix(inp[d], r)[0] if d in inp else r
    return out


def full_setting(c1_pattern, out_dir):
    N2 = _load("n2", "scripts/dev_p22_n2_emit.py")
    N2.C1SRC, N2.OUT = c1_pattern, out_dir
    N2.main(["medqa"])
    return load_arm(f"{out_dir}/arms/medqa_SACC_N2.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    arms = f"{a.out}/arms"
    c1 = emit_sacc_mask(a.out)
    for ds in DSS:
        write_arm(f"{arms}/{ds}_C1v3.jsonl", c1[ds], "C1v3")
    FX = fixer()
    inp = load_arm(INPUT["medqa"])
    n2 = full_setting(f"{arms}/{{ds}}_C1v3.jsonl", f"{a.out}/full")
    full = {d: FX.fix(inp[d], t)[0] if d in inp else t for d, t in n2.items()}
    bm = base_of("medqa")
    outs = {("medqa", "SACC_Nfix3"): full, ("medqa", "SACC_Nfix3_light"): light_v2path(bm, c1["medqa"], inp),
            ("tab", "SACC_Nfix3"): light(base_of("tab"), c1["tab"]), ("synthpai", "SACC_Nfix3"): c1["synthpai"]}
    for (ds, arm), t in outs.items():
        write_arm(f"{arms}/{ds}_{arm}.jsonl", t, arm)
    print(f"written to {arms} in {time.time() - t0:.0f} s")
    if a.verify:
        bad = 0
        for f in sorted(os.listdir("released/arms")):
            h = [hashlib.sha256(open(p, "rb").read()).hexdigest() for p in (f"released/arms/{f}", f"{arms}/{f}")]
            print(f"{f:32s} {'identical' if h[0] == h[1] else 'DIFFERS'}")
            bad += h[0] != h[1]
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
