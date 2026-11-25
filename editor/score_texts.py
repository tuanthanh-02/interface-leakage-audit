#!/usr/bin/env python
"""Score texts with the paper's adversary probe (Section 5 and Appendix C of the paper).

    # priors of an adversary (once per corpus and model; no text)
    python score_texts.py priors --ds medqa --adversary openai --model gpt-4o-mini --out my_priors_medqa.json
    # posteriors of one system's texts
    python score_texts.py posterior --ds medqa --texts released/arms/medqa_SACC_Nfix3.jsonl --method SACC \
        --adversary openai --model gpt-4o-mini --priors scores/priors/gpt4omini_medqa.json --out my_scores.json
    # a local vLLM server instead of the API
    python score_texts.py posterior ... --adversary vllm --base-url http://localhost:8032 --model llama-3.1-8b-instruct

Each entry (protected subject, attribute) is asked a lettered multiple-choice question in eight seeded candidate orders;
the posterior is the mean probability of the gold value. Entries whose prior reaches the corpus's frozen threshold are
left out. The subject reference is built from the scored text (Appendix C). For the API, put OPENAI_API_KEY=... in a
file `.env` at the package root.
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.environ.get("SACC_ROOT") or os.path.dirname(os.path.abspath(__file__))
os.environ["SACC_ROOT"] = ROOT
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src.relaware import candidates, probe, registry          # noqa: E402
import dev_phase9_lib as L9                                    # noqa: E402  (the subject reference)

TREE = {"medqa": "runs/relaware_medqa", "tab": "runs/relaware", "synthpai": "runs/relaware_synthpai"}
N_PERM = 8


def load_arm(p):
    return {json.loads(l)["metadata"]["data_id"]: json.loads(l)["anonymized_text"] for l in open(p)}


def make_probe(a):
    if a.adversary == "openai":
        from src.relaware.probe_openai import OpenAIProbe
        return OpenAIProbe(model=a.model, cache=a.cache)
    import dev_phase10_e2lib as E2                              # every option scored, teacher forcing where needed
    return E2.FullOptionProbe(a.base_url, a.model, log_path=a.out + ".probe_log.jsonl")


def setup(ds):
    reg = registry.load(f"{TREE[ds]}/registry.json")
    dec = json.load(open(f"{TREE[ds]}/phase0_decisions.json"))
    subs = {s["eval_subject_id"]: s for s in reg["subjects"]}
    nsub = {}
    for s in reg["subjects"]:
        nsub[s["doc_id"]] = nsub.get(s["doc_id"], 0) + 1
    return reg, dec, subs, nsub


def priors(a):
    reg, _dec, subs, _n = setup(a.ds)
    pr = make_probe(a)
    ents = [e for e in reg["entries"] if e.get("candidates")]

    def one(e):
        s = subs[e["eval_subject_id"]]
        perms = candidates.permutations(e, len(e["candidates"]), n_perm=N_PERM)
        p, _x, failed = pr.p_of(lambda o: probe.prior_prompt(s["canonical_handle"], reg["domain"],
                                                             probe.ATTR_Q[e["attribute"]], o),
                                e["candidates"], e["gold_value"], perms)
        return {"eval_subject_id": e["eval_subject_id"], "attribute": e["attribute"], "prior": None if failed else p}
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, ents))
    json.dump({"ds": a.ds, "model": a.model, "n_perm": N_PERM, "registry_hash": reg["_freeze_hash"], "priors": rows},
              open(a.out, "w"), indent=1)
    print(f"{len(rows)} priors -> {a.out}")
    return 0


def posterior_records(a, pr=None, only=None):
    reg, dec, subs, nsub = setup(a.ds)
    tau = dec["high_prior_tau"]
    lvl = {"OCCUPATION": dec["strata"].get("OCC"), "RELATIONSHIP": dec["strata"].get("REL")}
    P = {(r["eval_subject_id"], r["attribute"]): r["prior"] for r in json.load(open(a.priors))["priors"]}
    tx = load_arm(a.texts)
    ents = [e for e in reg["entries"] if e["doc_id"] in tx and e.get("candidates")
            and P.get((e["eval_subject_id"], e["attribute"])) is not None
            and P[(e["eval_subject_id"], e["attribute"])] < tau
            and (only is None or (e["eval_subject_id"], e["attribute"]) in only)]
    pr = pr or make_probe(a)

    def one(e):
        s = subs[e["eval_subject_id"]]
        t = tx[e["doc_id"]]
        ref = L9.audit_reference(a.ds, s["gold_mention"], t, nsub[e["doc_id"]])
        perms = candidates.permutations(e, len(e["candidates"]), n_perm=N_PERM)
        p, _x, failed = pr.p_of(lambda o: probe.posterior_prompt(t, ref["reference"], probe.ATTR_Q[e["attribute"]], o),
                                e["candidates"], e["gold_value"], perms)
        st = e["stratum"].get(lvl[e["attribute"]]) if lvl.get(e["attribute"]) else e["gold_value"]
        return {"method": a.method, "eval_subject_id": e["eval_subject_id"], "attribute": e["attribute"], "stratum": st,
                "prior": P[(e["eval_subject_id"], e["attribute"])], "p_source_aware": None if failed else p,
                "handle": ref["reference"]}
    with ThreadPoolExecutor(a.workers) as ex:
        return list(ex.map(one, ents))


def posterior(a):
    rows = posterior_records(a)
    json.dump({"ds": a.ds, "method": a.method, "model": a.model, "n_perm": N_PERM, "results": rows},
              open(a.out, "w"), indent=1)
    print(f"{len(rows)} entries -> {a.out}")
    return 0


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("priors", "posterior"))
    ap.add_argument("--ds", required=True, choices=tuple(TREE))
    ap.add_argument("--texts")
    ap.add_argument("--method", default="system")
    ap.add_argument("--adversary", choices=("openai", "vllm"), default="openai")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--base-url", default="http://localhost:8032")
    ap.add_argument("--priors")
    ap.add_argument("--cache", default="outputs/request_cache.sqlite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    return ap


if __name__ == "__main__":
    a = parser().parse_args()
    sys.exit(priors(a) if a.stage == "priors" else posterior(a))
