#!/usr/bin/env python3
"""Hitting-set anonymisation — pilot R1, greedy restoration (API: gpt-4o-mini, ledger-capped).

Prereg: runs/dev/preregs/hitting_set_prereg.md, Part A (RP-1..RP-3, K-R1, K-R2).
Starts from every candidate masked, where the posterior sits at its floor, and restores candidates one at a time in a
frozen order while the posterior stays at that floor. What can be restored is the room the method would have.

    dev_hitset_r1.py --smoke        3 documents per corpus, writes the projection; no full run without it
    dev_hitset_r1.py --run          the registered 60 MedQA + 40 TAB documents
    dev_hitset_r1.py --judge        CPU: RP-1..RP-3 and the kill conditions
Writes runs/dev/hitset/r1/
"""
import argparse
import collections
import hashlib
import json
import os
import random
import sys
import threading

SP = __import__("os").environ["SACC_ROOT"]
os.chdir(SP)
sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(SP, "scripts"))
from src.relaware import registry, profiles, probe                # noqa: E402
from src.relaware.probe_openai import OpenAIProbe                 # noqa: E402
import dev_phase10_repair as R                                    # noqa: E402
import dev_phase10_emit_m1 as M1E                                 # noqa: E402
import dev_saccv2_s0_census as C0                                 # noqa: E402
import dev_saccv2_s1_emit as M                                    # noqa: E402

OUT = "runs/dev/hitset/r1"                       # R1; Addendum 3 redirects outputs to r1b, cache and ledger stay here
BASE = "runs/dev/hitset/r1"
ORDERS = 1
assert OUT.startswith("runs/dev/"), "dev track only"
PREREG = "runs/dev/preregs/hitting_set_prereg.md"
COHORT = {"medqa": 60, "tab": 40}
SEED, DELTA, MODEL, CAP = 0, 0.05, "gpt-4o-mini", 0.60
PRICE = (0.15, 0.60)
TAGS = ("SEX", "OCCUPATION", "RELATIONSHIP")
import dev_phase9_lib as L9                                       # noqa: E402


def ORDER_SET(n):
    """Addendum 3: with two options both orders are enumerated exactly, which cancels answer-position bias"""
    if ORDERS == 1:
        return [list(range(n))]
    assert n == 2, "the exact two-order rule is registered for a binary attribute"
    return [[0, 1], [1, 0]]


def sha16(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


class Ledger:
    def __init__(self, path, cap):
        self.path, self.cap, self.lock, self.usd = path, cap, threading.Lock(), 0.0
        if os.path.exists(path):
            for l in open(path):
                self.usd += json.loads(l)["usd"]

    def add(self, tag, pin, pout):
        usd = (pin * PRICE[0] + pout * PRICE[1]) / 1e6
        with self.lock:
            self.usd += usd
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps({"tag": tag, "prompt_tokens": pin, "completion_tokens": pout,
                                     "usd": round(usd, 8)}) + "\n")

    def over(self):
        return self.usd >= self.cap


def build(ds, require_sex=True, require_pool=True):
    """documents, their SEX entry (optional) and the U5 candidate pool; reads no API"""
    tree, corp, *_ = M1E.CFG[ds]
    reg = registry.load(f"{tree}/registry.json")
    det = json.load(open(f"{tree}/detections_d1.json"))
    cues_of = {tuple(k.split("||")): v for k, v in det["cues"].items()}
    prof = profiles.for_registry(reg)
    SC = C0._load("sc", "scripts/dev_phase8_scope_control.py")
    _c, base_spans = C0.make_base(ds, M1E.CFG)
    subs = collections.defaultdict(list)
    for s in reg["subjects"]:
        subs[s["doc_id"]].append(s)
    ent = {}
    for e in reg["entries"]:
        if e["attribute"] == "SEX" and str(e.get("gold_value", "")).lower() in ("male", "female"):
            ent.setdefault(e["doc_id"], e)
    items = []
    for line in open(corp):
        rec = json.loads(line)
        did, src = rec["metadata"]["data_id"], rec["text"]
        if did not in ent and require_sex:
            continue
        if did not in subs:
            continue
        bsp = base_spans(did, src)
        base_txt = M1E.v1_sweep(src, bsp, [])
        m = M1E.offset_map(src, base_txt)
        # the frozen union, exactly as the emitter builds it
        union = {}
        for s in subs.get(did, []):
            for t in TAGS:
                for c in cues_of.get((s["eval_subject_id"], t), []):
                    s0, e0 = c["start"], c["end"]
                    if s0 in m and (e0 - 1) in m and m[e0 - 1] + 1 - m[s0] == e0 - s0:
                        union[(m[s0], m[e0 - 1] + 1, t)] = {"surface": c["surface"], "start": m[s0],
                                                            "end": m[e0 - 1] + 1}
        for t, cs in SC.global_groups(base_txt, list(TAGS), prof).items():
            for c in cs:
                union[(c["start"], c["end"], t)] = {"surface": base_txt[c["start"]:c["end"]],
                                                    "start": c["start"], "end": c["end"]}
        by = collections.defaultdict(list)
        for (s0, e0, t), c in union.items():
            by[t].append(dict(c))
        pool = set()
        if by:
            for q in M.plan_items(base_txt, by):
                if q["action"] != "KEEP":
                    ap = q.get("applied") or [q["start"], q["end"]]
                    pool.add((ap[0], ap[1]))
        for mm in M.SEXCLIN_RE.finditer(base_txt):
            if not C0.overlaps(mm.start(), mm.end(), list(pool)):
                pool.add((mm.start(), mm.end()))
        for x in M.emoji_extras(base_txt, list(pool)):
            pool.add((x["start"], x["end"]))
        pool = sorted(pool)
        if not pool and require_pool:
            continue
        e = ent.get(did)
        s = next((x for x in subs[did] if e and x["eval_subject_id"] == e["eval_subject_id"]), subs[did][0])
        items.append({"doc_id": did, "text": base_txt, "entry": e, "subject": s,
                      "n_subjects": len(subs[did]), "pool": pool,
                      "src": src, "bsp": bsp, "subjects": subs[did]})
    items.sort(key=lambda x: x["doc_id"])
    n = min(COHORT[ds], len(items))
    pick = sorted(random.Random(SEED).sample(range(len(items)), n))
    return [items[i] for i in pick], len(items)


def render(base_txt, masked):
    out = base_txt
    for s0, e0 in sorted(masked, reverse=True):
        out = out[:s0] + R.MASK + out[e0:]
    return out


def run(a):
    if "RP-1" not in open(PREREG).read():
        sys.exit("the hitting-set prereg must be registered first")
    os.makedirs(OUT, exist_ok=True)
    ledger = Ledger(f"{BASE}/ledger.jsonl", CAP)
    if not a.smoke:
        sm = f"{BASE}/smoke.json"
        if not os.path.exists(sm):
            sys.exit("run --smoke first")
        proj = json.load(open(sm))["projected_usd"] * ORDERS
        spent = Ledger(f"{BASE}/ledger.jsonl", CAP).usd
        if proj > CAP - spent:
            sys.exit(f"projection {proj:.3f} USD exceeds what is left of the cap ({CAP - spent:.3f}); not started")
    pr = OpenAIProbe(model=MODEL, cache=f"{BASE}/probe_cache.sqlite")
    spend_before = ledger.usd            # the ledger is cumulative across runs; a projection uses this run only
    records, failed = [], 0
    for ds in COHORT:
        items, total = build(ds)
        if a.smoke:
            items = items[:3]
        for it in items:
            e, s = it["entry"], it["subject"]
            ref = L9.audit_reference(ds, s["gold_mention"], it["text"], it["n_subjects"])["reference"]
            opts, gold = e["candidates"], e["gold_value"]
            perms = ORDER_SET(len(opts))

            def p_of(text):
                before = (pr.stats.get("prompt_tokens", 0), pr.stats.get("completion_tokens", 0))
                p, _per, bad = pr.p_of(lambda o: probe.posterior_prompt(text, ref, probe.ATTR_Q["SEX"], o),
                                       opts, gold, perms)
                after = (pr.stats.get("prompt_tokens", 0), pr.stats.get("completion_tokens", 0))
                ledger.add(f"{ds}|{it['doc_id']}", after[0] - before[0], after[1] - before[1])
                return None if bad else p

            pool = it["pool"]
            masked = set(pool)
            p0 = p_of(render(it["text"], masked))
            if p0 is None:
                failed += 1
                continue
            floor = max(p0, 1.0 / max(2, len(opts))) + DELTA
            order = sorted(pool, key=lambda sp: (-(sp[1] - sp[0]), sp[0]))
            restored, trace = [], []
            for sp in order:
                if ledger.over():
                    print(f"cap {CAP} reached; stopping")
                    break
                trial = masked - {sp}
                p = p_of(render(it["text"], trial))
                if p is None:
                    failed += 1
                    continue
                keep = p <= floor
                if keep:
                    masked = trial
                    restored.append(sp)
                trace.append({"span": list(sp), "chars": sp[1] - sp[0], "p": round(p, 6), "kept": int(keep)})
            p_end = p_of(render(it["text"], masked))
            pool_chars = sum(e0 - s0 for s0, e0 in pool)
            rec = {"ds": ds, "doc_id": it["doc_id"], "pool": len(pool), "pool_chars": pool_chars,
                   "restored": len(restored), "restored_chars": sum(e0 - s0 for s0, e0 in restored),
                   "p0": round(p0, 6), "floor": round(floor, 6),
                   "p_end": None if p_end is None else round(p_end, 6),
                   "p0_above_chance": int(p0 > 1.0 / max(2, len(opts)) + DELTA),
                   "restored_frac": round(len(restored) / len(pool), 4),
                   "restored_char_frac": round(sum(e0 - s0 for s0, e0 in restored) / max(1, pool_chars), 4),
                   "trace": trace}
            records.append(rec)
            print(f"  {ds} {it['doc_id']}: pool {len(pool)}, restored {len(restored)} "
                  f"({rec['restored_frac']:.0%} of spans, {rec['restored_char_frac']:.0%} of chars), "
                  f"p0 {p0:.3f} -> {rec['p_end']}")
    tag = "smoke" if a.smoke else "run"
    blob = {"tag": tag, "model": MODEL, "orders": ORDERS, "delta": DELTA, "records": records,
            "failed_calls": failed, "ledger_usd": round(ledger.usd, 6),
            "spent_this_run": round(ledger.usd - spend_before, 6), "prereg_sha16": sha16(PREREG)}
    if a.smoke:
        spent_here = ledger.usd - spend_before
        per_doc = spent_here / max(1, len(records))
        blob["projected_usd"] = round(per_doc * sum(COHORT.values()), 4)
        blob["spent_this_run"] = round(spent_here, 6)
    with open(f"{OUT}/{tag}.json.tmp", "w") as fh:
        json.dump(blob, fh, ensure_ascii=False, indent=1)
    os.replace(f"{OUT}/{tag}.json.tmp", f"{OUT}/{tag}.json")
    print(f"[{tag}] {len(records)} documents, {failed} failed calls, {ledger.usd:.4f} USD"
          + (f"; projected {blob['projected_usd']:.3f} USD" if a.smoke else ""))
    return 0


def priors():
    """Addendum 2: the prober's own prior per entry, from the frozen prior prompt (handle and domain, no text)"""
    if "Addendum 2" not in open(PREREG).read():
        sys.exit("Addendum 2 must be registered before any prior call")
    blob = json.load(open(f"{OUT}/run.json"))
    want = {(r["ds"], r["doc_id"]) for r in blob["records"]}
    ledger = Ledger(f"{BASE}/ledger.jsonl", CAP)
    pr = OpenAIProbe(model=MODEL, cache=f"{BASE}/probe_cache.sqlite")
    rows = []
    for ds in COHORT:
        tree, *_ = M1E.CFG[ds]
        domain = registry.load(f"{tree}/registry.json")["domain"]
        items, _n = build(ds)
        for it in items:
            if (ds, it["doc_id"]) not in want:
                continue
            e, s = it["entry"], it["subject"]
            opts, gold = e["candidates"], e["gold_value"]
            before = (pr.stats.get("prompt_tokens", 0), pr.stats.get("completion_tokens", 0))
            p, _per, bad = pr.p_of(lambda o: probe.prior_prompt(s["canonical_handle"], domain, probe.ATTR_Q["SEX"], o),
                                   opts, gold, ORDER_SET(len(opts)))
            after = (pr.stats.get("prompt_tokens", 0), pr.stats.get("completion_tokens", 0))
            ledger.add(f"prior|{ds}|{it['doc_id']}", after[0] - before[0], after[1] - before[1])
            rows.append({"ds": ds, "doc_id": it["doc_id"], "handle": s["canonical_handle"],
                         "prior": None if bad else round(p, 6), "n_options": len(opts)})
    with open(f"{OUT}/priors.json.tmp", "w") as fh:
        json.dump({"model": MODEL, "priors": rows, "ledger_usd_total": round(ledger.usd, 6)}, fh, indent=1)
    os.replace(f"{OUT}/priors.json.tmp", f"{OUT}/priors.json")
    print(f"{len(rows)} priors, {sum(1 for r in rows if r['prior'] is None)} failed; ledger total {ledger.usd:.4f} USD")
    return 0


def rejudge():
    """Addendum 2: re-read the recorded R1 run with the prior floor; no restoration call is made"""
    import dev_phase8_lib as L8
    blob = json.load(open(f"{OUT}/run.json"))
    pri = {(r["ds"], r["doc_id"]): r for r in json.load(open(f"{OUT}/priors.json"))["priors"]}
    tau = {ds: L8.decisions(ds)["high_prior_tau"] for ds in COHORT}
    by = collections.defaultdict(list)
    for r in blob["records"]:
        q = pri.get((r["ds"], r["doc_id"]))
        if q is None or q["prior"] is None:
            continue
        chance = 1.0 / max(2, q["n_options"])
        floor = max(q["prior"], chance)
        by[r["ds"]].append(dict(r, prior=q["prior"], excluded=q["prior"] >= tau[r["ds"]],
                                in_scope=r["p0"] <= floor + DELTA))
    out = {"tau": tau}
    for ds, v in by.items():
        kept = [r for r in v if not r["excluded"]]
        ins = [r for r in kept if r["in_scope"]]
        oos = [r for r in kept if not r["in_scope"]]
        out[ds] = {"documents": len(v), "excluded_high_prior": len(v) - len(kept), "in_scope": len(ins),
                   "out_of_scope": len(oos),
                   "out_of_scope_share": round(len(oos) / max(1, len(kept)), 3),
                   "median_prior": median([r["prior"] for r in v]),
                   "median_prior_out_of_scope": median([r["prior"] for r in oos]),
                   "median_restored_frac": median([r["restored_frac"] for r in ins]),
                   "median_restored_char_frac": median([r["restored_char_frac"] for r in ins]),
                   "chance_floor_in_scope_was": sum(1 for r in v if not r["p0_above_chance"])}
    out["RP-1"] = {"holds": any((out[ds]["median_restored_frac"] or 0) >= .30 for ds in by)}
    out["RP-2"] = {"holds": any((out[ds]["median_restored_char_frac"] or 0) >= .20 for ds in by)}
    out["K-R2"] = {ds: out[ds]["out_of_scope_share"] > .5 for ds in by}
    with open(f"{OUT}/r1_predictions_prior_floor.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


def median(xs):
    xs = sorted(xs)
    return None if not xs else (xs[len(xs) // 2] if len(xs) % 2 else (xs[len(xs) // 2 - 1] + xs[len(xs) // 2]) / 2)


def judge():
    blob = json.load(open(f"{OUT}/run.json"))
    by = collections.defaultdict(list)
    for r in blob["records"]:
        by[r["ds"]].append(r)
    ins = {ds: [r for r in v if not r["p0_above_chance"]] for ds, v in by.items()}      # Addendum 1: in scope
    out = {"documents": {ds: len(v) for ds, v in by.items()},
           "in_scope": {ds: len(v) for ds, v in ins.items()},
           "out_of_scope_share": {ds: round(sum(r["p0_above_chance"] for r in v) / len(v), 3) for ds, v in by.items()},
           "median_restored_frac": {ds: median([r["restored_frac"] for r in v]) for ds, v in ins.items()},
           "median_restored_char_frac": {ds: median([r["restored_char_frac"] for r in v]) for ds, v in ins.items()},
           "median_restored_frac_ALL_DOCS": {ds: median([r["restored_frac"] for r in v]) for ds, v in by.items()},
           "mean_pool": {ds: round(sum(r["pool"] for r in v) / len(v), 1) for ds, v in by.items()},
           "usd": blob["ledger_usd"]}
    out["RP-1"] = {"holds": any((v or 0) >= .30 for v in out["median_restored_frac"].values())}
    out["RP-2"] = {"holds": any((v or 0) >= .20 for v in out["median_restored_char_frac"].values())}
    bad = [r["doc_id"] for r in blob["records"] if r["p_end"] is not None and r["p_end"] > r["floor"] + 1e-9]
    out["RP-3"] = {"violations": bad[:5], "n_violations": len(bad), "holds": not bad}
    out["kill"] = {"K-R1": all((v or 0) < .10 for v in out["median_restored_frac"].values()),
                   "K-R2": {ds: v > .5 for ds, v in out["out_of_scope_share"].items()}}
    out["note"] = ("RP-1 and RP-2 are read on in-scope documents only (Addendum 1): a document counts only if masking "
                   "the whole pool brings the prober to chance.")
    with open(f"{OUT}/r1_predictions.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--priors", action="store_true")
    ap.add_argument("--rejudge", action="store_true")
    ap.add_argument("--orders", type=int, default=1, choices=(1, 2))
    a = ap.parse_args()
    if a.orders == 2:
        ORDERS = 2
        OUT = "runs/dev/hitset/r1b"
        os.makedirs(OUT, exist_ok=True)
    if a.judge:
        sys.exit(judge())
    if a.priors:
        sys.exit(priors())
    if a.rejudge:
        sys.exit(rejudge())
    if not (a.smoke or a.run):
        sys.exit("give --smoke, --run or --judge")
    sys.exit(run(a))
