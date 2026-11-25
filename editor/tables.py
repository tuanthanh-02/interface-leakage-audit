#!/usr/bin/env python
"""Recompute the paper's main privacy tables from the released per-entry scores (numpy only).

    python tables.py

Reads scores/main/{corpus}_entries.jsonl (the development adversary's scores of every system, as the tables use them)
and recomputes Table 1's privacy levels and U_nli and Table 2's paired differences with the paper's balanced endpoint
and document bootstrap (4,000 draws, seed 0, two-sided 99%). Every number is compared with scores/main/expected.json.
"""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(ROOT, "scores", "main")
DELTA, N_BOOT, SEED = .05, 4000, 0
F = {"L": lambda p, pi: max(0., min(1., (p - pi) / (1 - pi))),
     "DISC": lambda p, pi: 0. if abs(p - pi) <= DELTA else 1.,
     "LEAK": lambda p, pi: 1. if p > pi + DELTA else 0.}
NAMES = {"Detector": "Anonymised input", "B2": "Global mask", "B6": "Global ladder", "INTACT": "INTACT",
         "Adaptive": "Adaptive", "C1": "SACC-mask", "SACC_Nstar": "SACC", "LLM": "LLM rewrite"}


class Panel:
    """balanced endpoint (stratum within attribute within corpus) with a document-cluster bootstrap"""

    def __init__(self, keys, prior, stratum, tags):
        self.keys = [k for k in keys if k[1] in tags]
        self.prior = prior
        docs = sorted({k[0].split(":")[0] for k in self.keys})
        di = {d: i for i, d in enumerate(docs)}
        cells = sorted({(k[1], stratum[k]) for k in self.keys})
        ci = {c: j for j, c in enumerate(cells)}
        self.ktag = np.array([tags.index(c[0]) for c in cells])
        self.ntags = len(tags)
        self.ix = [(di[k[0].split(":")[0]], ci[(k[1], stratum[k])]) for k in self.keys]
        self.shape = (len(docs), len(cells))
        self.C = np.zeros(self.shape)
        for i, j in self.ix:
            self.C[i, j] += 1
        rng = np.random.default_rng(SEED)
        self.W = rng.multinomial(len(docs), np.full(len(docs), 1 / len(docs)), size=N_BOOT).astype(float)
        self.one = np.ones((1, len(docs)))

    def _agg(self, S, W):
        tot, cnt = W @ S, W @ self.C
        have = cnt > 0
        mean = np.divide(tot, cnt, out=np.zeros_like(tot), where=have)
        vals = []
        for t in range(self.ntags):
            col = self.ktag == t
            h = have[:, col]
            vals.append(np.where(h, mean[:, col], 0).sum(1) / np.maximum(h.sum(1), 1))
        return np.mean(vals, axis=0)

    def scores(self, P, fn):
        S = np.zeros(self.shape)
        for (i, j), k in zip(self.ix, self.keys):
            S[i, j] += fn(P[k], self.prior[k])
        return S

    def level(self, P, fn):
        return float(self._agg(self.scores(P, fn), self.one)[0])

    def contrast(self, PA, PB, fn):
        SA, SB = self.scores(PA, fn), self.scores(PB, fn)
        d = float(self._agg(SA, self.one)[0] - self._agg(SB, self.one)[0])
        bs = self._agg(SA, self.W) - self._agg(SB, self.W)
        return {"d": d, "lo": float(np.percentile(bs, .5)), "hi": float(np.percentile(bs, 99.5))}


def main():
    X = json.load(open(os.path.join(M, "expected.json")))
    worst, checked = 0.0, 0

    def check(got, want):
        nonlocal worst, checked
        worst = max(worst, abs(round(got, 4) - want)); checked += 1

    for ds in ("medqa", "tab", "synthpai"):
        tags = tuple(X[ds]["tags"])
        P, PRI, STR, COMMON, SAME, PAIR = {}, {}, {}, set(), set(), {"Adaptive": {}, "SACC": {}}
        for l in open(os.path.join(M, f"{ds}_entries.jsonl")):
            r = json.loads(l)
            k = (r["entry"], r["attribute"])
            PRI[k], STR[k] = r["prior"], r["stratum"]
            for s, v in r["p"].items():
                P.setdefault(s, {})[k] = v
            if r["common"]:
                COMMON.add(k)
            if r["adaptive_same_reference"]:
                SAME.add(k)
                for s in PAIR:
                    PAIR[s][k] = r["same_reference_pair"][s]
        print(f"\n{ds}: {len(PRI)} entries, {len(COMMON)} shared by every ranked system")
        pan = Panel(sorted(COMMON), PRI, STR, tags)
        print(f"  {'System':18s} {'L':>6s} {'DISC':>6s} {'LEAK':>6s}   (Table 1, shared entries)")
        for s, want in X[ds]["levels"].items():
            got = {e: pan.level(P[s], F[e]) for e in ("L", "DISC", "LEAK")}
            for e in got:
                check(got[e], want[e])
            print(f"  {NAMES[s]:18s} " + " ".join(f"{got[e]:6.3f}" for e in ("L", "DISC", "LEAK")))
        lk = sorted(k for k in P.get("LLM", {}) if k in P["SACC_Nstar"])
        if lk:
            lp = Panel(lk, PRI, STR, tags)
            got = {e: lp.level(P["LLM"], F[e]) for e in ("L", "DISC", "LEAK")}
            for e in got:
                check(got[e], X[ds]["llm_levels"][e])
            print(f"  {'LLM rewrite':18s} " + " ".join(f"{got[e]:6.3f}" for e in ("L", "DISC", "LEAK")) + "   (own entries)")
        print(f"  {'System - SACC':18s} {'dL':>22s} {'dDISC':>22s} {'dLEAK':>22s}   (Table 2)")
        for s, want in X[ds]["vs_sacc"].items():
            ks = sorted(k for k in P[s] if k in P["SACC_Nstar"])
            pp = Panel(ks, PRI, STR, tags)
            c = {e: pp.contrast(P[s], P["SACC_Nstar"], F[e]) for e in ("L", "DISC", "LEAK")}
            for e in c:
                for x in ("d", "lo", "hi"):
                    check(c[e][x], want[e][x])
            print(f"  {NAMES[s]:18s} " + " ".join(f"{c[e]['d']:+.3f} [{c[e]['lo']:+.3f},{c[e]['hi']:+.3f}]" for e in c))
        if "adaptive_same_reference" in X[ds]:
            pp = Panel(sorted(SAME), PRI, STR, tags)
            c = {e: pp.contrast(PAIR["Adaptive"], PAIR["SACC"], F[e]) for e in ("L", "DISC", "LEAK")}
            for e in c:
                for x in ("d", "lo", "hi"):
                    check(c[e][x], X[ds]["adaptive_same_reference"][e][x])
            print(f"  {'Adaptive (same ref)':18s} " + " ".join(f"{c[e]['d']:+.3f} [{c[e]['lo']:+.3f},{c[e]['hi']:+.3f}]" for e in c))
        U = json.load(open(os.path.join(M, f"{ds}_unli.json")))["systems"]
        docs = sorted(set.intersection(*[set(v) for v in U.values()]))
        for s, want in X[ds]["unli_levels"].items():
            check(float(np.mean([U[s][d] for d in docs])), want)
        print("  U_nli (Table 1): " + ", ".join(f"{NAMES[s]} {np.mean([U[s][d] for d in docs]):.3f}" for s in X[ds]["unli_levels"]))
    ok = worst < 6e-4
    print(f"\n{checked} numbers compared with the paper; largest difference {worst:.4f}: {'MATCH' if ok else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
