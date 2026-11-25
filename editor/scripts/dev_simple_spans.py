#!/usr/bin/env python
"""S1 — a simple, registry-blind span layer, and its stage-1 measurement.

Prereg: runs/dev/preregs/v2_simple_spanlayer_prereg.md (frozen before this
file was written). Dev track: every output goes under runs/dev/.

S1 replaces `rel_phase1_oracle.gold_spans` — the ORACLE layer the `sacc_d1`
arm is built on — with four registry-blind components:

    NAME          spaCy en_core_web_sm NER, PERSON only
    SEX           closed-class value gazetteer
    RELATIONSHIP  closed-class marital/partnership status gazetteer
    OCCUPATION    WordNet: noun (or bigram) whose hypernym path reaches
                  person.n.01 through worker.n.01 or professional.n.01;
                  spaCy POS gates it to NOUN/PROPN so verb homographs
                  ("he chairs the board", "to nurse") cannot fire

Nothing here reads the registry. `gold_spans` is imported ONLY as the
measurement target, never as an input to detection.

    python scripts/dev_simple_spans.py --ds synthpai,medqa,tab,panorama \
        --out runs/dev/detector/s1_stage1.json
"""
import argparse
import collections
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
from src.relaware import registry                                  # noqa: E402

TREES = {"synthpai": "runs/relaware_synthpai", "medqa": "runs/relaware_medqa",
         "tab": "runs/relaware", "panorama": "runs/relaware_panorama",
}
CORPUS = {"synthpai": "data/spia/spia_synthpai.jsonl",
          "medqa": "data/spia/spia_medqa.jsonl",
          "tab": "data/spia/spia_tab_144.jsonl",
          "panorama": "data/spia/spia_panorama_151.jsonl",
}

# ---------------------------------------------------------------- gazetteers
# Closed classes. Enumerable without seeing any corpus; that is the whole
# claim being made for them. Written as alternations, longest-first inside
# each group so "in a relationship" wins over "relationship".
SEX_TERMS = [
    "gentlemen", "gentleman", "females", "female", "males", "male",
    "women", "woman", "girls", "girl", "boys", "boy", "guys", "guy",
    "ladies", "lady", "men", "man", "madam", "sir",
]
# AMENDMENT 1 / A2: titles are matched CASE-SENSITIVELY and must be followed
# by whitespace, the same guard the frozen THIRD["title"] uses. Without it
# "miss" the verb fired 16 times on SynthPAI alone.
TITLE_PAT = re.compile(r"\b(?:Mr|Mrs|Ms|Miss)\.?(?=\s)")
REL_TERMS = [
    "in a relationship", "in a long-term relationship", "in a partnership",
    "girlfriends", "girlfriend", "boyfriends", "boyfriend",
    "unmarried", "married", "marriage", "divorcing", "divorced", "divorce",
    "separated", "widowered", "widowed", "widower", "widows", "widow",
    "engaged", "engagement", "partnered", "cohabiting", "cohabitating",
    "spouses", "spouse", "husbands", "husband", "wives", "wife",
    "fiancee", "fiance", "fiancée", "fiancé",
    # AMENDMENT 1 / A1: plurals of terms already listed. No new concept --
    # SEX_TERMS carried its plurals from the start and this list did not.
    "singles", "single", "marriages", "divorces", "engagements",
    "widowers", "partners", "relationships",
]


def _alt(terms):
    return re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms)
                      + r")\b", re.I)


SEX_PAT = _alt(SEX_TERMS)
REL_PAT = _alt(REL_TERMS)

# ------------------------------------------------------------ WordNet OCC
_ART = re.compile(r"^(?:the|a|an)\s+", re.I)


class OccLex:
    """noun -> is it an occupation, by WordNet hypernymy. Memoised.

    AMENDMENT 1 / A3 -- two gates, both pre-existing, reported side by side:
      tight  path reaches person.n.01 THROUGH worker.n.01 or
             professional.n.01   (the frozen stage-1 gate)
      wide   path reaches person.n.01   (the criterion dev_ladder_wordnet.py
             already uses, registered 2026-09-02)
    """

    def __init__(self, gate="tight"):
        from nltk.corpus import wordnet as wn
        assert gate in ("tight", "wide")
        self.gate = gate
        self.wn = wn
        self.person = wn.synset("person.n.01")
        self.roots = {wn.synset("worker.n.01"), wn.synset("professional.n.01")}
        self._cache = {}

    def __call__(self, surface):
        t = _ART.sub("", surface.lower().strip()).replace(" ", "_")
        if t in self._cache:
            return self._cache[t]
        hit = False
        for syn in self.wn.synsets(t, pos="n"):
            for path in syn.hypernym_paths():
                if self.person not in path:
                    continue
                if self.gate == "wide" or any(r in path for r in self.roots):
                    hit = True
                    break
            if hit:
                break
        self._cache[t] = hit
        return hit


# ------------------------------------------------------------------- detect
def _nonoverlapping(spans):
    """longest-first, left to right — the same rule gold_spans uses."""
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    kept, last = [], -1
    for s in spans:
        if s[0] >= last:
            kept.append(s)
            last = s[1]
    return kept


def s1_spans(text, nlp, occ):
    """The S1 span layer for one document. Reads text only."""
    doc = nlp(text)
    spans = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            spans.append((ent.start_char, ent.end_char, "NAME"))
    for m in SEX_PAT.finditer(text):
        spans.append((m.start(), m.end(), "SEX"))
    for m in TITLE_PAT.finditer(text):
        spans.append((m.start(), m.end(), "SEX"))
    for m in REL_PAT.finditer(text):
        spans.append((m.start(), m.end(), "RELATIONSHIP"))
    # OCCUPATION: bigram first (software developer, tour guide), then unigram.
    toks = [t for t in doc if not t.is_space]
    for i, t in enumerate(toks):
        if t.pos_ not in ("NOUN", "PROPN"):
            continue
        nxt = toks[i + 1] if i + 1 < len(toks) else None
        if (nxt is not None and nxt.pos_ in ("NOUN", "PROPN")
                and nxt.idx > t.idx and occ(text[t.idx:nxt.idx + len(nxt)])):
            spans.append((t.idx, nxt.idx + len(nxt), "OCCUPATION"))
            continue
        if occ(t.text):
            spans.append((t.idx, t.idx + len(t.text), "OCCUPATION"))
    return _nonoverlapping(spans)


# -------------------------------------------------------------- measurement
_W = re.compile(r"\w")


def midword(text, a, b):
    """a boundary that cuts a word in half — the quarantine check"""
    left = a > 0 and _W.match(text[a - 1]) and _W.match(text[a])
    right = b < len(text) and _W.match(text[b - 1]) and _W.match(text[b])
    return bool(left or right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="synthpai,medqa,tab,panorama")
    ap.add_argument("--out", default="runs/dev/detector/s1_stage1.json")
    ap.add_argument("--spacy-model", default="en_core_web_sm")
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--gate", default="tight,wide",
                    help="AMENDMENT 1 / A3: WordNet OCCUPATION gates to "
                         "measure, comma separated")
    args = ap.parse_args()
    assert args.out.startswith("runs/dev/"), "dev track: runs/dev/ only"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "p1", os.path.join(REPO, "scripts", "rel_phase1_oracle.py"))
    p1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p1)                 # the REAL gold_spans, as the
                                                # measurement TARGET only

    import spacy
    nlp = spacy.load(args.spacy_model, disable=["lemmatizer"])

    report = {"spacy_model": args.spacy_model,
              "prereg": "runs/dev/preregs/v2_simple_spanlayer_prereg.md",
              "corpora": {}}

    for gate in args.gate.split(","):
      occ = OccLex(gate)
      for ds in args.ds.split(","):
          reg = registry.load(f"{TREES[ds]}/registry.json")
          docs = [json.loads(l) for l in open(CORPUS[ds])]
          gtot = gcov = s1chars = gchars = 0
          gn = gnhit = 0
          per_tag = collections.defaultdict(lambda: [0, 0])
          mid = ns1 = 0
          missed = collections.Counter()
          fired = collections.Counter()
          samples = []
          for rec in docs:
              did = rec["metadata"]["data_id"]
              text = rec["text"]
              gs = p1.gold_spans(text, reg, did)
              ss = s1_spans(text, nlp, occ)
              ns1 += len(ss)
              s1chars += sum(b - a for a, b, _ in ss)
              gchars += sum(b - a for a, b, _ in gs)
              for a, b, _t in ss:
                  if midword(text, a, b):
                      mid += 1
                  fired[text[a:b].lower()] += 1
              cov = set()
              for a, b, _t in ss:
                  cov.update(range(a, b))
              for a, b, tag in gs:
                  n = b - a
                  gtot += n
                  gn += 1
                  ov = len(cov & set(range(a, b)))
                  gcov += ov
                  per_tag[tag][0] += 1
                  if ov > 0:
                      gnhit += 1
                      per_tag[tag][1] += 1
                  else:
                      missed[(tag, text[a:b].lower())] += 1
              if len(samples) < args.samples and ss:
                  samples.append({"doc": did,
                                  "spans": [[text[a:b], t] for a, b, t in ss[:20]]})
          report["corpora"][f"{ds}/{gate}"] = {
              "n_docs": len(docs),
              "gold_spans": gn,
              "s1_spans": ns1,
              "char_recall": round(gcov / gtot, 4) if gtot else None,
              "span_recall_any_overlap": round(gnhit / gn, 4) if gn else None,
              "per_tag_recall": {k: [v[0], v[1],
                                     round(v[1] / v[0], 4) if v[0] else None]
                                 for k, v in sorted(per_tag.items())},
              "gold_masked_chars": gchars,
              "s1_masked_chars": s1chars,
              "volume_ratio": round(s1chars / gchars, 3) if gchars else None,
              "midword_spans": mid,
              "midword_rate": round(mid / ns1, 4) if ns1 else None,
              "top_missed": [[list(k), v] for k, v in missed.most_common(20)],
              "top_fired": fired.most_common(25),
              "samples": samples,
          }
          c = report["corpora"][f"{ds}/{gate}"]
          print(f"{ds:9s} {gate:5s} docs {c['n_docs']:5d}  gold {gn:6d}  s1 {ns1:7d}  "
                f"char-recall {c['char_recall']}  span-recall "
                f"{c['span_recall_any_overlap']}  vol x{c['volume_ratio']}  "
                f"midword {c['midword_rate']}", flush=True)
          print("           per-tag:", dict(c["per_tag_recall"]), flush=True)

    tmp = args.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(report, fh, indent=1)
    os.replace(tmp, args.out)
    print("->", args.out)


if __name__ == "__main__":
    main()
