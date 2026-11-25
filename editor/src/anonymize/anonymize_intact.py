"""
INTACT anonymization (Pilan et al., arXiv:2412.12928) over SPIA documents.

INTACT expects the disclosive spans as input -- the paper puts span detection
explicitly out of scope and uses TAB's manual annotations. The same choice is
offered here:

  --spans gold        SPIA's entity annotations (default; matches the paper,
                      but gives INTACT an oracle detector the other baselines
                      do not have -- say so when comparing)
  --spans longformer  spans predicted by the TAB Longformer, i.e. the same
                      detector the Longformer baseline uses

Usage:
    python src/anonymize/anonymize_intact.py -i data/spia/spia_tab_144.jsonl \\
        --api_provider ollama --model llama-3.1-8b-instruct --workers 8
"""
import argparse
import json
import os
import sys

from src.anonymize.utils import run_batch
from src.baseline.intact.anonymizer import IntactAnonymizer
from src.baseline.intact.matching import load_allowed_words

ENTITY_FILE = {
    "tab": "data/entity/entity_tab_144.jsonl",
    "panorama": "data/entity/entity_panorama_151.jsonl",
    # derived from the registry, not shipped with the corpus; identical to the
    # spans the medqa_span_only floor masks. scripts/dev_make_medqa_entity_file.py
    "medqa": "data/entity/entity_medqa_1249.jsonl",
}
FREQ_WORDS = {
    "tab": "src/baseline/intact/freq_words_tab.json",
    "panorama": "src/baseline/intact/freq_words_panorama.json",
    "synthpai": "src/baseline/intact/freq_words_synthpai.json",
    "medqa": "src/baseline/intact/freq_words_medqa.json",
}


def detect_dataset(path):
    p = path.lower()
    # SynthPAI first: it groups one Reddit author's comments, so it takes the
    # social-media handling, and checking it before the others keeps a run tag
    # that happens to contain "tab" from stealing it.
    if "synthpai" in p:
        return "synthpai"
    if "medqa" in p:
        return "medqa"
    if "panorama" in p:
        return "panorama"
    if "tab" in p:
        return "tab"
    raise ValueError(f"Cannot tell dataset from path: {path}. "
                     "Expected 'tab', 'panorama', 'synthpai' or 'medqa'.")


LONGFORMER_RUN = {
    "synthpai": "runs/synthpai_longformer_longformer/anonymized.jsonl",
    "tab": "runs/tab_longformer_longformer/anonymized.jsonl",
    "panorama": "runs/panorama_longformer_longformer/anonymized.jsonl",
    "medqa": "runs/medqa_longformer_longformer/anonymized.jsonl",
}


def load_spans(dataset, spans_source, spans_file=None):
    """Return {data_id: [entity, ...]}."""
    if spans_source == "gold":
        path = spans_file or ENTITY_FILE[dataset]
    else:
        path = spans_file or LONGFORMER_RUN[dataset]
    if not path or not os.path.exists(path):
        raise SystemExit(f"Span file not found: {path}")
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            did = d.get("metadata", {}).get("data_id")
            ents = d.get("entities") or d.get("masked_spans") or []
            if did:
                out[did] = ents
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="INTACT truthful sanitization")
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", default=None)
    ap.add_argument("--api_provider", default="ollama",
                    choices=["openai", "anthropic", "ollama"])
    ap.add_argument("--model", "-m", default="llama-3.1-8b-instruct")
    ap.add_argument("--spans", default="gold", choices=["gold", "longformer"])
    ap.add_argument("--fix-article-collision", action="store_true",
                    help="drop one of two adjacent articles when the source "
                         "already supplies one and the replacement carries its "
                         "own -- 'A an elderly female patient'. A surface "
                         "repair: same cues, same spans, same ladder. Off by "
                         "default so the frozen arms are byte-identical.")
    ap.add_argument("--spans_file", default=None,
                    help="explicit span file (required for --spans longformer)")
    ap.add_argument("--candidates", type=int, default=5, help="m, per the paper")
    ap.add_argument("--guesses", type=int, default=5, help="p, per the paper")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--workers", "-w", type=int, default=1)
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap.parse_args()


def main():
    from src.llm.client import call_api_with_param, set_api_provider

    args = parse_args()
    dataset = detect_dataset(args.input)
    set_api_provider(args.api_provider, args.model)

    spans = load_spans(dataset, args.spans, args.spans_file)
    if dataset not in FREQ_WORDS:
        raise SystemExit(
            f"no frequent-word list for '{dataset}'. Generate one with:\n"
            f"  python scripts/build_freq_words.py --input <gt.jsonl> "
            f"--out src/baseline/intact/freq_words_{dataset}.json")
    allowed = load_allowed_words(FREQ_WORDS[dataset])
    if not allowed:
        print(f"note: no frequent-word list for {dataset}; matching will be strict")

    with open(args.input, encoding="utf-8") as fh:
        json_list = [json.loads(l) for l in fh if l.strip()]
    print(f"Loaded {len(json_list)} documents; spans for {len(spans)} documents "
          f"({args.spans})")

    def call_llm(prompt, temperature, max_tokens):
        return call_api_with_param(prompt, temperature=temperature,
                                   max_tokens=max_tokens)

    totals = {"llm_gen": 0, "llm_guess": 0, "fallback": 0, "heuristic": 0}

    def one(rec):
        did = rec.get("metadata", {}).get("data_id")
        ents = spans.get(did)
        if not ents:
            return rec["text"]              # nothing detected: nothing to change
        an = IntactAnonymizer(call_llm, allowed, args.candidates, args.guesses,
                              args.temperature, args.max_tokens,
                              verbose=args.verbose,
                              fix_article=args.fix_article_collision)
        out = an.sanitize(rec["text"], ents)
        for k in totals:
            totals[k] += an.stats[k]
        return out

    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        outdir = f"data/spia/anonymized/{dataset}"
        os.makedirs(outdir, exist_ok=True)
        args.output = os.path.join(
            outdir, f"{base}_intact_{args.model.replace('/', '_')}.jsonl")

    n_ok, n_err = run_batch(json_list, one, args.output,
                            workers=args.workers, desc="INTACT")
    print(f"Success: {n_ok}, Errors: {n_err}")
    print(f"LLM calls: {totals['llm_gen']} generation + {totals['llm_guess']} guess "
          f"| heuristic spans: {totals['heuristic']} | fallbacks: {totals['fallback']}")
    print("Processing completed!")


if __name__ == "__main__":
    main()
