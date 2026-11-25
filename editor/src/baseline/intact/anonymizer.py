"""
INTACT — INference-guided Truthful sAnitization for Clear Text.

Pilan, Manzanares-Salor, Sanchez and Lison, "Truthful Text Sanitization Guided
by Inference Attacks" (arXiv:2412.12928); reference implementation at
https://github.com/IldikoPilan/text_sanitization.

Unlike the other SPIA baselines, INTACT does not delete or redact: it replaces
each disclosive span with a *truthful generalisation* — a broader term that
still subsumes the original ("Catholic" -> "Monotheist"). Two stages:

  1. generate m candidates per span, sorted most-specific to most-generic;
  2. pick the most specific candidate an LLM adversary cannot guess back from
     the candidate plus the rest of the (already partly sanitised) document.

The paper runs both stages on a local Mistral-7B-Instruct; here they go through
the repository's provider abstraction so any backbone can be used, and the
candidate list is evaluated lazily — the search stops at the first safe
candidate, which is what Algorithm 1 selects anyway and costs far fewer calls.
"""
import re
from typing import Dict, List, Optional

from src.baseline.intact import heuristics, matching
from src.baseline.intact.prompts import (examples, messages,
                                         messages_reidentif_1shot)

DEFAULT_M_CANDIDATES = 5
DEFAULT_P_GUESSES = 5


def _render(turns: List[Dict[str, str]]) -> str:
    """Flatten the multi-turn one-shot prompt into a single string.

    The reference implementation applies Mistral's chat template; the providers
    here take one user message, so the turns are laid out explicitly with role
    markers, preserving the example/answer structure the authors rely on.
    """
    parts = []
    for t in turns:
        tag = {"user": "User", "assistant": "Assistant"}[t["role"]]
        parts.append(f"{tag}: {t['content']}")
    return "\n\n".join(parts)


def _parse_list(response: str, limit: int) -> List[str]:
    """Read back the hyphen-prefixed list the prompts ask for."""
    out = []
    for line in (response or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^[-*•]\s*(.+)$", line) or re.match(r"^\d+[.)]\s*(.+)$", line)
        if m:
            cand = m.group(1).strip().strip('"').strip()
            cand = re.sub(r"\s*\(.*?most (specific|generic).*?\)\s*$", "", cand, flags=re.I)
            if cand:
                out.append(cand)
    if not out:                      # model ignored the format; salvage lines
        out = [l.strip(' -"') for l in (response or "").splitlines() if l.strip()]
    return out[:limit]


class IntactAnonymizer:
    def __init__(self, call_llm, allowed_words, m_candidates=DEFAULT_M_CANDIDATES,
                 p_guesses=DEFAULT_P_GUESSES, temperature=0.3, max_tokens=512,
                 ngram_size=4, verbose=False, fix_article=False):
        """
        Args:
            call_llm: callable(prompt, temperature, max_tokens) -> str
            allowed_words: frequent lemmas that may match without counting as a leak
        """
        self.call_llm = call_llm
        self.allowed = allowed_words
        self.m = m_candidates
        self.p = p_guesses
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ngram_size = ngram_size
        self.verbose = verbose
        self.fix_article = fix_article
        self.stats = {"llm_gen": 0, "llm_guess": 0, "fallback": 0,
                      "heuristic": 0, "safe_at_rank": []}

    # ---------------- stage 1: candidate generation ------------------------
    def generate_candidates(self, span: str, sentence: str, ent_type: str) -> List[str]:
        ex = examples.get(ent_type) or examples["MISC"]
        ex_out = "\n".join(f"- {r}" for r in ex["out"])
        turns = [
            {"role": "user",
             "content": messages[0]["content"].format(ex["in"], ex["in"].split("[[")[1].split("]]")[0])},
            {"role": "assistant", "content": messages[1]["content"].format(ex_out)},
            {"role": "user", "content": messages[2]["content"].format(sentence, span)},
        ]
        resp = self.call_llm(_render(turns), self.temperature, self.max_tokens)
        self.stats["llm_gen"] += 1
        cands = _parse_list(resp, self.m)
        # drop candidates that simply restate the original
        return [c for c in cands if c.strip().lower() != span.strip().lower()]

    # ---------------- stage 2: inference attack ----------------------------
    def guess(self, bracketed_doc: str, replacement: str) -> List[str]:
        turns = list(messages_reidentif_1shot[:2]) + [
            {"role": "user",
             "content": messages_reidentif_1shot[2]["content"].format(bracketed_doc, replacement)}]
        resp = self.call_llm(_render(turns), self.temperature, self.max_tokens)
        self.stats["llm_guess"] += 1
        return _parse_list(resp, self.p)

    # ---------------- full document ----------------------------------------
    # A span that is a bare name or organisation stands alone in the sentence,
    # so a replacement noun phrase carrying its own article reads correctly:
    # "John Smith" -> "The person with a two-word name". A span that IS the
    # subject's noun phrase does not: the source already supplies the article,
    # and the result is "A an elderly female patient". On the three corpora
    # whose spans come from a TAB-trained detector this fires on 0-1.4 % of
    # edited documents; on MedQA's gold spans it fires on 48 %.
    #
    # Opt-in, so the frozen arms' code path is byte-identical by default.
    _ART_TAIL = re.compile(r"(?:^|(?<=[\s(]))([Aa]n?|[Tt]he)\s+$")
    _ART_HEAD = re.compile(r"^(an?|the)\s+(?=\S)", re.I)

    @classmethod
    def _drop_duplicate_article(cls, prefix: str, repl: str):
        """Return (prefix, repl) with one of the two articles removed."""
        m = cls._ART_TAIL.search(prefix)
        if not m or not cls._ART_HEAD.match(repl):
            return prefix, repl
        # keep the replacement's article -- it agrees with its own head noun,
        # which the source's does not ("A" before "elderly" would be wrong)
        prefix = prefix[:m.start(1)]
        if m.group(1)[0].isupper():
            repl = repl[0].upper() + repl[1:]
        return prefix, repl

    def sanitize(self, text: str, entities: List[dict]) -> str:
        """Sanitize `text` given detected spans (SPIA entity-annotation format)."""
        ents = [e for e in entities
                if e.get("start_offset") is not None and e.get("end_offset") is not None]
        ents.sort(key=lambda e: (e["start_offset"], e["end_offset"]))
        # drop spans nested inside an earlier one; offsets must stay disjoint
        kept, last_end = [], -1
        for e in ents:
            if e["start_offset"] >= last_end:
                kept.append(e)
                last_end = e["end_offset"]
        ents = kept
        if not ents:
            return text

        counter = heuristics.LabelCounter()
        nlp = matching.nlp()
        doc = nlp(text)
        sent_of = {}
        for s in doc.sents:
            sent_of[(s.start_char, s.end_char)] = s.text

        def sentence_for(e):
            for (a, b), s in sent_of.items():
                if a <= e["start_offset"] < b:
                    return s
            return text[max(0, e["start_offset"] - 200):e["end_offset"] + 200]

        # candidates per *entity* (not per mention) so repeats stay consistent
        cand_by_entity: Dict[str, List[str]] = {}
        chosen: Dict[str, str] = {}
        for e in ents:
            key = str(e.get("entity_id") or e["span_text"].lower())
            et = e.get("entity_type", "MISC")
            if key in cand_by_entity:
                continue
            if et in heuristics.DIRECT_LABELS:
                chosen[key] = counter.label_for(et, key)
                cand_by_entity[key] = [chosen[key]]
                self.stats["heuristic"] += 1
            elif et == "DATETIME":
                ladder = heuristics.date_ladder(e["span_text"])
                cand_by_entity[key] = ladder or [et]
                self.stats["heuristic"] += 1
            else:
                cand_by_entity[key] = self.generate_candidates(
                    e["span_text"], sentence_for(e), et) or []

        def render(sel: Dict[str, str], bracket_key: Optional[str] = None) -> str:
            """Rebuild the document, optionally bracketing one entity's mention."""
            out, cur = [], 0
            for e in ents:
                key = str(e.get("entity_id") or e["span_text"].lower())
                repl = sel.get(key)
                if repl is None:
                    cands = cand_by_entity.get(key) or []
                    repl = cands[0] if cands else e.get("entity_type", "MISC")
                chunk = text[cur:e["start_offset"]]
                if self.fix_article:
                    chunk, repl = self._drop_duplicate_article(chunk, repl)
                out.append(chunk)
                out.append(f"[[{repl}]]" if key == bracket_key else repl)
                cur = e["end_offset"]
            out.append(text[cur:])
            return "".join(out)

        # Step 2 of Algorithm 1: walk the candidate list, keep the first safe one
        for e in ents:
            key = str(e.get("entity_id") or e["span_text"].lower())
            if key in chosen:
                continue
            et = e.get("entity_type", "MISC")
            cands = cand_by_entity.get(key) or []
            picked = None
            for rank, cand in enumerate(cands):
                d_prime = render(chosen, bracket_key=key)
                guesses = self.guess(d_prime, cand)
                hit, why = matching.is_guessed(guesses, e["span_text"], et,
                                               self.allowed, self.ngram_size)
                if self.verbose:
                    print(f"    [{et}] {e['span_text']!r} <- {cand!r} "
                          f"{'GUESSED (' + str(why) + ')' if hit else 'safe'}")
                if not hit:
                    picked = cand
                    self.stats["safe_at_rank"].append(rank + 1)
                    break
            if picked is None:                      # every candidate was risky
                picked = counter.label_for(et, key)
                self.stats["fallback"] += 1
            chosen[key] = picked

        return render(chosen)
