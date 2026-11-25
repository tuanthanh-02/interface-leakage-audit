"""The MCQ probe: prior and posterior over frozen candidate sets.

Two estimators.

`logprob` (primary, local vLLM): the options are lettered in the prompt, the
model answers with one letter at max_tokens=1, and the distribution is read
off the top-logprobs at that single position, renormalised over the option
letters. One request per (item, permutation); requests are cheap and cached
against the shared document prefix on a local server. This is the estimator
the archived p_gold probe validated.

`json` (batchable, for API evaluators and the §17.4 batch pilot): B items per
request, structured JSON out, one letter per item. Probability is then the
frequency of each option over the N_PERM permutations -- coarser than logprob,
which is exactly what the pilot's score-drift criterion (|Δ| <= .02, §17.5)
is there to measure. Failure ladder per §17.6: one retry at B/2, then
item-level fallback, then a recorded EVALUATION_FAILURE -- never a silent drop.

Prior (§7): canonical handle + domain + candidates, no text.
Posterior: text + output handle + the same candidates.
"""
import json
import math
import re
import string
import time
import urllib.request

LETTERS = string.ascii_uppercase


def prior_prompt(handle, domain, attr_q, options):
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (f'Consider "{handle}", a person involved in {domain}. '
            f"No further information is available.\n"
            f"Question: What is their {attr_q}?\n{lines}\n"
            f"Answer with a single letter.\nAnswer:")


def posterior_prompt(text, handle, attr_q, options):
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return (f"Text:\n{text}\n\n"
            f'Question: Considering the person referred to as "{handle}" in '
            f"the text, what is their {attr_q}?\n{lines}\n"
            f"Answer with a single letter.\nAnswer:")


ATTR_Q = {"SEX": "sex", "RELATIONSHIP": "relationship status",
          "OCCUPATION": "occupation"}


class Probe:
    """Counters follow plan §17: every request, retry and failure is recorded."""

    def __init__(self, base_url="http://localhost:8000", model="",
                 timeout=240, top_logprobs=20):
        self.url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.top = top_logprobs
        # parse_failures: the reply was not the promised JSON at all.
        # item_schema_failures: the JSON parsed but an item was missing or
        # carried an out-of-range letter -- the §17.4 "schema validation
        # failure", which is a different quantity from parse success and is
        # counted separately so the two §17.5 criteria stay two criteria.
        self.stats = {"requests": 0, "retries": 0, "split_fallbacks": 0,
                      "item_fallbacks": 0, "failures": 0,
                      "prompt_tokens": 0, "completion_tokens": 0,
                      "truncations": 0, "parse_failures": 0,
                      "item_schema_failures": 0, "partial_items": 0,
                      "latency_s": 0.0}

    # ---------------------------------------------------------------- http
    def _post(self, path, payload):
        t0 = time.time()
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        self.stats["requests"] += 1
        self.stats["latency_s"] += time.time() - t0
        u = out.get("usage", {})
        self.stats["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.stats["completion_tokens"] += u.get("completion_tokens", 0)
        return out

    # ------------------------------------------------------- logprob mode
    def letter_dist(self, prompt, n_options):
        out = self._post("/v1/completions", {
            "model": self.model, "prompt": prompt, "max_tokens": 1,
            "temperature": 0.0, "logprobs": self.top})
        lp = out["choices"][0].get("logprobs") or {}
        tops = (lp.get("top_logprobs") or [{}])[0] or {}
        best = {}
        for tok, val in tops.items():
            t = tok.strip()
            if len(t) == 1 and t in LETTERS[:n_options]:
                i = LETTERS.index(t)
                best[i] = max(best.get(i, -math.inf), val)
        if not best:
            # the answer token was outside top-k: fall back to the sampled one
            t = (out["choices"][0].get("text") or "").strip()[:1]
            if t in LETTERS[:n_options]:
                best[LETTERS.index(t)] = 0.0
            else:
                self.stats["parse_failures"] += 1
                return None
        z = sum(math.exp(v) for v in best.values())
        return {i: math.exp(v) / z for i, v in best.items()}

    def p_of(self, prompt_fn, options, gold, perms):
        """Mean probability of `gold` over the given permutations.
        prompt_fn(perm_options) -> prompt. Returns (p, per_perm, failed)."""
        ps = []
        for idx in perms:
            opts = [options[i] for i in idx]
            dist = self.letter_dist(prompt_fn(opts), len(opts))
            if dist is None:
                continue
            ps.append(dist.get(opts.index(gold) if gold in opts
                               else idx.index(options.index(gold)), 0.0))
        if not ps:
            self.stats["failures"] += 1
            return None, [], True
        if len(ps) < len(perms):
            # scored on fewer than the frozen number of candidate orders --
            # §18 Tier A/D requires all of them, so this is recorded, not
            # silently averaged away
            self.stats["partial_items"] += 1
        return sum(ps) / len(ps), ps, False

    # ----------------------------------------------------------- json mode
    def _chat_json(self, system, user, max_tokens):
        out = self._post("/v1/chat/completions", {
            "model": self.model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]})
        ch = out["choices"][0]
        if ch.get("finish_reason") == "length":
            self.stats["truncations"] += 1
        return ch["message"]["content"]

    def batched_letters(self, items, batch_size):
        """items: list of {"id", "prompt_body", "n_options"}. Returns
        {id: letter_index or None (EVALUATION_FAILURE)}.

        Exactly the §17.6 ladder, three rungs and no more: the failing batch
        is rerun ONCE at B/2, any still-failing half falls to item-level, and
        an item that fails there is recorded -- never silently dropped."""
        answers = {}

        def item_level(its):
            for it in its:
                self.stats["item_fallbacks"] += 1
                got = self._try_batch([it])
                answers[it["id"]] = (got.get(str(it["id"])) if got else None)
                if got is None:
                    self.stats["failures"] += 1

        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            got = self._try_batch(batch)
            if got is not None:
                for it in batch:
                    answers[it["id"]] = got.get(str(it["id"]))
                continue
            if len(batch) == 1:
                item_level(batch)
                continue
            self.stats["retries"] += 1                  # rung 2: once at B/2
            half = max(1, len(batch) // 2)
            for part in (batch[:half], batch[half:]):
                if not part:
                    continue
                self.stats["split_fallbacks"] += 1
                got = self._try_batch(part)
                if got is not None:
                    for it in part:
                        answers[it["id"]] = got.get(str(it["id"]))
                else:                                    # rung 3: item-level
                    item_level(part)
        return answers

    def _try_batch(self, batch):
        body = "\n\n".join(f'[item {it["id"]}]\n{it["prompt_body"]}'
                           for it in batch)
        sys_p = ("Answer every item with a single option letter. Reply with "
                 "ONLY a JSON object mapping item id to letter, e.g. "
                 '{"3": "B", "7": "A"}. No other text.')
        try:
            raw = self._chat_json(sys_p, body, max_tokens=16 * len(batch) + 32)
        except Exception:
            return None
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            self.stats["parse_failures"] += 1
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            self.stats["parse_failures"] += 1
            return None
        out = {}
        for it in batch:
            v = str(obj.get(str(it["id"]), "")).strip().upper()[:1]
            if v and v in LETTERS[:it["n_options"]]:
                out[str(it["id"])] = LETTERS.index(v)
            else:
                # JSON parsed but this item is missing or invalid: the §17.4
                # schema-validation failure, counted as itself
                self.stats["item_schema_failures"] += 1
                return None
        return out
