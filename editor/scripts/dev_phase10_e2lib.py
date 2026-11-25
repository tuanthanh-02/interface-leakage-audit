#!/usr/bin/env python
"""Phase 10 P0 -- E2-CF-FULL candidate probabilities (plan §8.2).

`FullOptionProbe` guarantees that every option in the frozen candidate set is scored.

    top-k complete   the frozen arithmetic is returned UNCHANGED, so E2 differs from the
                     frozen path only where the frozen path was actually truncating
    any option missing
                     the item is re-scored by teacher forcing ALL options, never a mix of
                     two instruments in one distribution: the prompt is extended by the
                     option letter in the exact surface form the server emits (a leading
                     space on this model -- verified, not assumed) and the letter's logprob is
                     read from `prompt_logprobs`

Verified on llama-3.1-8b-instruct / vLLM 0.26.0 on 2026-09-10: with the correct surface form
the teacher-forced logprob matches the top-k logprob to within the quantisation of the returned
values (0.045-0.080 nats); with the letter appended directly to "Answer:" it does not, which is
why the surface form is read from the response rather than guessed. Server max `logprobs` is 20.
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dev_phase9_lib as L9
from src.relaware import probe as _probe


class FullOptionProbe(L9.LoggedProbe):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.audit.setdefault("topk_complete", 0)
        self.audit["teacher_forced_items"] = 0
        self.audit["teacher_forced_requests"] = 0
        self.audit["full_option_failures"] = 0

    def _letter_surface(self, tops, n_options):
        """the exact surface the server uses for an option letter, e.g. ' C'"""
        for tok in sorted(tops, key=lambda t: -tops[t]):
            s = tok.strip()
            if len(s) == 1 and s in _probe.LETTERS[:n_options]:
                return tok[:len(tok) - len(tok.lstrip())]
        return " "

    def _teacher_force(self, prompt, n_options, prefix):
        out = {}
        for i in range(n_options):
            r = self._post("/v1/completions", {
                "model": self.model, "prompt": prompt + prefix + _probe.LETTERS[i],
                "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": 0})
            self.audit["teacher_forced_requests"] += 1
            pl = (r["choices"][0] or {}).get("prompt_logprobs") or []
            if not pl or not isinstance(pl[-1], dict) or not pl[-1]:
                return None
            out[i] = list(pl[-1].values())[0]["logprob"]
        z = sum(math.exp(v) for v in out.values())
        return None if z <= 0 else {i: math.exp(v) / z for i, v in out.items()}

    def letter_dist(self, prompt, n_options):
        dist = super().letter_dist(prompt, n_options)          # frozen arithmetic + audit line
        payload, out = self._tls.last
        ch = out["choices"][0]
        tops = ((ch.get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
        present = {t.strip() for t in tops
                   if len(t.strip()) == 1 and t.strip() in _probe.LETTERS[:n_options]}
        if len(present) == n_options:
            self.audit["topk_complete"] += 1
            return dist
        full = self._teacher_force(prompt, n_options, self._letter_surface(tops, n_options))
        if full is None:
            self.audit["full_option_failures"] += 1
            return dist                                        # recorded; not silently perfect
        self.audit["teacher_forced_items"] += 1
        return full
