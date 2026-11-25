"""OpenAI-transport probe: same interface as the frozen `probe.Probe`, but
speaking /v1/chat/completions with an Authorization header.

WHY THIS IS A SEPARATE FILE. `src/relaware/probe.py` is inside the SACC-Core
v1 manifest and is not modified. This subclass overrides only the two methods
that touch the wire; `p_of`, the prompts, the renormalisation rule and the
counters are inherited from the frozen class, so the estimator arithmetic is
the frozen one.

WHAT IS NOT IDENTICAL, and must be stated wherever these numbers appear:
  * the frozen probe posts a raw prompt to /v1/completions; gpt-4o-mini does
    not exist on that endpoint, so the same prompt is sent as a single user
    message to /v1/chat/completions;
  * chat logprobs arrive as a LIST of {token, logprob}, not the legacy dict.
Absolute scores from this path are therefore not comparable to the four
frozen families. Only paired differences within this path are, and only
against priors built through this same path.

The key is read from spia/.env and never logged.
"""
import hashlib
import json
import math
import os
import sqlite3
import threading
import time
import urllib.request

from . import probe

LETTERS = probe.LETTERS
_LOCK = threading.Lock()


class OpenAIProbe(probe.Probe):
    def __init__(self, model="gpt-4o-mini", cache=None, timeout=120,
                 top_logprobs=20, **kw):
        super().__init__(base_url="https://api.openai.com", model=model,
                         timeout=timeout, top_logprobs=top_logprobs)
        self.key = self._read_key()
        self.db = None
        if cache:
            os.makedirs(os.path.dirname(cache), exist_ok=True)
            self.db = sqlite3.connect(cache, check_same_thread=False)
            self.db.execute("CREATE TABLE IF NOT EXISTS kv"
                            "(k TEXT PRIMARY KEY, v TEXT)")
            self.db.commit()

    @staticmethod
    def _read_key():
        here = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        for line in open(os.path.join(here, ".env")):
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
        raise SystemExit("OPENAI_API_KEY not found in spia/.env")

    # ---------------------------------------------------------------- http
    def _post(self, path, payload):
        k = hashlib.sha256((path + json.dumps(payload, sort_keys=True))
                           .encode()).hexdigest()
        if self.db is not None:
            with _LOCK:
                row = self.db.execute("SELECT v FROM kv WHERE k=?",
                                      (k,)).fetchone()
            if row:
                self.stats["requests"] += 1
                return json.loads(row[0])
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url + path, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.key})
        last = None
        for attempt in range(5):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                self.stats["requests"] += 1
                self.stats["latency_s"] += time.time() - t0
                u = out.get("usage", {})
                self.stats["prompt_tokens"] += u.get("prompt_tokens", 0)
                self.stats["completion_tokens"] += u.get("completion_tokens", 0)
                if self.db is not None:
                    with _LOCK:
                        self.db.execute("INSERT OR REPLACE INTO kv VALUES(?,?)",
                                        (k, json.dumps(out)))
                        self.db.commit()
                return out
            except Exception as e:                     # 429/5xx -> back off
                last = e
                self.stats["retries"] += 1
                time.sleep(min(2 ** attempt, 20))
        raise last

    # ------------------------------------------------------- logprob mode
    def letter_dist(self, prompt, n_options):
        """Frozen renormalisation, chat-completions transport."""
        out = self._post("/v1/chat/completions", {
            "model": self.model, "temperature": 0.0, "max_tokens": 1,
            "logprobs": True, "top_logprobs": self.top,
            "messages": [{"role": "user", "content": prompt}]})
        ch = out["choices"][0]
        content = (ch.get("logprobs") or {}).get("content") or []
        tops = content[0].get("top_logprobs", []) if content else []
        best = {}
        for e in tops:
            t = str(e.get("token", "")).strip()
            if len(t) == 1 and t in LETTERS[:n_options]:
                i = LETTERS.index(t)
                best[i] = max(best.get(i, -math.inf), e["logprob"])
        if not best:
            t = (ch.get("message", {}).get("content") or "").strip()[:1]
            if t in LETTERS[:n_options]:
                best[LETTERS.index(t)] = 0.0
            else:
                self.stats["parse_failures"] += 1
                return None
        z = sum(math.exp(v) for v in best.values())
        return {i: math.exp(v) / z for i, v in best.items()}

    # ----------------------------------------------------------- json mode
    def _chat_json(self, system, user, max_tokens):
        out = self._post("/v1/chat/completions", {
            "model": self.model, "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]})
        ch = out["choices"][0]
        if ch.get("finish_reason") == "length":
            self.stats["truncations"] += 1
        return ch["message"]["content"]
