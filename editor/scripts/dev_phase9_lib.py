"""Phase 9 shared machinery: the audit-reference rule (V1), the logged probe
(V2), attribute-cue lexicons, and small helpers. Nothing frozen is modified;
`LoggedProbe` subclasses `src.relaware.probe.Probe` and calls the frozen
arithmetic through `super()`.
"""
import hashlib, json, os, re, sys, threading, time
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from src.relaware import probe as _probe, profiles as _profiles

# --------------------------------------------------------------- lexicons
# Union of the frozen profile lexicons (THIRD, FIRST, MedQA extension) plus the
# generic demographic nouns, so "carries a protected-attribute cue" is judged
# by the same vocabulary the detector uses, widened rather than narrowed.
_T, _F, _M = _profiles.THIRD, _profiles.FIRST, _profiles.PROFILES["medqa"].lex
SEX_CUE = re.compile("|".join([
    _T["pronoun"], _T["title"], _T["gendered_kin"], _M["pronoun"],
    r"\b(man|woman|men|women|male|female|boy|girl|gentleman|lady|guy|dude|mom|mum|dad|"
    r"husband|wife|boyfriend|girlfriend|widow|widower|mr|mrs|ms|miss)\b\.?"]), re.I)
OCC_CUE = re.compile("|".join([
    _T["occ_role"], _M["occ_role"], _F["occ_role"],
    r"\b(agent|chairman|chairwoman|resident|intern|clerk|staff|employee|worker|"
    r"official|servant|lawyer|attorney|counsel|professor|student|nurse|doctor|physician|"
    r"surgeon|technician|therapist|analyst|designer|developer|manager|consultant|"
    r"scientist|researcher|artist|writer|chef|driver|pilot|police|soldier|officer|"
    r"judge|prosecutor|patient)\b"]), re.I)
REL_CUE = re.compile("|".join([
    _T["kinship"], _F["kinship"],
    r"\b(married|divorced|single|widowed|engaged|spouse|partner|couple|wedding|"
    r"fianc\w+|boyfriend|girlfriend|ex-wife|ex-husband)\b"]), re.I)
CUES = {"SEX": SEX_CUE, "OCCUPATION": OCC_CUE, "RELATIONSHIP": REL_CUE}
_STOP = {"the", "a", "an", "of", "and", "with", "in", "on", "at", "to", "for", "who", "is",
         "was", "from", "by", "s", "person", "individual", "unnamed"}


def attribute_cues_in(s):
    """Which protected attributes a string carries a lexical cue for."""
    return sorted(a for a, rx in CUES.items() if rx.search(s or ""))


def neutralize(form):
    """Strip attribute-bearing tokens from a mention form. SEX nouns and
    pronouns become 'person'; titles, occupation and relationship words are
    dropped. Returns '' when no content token survives."""
    s = form
    s = re.sub(_T["title"], "", s)
    s = re.sub(r"\b(Mr|Mrs|Ms|Miss|Dr)\.?\s*", "", s)
    s = SEX_CUE.sub("person", s)
    s = OCC_CUE.sub("", s)
    s = REL_CUE.sub("", s)
    s = re.sub(r"\bperson(\s+person)+\b", "person", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:-'\"()")
    toks = [t for t in re.findall(r"[A-Za-z0-9][\w'-]*", s) if t.lower() not in _STOP]
    return s if toks else ""


def _content_tokens(s):
    return [t for t in re.findall(r"[A-Za-z0-9][\w-]*", s) if t.lower() not in _STOP and len(t) > 2]


def audit_reference(corpus, gold_forms, arm_text, n_subjects_in_doc):
    """The Phase 9 audit-reference rule (runs/dev/phase9/audit_reference_rules.md).
    Returns dict(reference, status, source_form, verbatim, cue_free)."""
    if corpus == "synthpai":
        return {"reference": "the author of the post", "status": "IDENTIFIABLE_BY_REGISTER",
                "source_form": None, "verbatim": True, "cue_free": True}
    # R1: a verbatim, cue-free gold form
    cands = [f for f in gold_forms if f and len(f) > 3 and f in arm_text and not attribute_cues_in(f)]
    if cands:
        f = max(cands, key=len)
        return {"reference": f'the person referred to as "{f}"', "status": "IDENTIFIABLE_BY_STRING",
                "source_form": f, "verbatim": True, "cue_free": True}
    # R2: a neutralized descriptor whose content tokens still occur in the text
    for f in gold_forms:
        n = neutralize(f)
        if not n:
            continue
        toks = _content_tokens(n)
        if toks and any(re.search(r"\b" + re.escape(t) + r"\b", arm_text, re.I) for t in toks):
            return {"reference": f'the person the text introduces as "{n}"', "status": "IDENTIFIABLE_BY_NEUTRALIZED_DESCRIPTOR",
                    "source_form": f, "verbatim": False, "cue_free": True}
    # R3: single protected subject -> default; else not identifiable by rule
    if n_subjects_in_doc == 1:
        return {"reference": "the person this text is mainly about", "status": "SINGLE_SUBJECT_DEFAULT",
                "source_form": None, "verbatim": False, "cue_free": True}
    return {"reference": "a person in this text whom the audit cannot name without revealing a protected attribute",
            "status": "SUBJECT_NOT_IDENTIFIABLE_BY_RULE", "source_form": None, "verbatim": False, "cue_free": True}


# --------------------------------------------------------------- logged probe
def task_type_of(prompt_or_user):
    s = prompt_or_user or ""
    if s.startswith('Consider "'):
        return "prior"
    if s.startswith("Text:\n"):
        return "posterior_source_aware_or_bb"
    if s.startswith("You are aligning people"):
        return "aligner_source_aware"
    if s.startswith("Read the following text and list every distinct person"):
        return "black_box_inventory"
    return "other"


class LoggedProbe(_probe.Probe):
    """Frozen arithmetic (`super().letter_dist`, `super()._chat_json`) with a
    per-request audit line. The raw HTTP response is captured by wrapping
    `_post`; the audit fields are computed from the SAME raw response the
    frozen code parsed, so nothing about the distribution changes."""

    def __init__(self, *a, log_path=None, **k):
        super().__init__(*a, **k)
        self._log_path = log_path
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._n = 0
        self.audit = {"fallback_used": 0, "missing_options_nonzero": 0, "parse_failure": 0,
                      "chat_truncated": {}, "requests_by_task": {}}
        if log_path:
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def _post(self, path, payload):
        out = super()._post(path, payload)
        self._tls.last = (payload, out)
        return out

    def _write(self, rec):
        with self._lock:
            self._n += 1
            rec["request_id"] = self._n
            self.audit["requests_by_task"][rec["task_type"]] = self.audit["requests_by_task"].get(rec["task_type"], 0) + 1
            if self._log_path:
                with open(self._log_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")

    def letter_dist(self, prompt, n_options):
        dist = super().letter_dist(prompt, n_options)
        payload, out = self._tls.last
        ch = out["choices"][0]
        tops = ((ch.get("logprobs") or {}).get("top_logprobs") or [{}])[0] or {}
        present = {t.strip() for t in tops if len(t.strip()) == 1 and t.strip() in _probe.LETTERS[:n_options]}
        sampled = (ch.get("text") or "").strip()[:1]
        fallback = (not present) and dist is not None
        rec = {"task_type": task_type_of(prompt), "finish_reason": ch.get("finish_reason"),
               "candidate_count": n_options, "options_in_topk": len(present),
               "missing_option_count": n_options - len(present),
               "fallback_used": bool(fallback), "sampled_token": sampled if fallback else None,
               "failure_code": None if dist is not None else "PARSE_FAILURE_NO_OPTION_LETTER",
               "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16]}
        if fallback: self.audit["fallback_used"] += 1
        if rec["missing_option_count"] and dist is not None: self.audit["missing_options_nonzero"] += 1
        if dist is None: self.audit["parse_failure"] += 1
        self._write(rec)
        return dist

    def _chat_json(self, system, user, max_tokens):
        txt = super()._chat_json(system, user, max_tokens)
        payload, out = self._tls.last
        ch = out["choices"][0]
        tt = task_type_of(user)
        rec = {"task_type": tt, "finish_reason": ch.get("finish_reason"), "max_tokens": max_tokens,
               "candidate_count": None, "missing_option_count": None, "fallback_used": None,
               "failure_code": "TRUNCATED" if ch.get("finish_reason") == "length" else None,
               "prompt_sha256_16": hashlib.sha256((system + "\n" + user).encode()).hexdigest()[:16]}
        if ch.get("finish_reason") == "length":
            self.audit["chat_truncated"][tt] = self.audit["chat_truncated"].get(tt, 0) + 1
        self._write(rec)
        return txt


def sha16(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# --------------------------------------------------------------- regions
def regions(source, edited, ctx=120, merge_gap_tokens=2):
    """Token-level changed regions between `source` and `edited` (difflib on
    whitespace-delimited tokens, adjacent changes merged), each with its
    character span in both strings and a context window in the edited text.
    Token-level so a pronoun swap is one region, not three character slivers."""
    import difflib
    def toks(s):
        out, pos = [], 0
        for m in re.finditer(r"\S+|\s+", s):
            out.append((m.group(0), m.start(), m.end()))
        return out
    A, B = toks(source), toks(edited)
    sm = difflib.SequenceMatcher(None, [t for t, _, _ in A], [t for t, _, _ in B], autojunk=False)
    raw = [(a0, a1, b0, b1) for op, a0, a1, b0, b1 in sm.get_opcodes() if op != "equal"]
    merged = []
    for a0, a1, b0, b1 in raw:
        if merged and b0 - merged[-1][3] <= merge_gap_tokens and a0 - merged[-1][1] <= merge_gap_tokens:
            m = merged[-1]; merged[-1] = (m[0], a1, m[2], b1)
        else:
            merged.append((a0, a1, b0, b1))
    out = []
    for a0, a1, b0, b1 in merged:
        sa = A[a0][1] if a0 < len(A) else (A[-1][2] if A else 0); ea = A[a1 - 1][2] if a1 > a0 else sa
        sb = B[b0][1] if b0 < len(B) else (B[-1][2] if B else 0); eb = B[b1 - 1][2] if b1 > b0 else sb
        out.append({"src_span": [sa, ea], "src": source[sa:ea], "edt_span": [sb, eb], "edt": edited[sb:eb],
                    "context": edited[max(0, sb - ctx):sb] + "⟦" + edited[sb:eb] + "⟧" + edited[eb:eb + ctx]})
    return out
