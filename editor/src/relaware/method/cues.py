"""Cue detection and relation-linked cue groups (plan §12.2-12.3).

Phase 1 runs the ORACLE variant: subject identity and protected propositions
come from the gold registry (which subjects exist and which attributes are
protected), because Phase 1's question is whether JOINT SELECTION beats
independent editing when detection is not the bottleneck. Automatic detection
is Phase 2, and Detection Coverage (§6.3) is measured there.

What stays method-side and public even in oracle mode: the pronoun sets, the
kinship and occupation lexicons, and the linking heuristic. The oracle only
supplies WHO and WHICH attribute -- never the evaluator's candidate sets,
priors, or frozen option orders.

A cue is (span_start, span_end, surface, kind). A cue GROUP is every cue in
the document that supports the same protected proposition (subject, attribute)
-- pronouns and titles for SEX, kinship phrases for RELATIONSHIP, role and
work phrases for OCCUPATION.
"""
import re

from ..profiles import PROFILES

_DEFAULT = PROFILES["tab"]
# module-level names kept for callers that predate profiles
PRONOUN_PAT = _DEFAULT.lex["pronoun"]
GENDERED_KIN = _DEFAULT.lex["gendered_kin"]
KINSHIP = _DEFAULT.lex["kinship"]
OCC_ROLE = _DEFAULT.lex["occ_role"]


# Splitting at every ".!?" would cut after "Mr." and strand the name in a
# fragment with no cue context; the lookbehinds keep the common title and
# citation abbreviations glued to what follows. Pronoun-only sentences (no
# surface form at all) still yield no cues here -- coreference is Phase 2's
# job, and Detection Coverage (§6.3) is where that gap gets measured.
_SPLIT = re.compile(r"(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)"
                    r"(?<!\bNo\.)(?<!\bpp\.)(?<!\bv\.)(?<=[.!?])\s+")


def _sentences(text):
    out, pos = [], 0
    for m in _SPLIT.finditer(text):
        out.append((pos, m.start()))
        pos = m.end()
    out.append((pos, len(text)))
    return out


_TURN = re.compile(r"(?m)^([^\n:]{1,40}?)\s*:\s")


def _own_turns(text, surface_forms):
    """Spans of the speaker turns belonging to this subject: from their
    speaker prefix to the next speaker prefix. Fuzzy prefix match, because
    the gold head is "jacob_m" where the text says "Name: Jacob_M"."""
    turns = [(m.group(1).strip().lower(), m.end()) for m in _TURN.finditer(text)]
    turns.append(("", len(text)))
    spans = []
    forms = [f.lower() for f in surface_forms if len(f) > 2]
    for (spk, a), (_, b) in zip(turns, turns[1:]):
        if any(f == spk or f in spk or spk in f for f in forms if spk):
            spans.append((a, b))
    return spans


def sentences_about(text, surface_forms, profile=_DEFAULT):
    """Sentence spans in which the subject is discussed.

    First-person corpora take every sentence: the subject IS the author, is
    never named, and writes "I". Gating those on a surface form yields zero
    cues on all 281 SynthPAI documents -- the pipeline then completes and
    reports four identical method arms."""
    if profile.subject_is_author:
        return _sentences(text)
    spans = []
    for a, b in _sentences(text):
        low = text[a:b].lower()
        if any(f.lower() in low for f in surface_forms if len(f) > 3):
            spans.append((a, b))
    if getattr(profile, "turn_scope", False):
        # own speaker turns, merged with the mention sentences and deduped
        merged = sorted(set(spans) | set(_own_turns(text, surface_forms)))
        out = []
        for a, b in merged:
            if out and a < out[-1][1]:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out
    return spans


def _find_all(text, a, b, pattern, kind):
    return [{"start": a + m.start(), "end": a + m.end(),
             "surface": m.group(0), "kind": kind}
            for m in re.finditer(pattern, text[a:b], re.I)]


def detect_groups(text, subject_forms, protected, profile=_DEFAULT):
    """Build cue groups for one subject.

    subject_forms: the subject's surface forms (oracle-supplied in Phase 1).
    protected: iterable of attributes to protect, e.g. {"SEX", "OCCUPATION"}.
    Returns {attribute: [cue, ...]} -- one relation-linked group per
    proposition (§12.3). Mention cues (names/handles) are returned under
    "MENTION" so the editor can pseudonymize instead of delete (guardrail:
    Subject Removal)."""
    groups = {a: [] for a in protected}
    groups["MENTION"] = []
    lex = profile.lex
    about = sentences_about(text, subject_forms, profile)

    if not profile.subject_is_author:
        for f in subject_forms:
            if len(f) <= 3:
                continue
            for m in re.finditer(re.escape(f), text, re.I):
                groups["MENTION"].append({
                    "start": m.start(), "end": m.end(),
                    "surface": m.group(0), "kind": "mention"})
    for a, b in about:
        if "SEX" in protected:
            # three sub-patterns: pronouns need \b on both sides; titles take
            # an optional dot and are anchored by the following space instead
            # (a trailing \b after "." can never match before whitespace);
            # gendered kinship nouns are SEX cues too -- §12.3 lists
            # "husband" under SEX -- so a SEX-protected subject whose
            # RELATIONSHIP is not protected still gets them neutralized
            groups["SEX"] += _find_all(text, a, b, lex["pronoun"], "pronoun")
            groups["SEX"] += _find_all(text, a, b, lex["title"], "title")
            groups["SEX"] += _find_all(text, a, b, lex["gendered_kin"],
                                       "gendered_kin")
        if "RELATIONSHIP" in protected:
            groups["RELATIONSHIP"] += _find_all(text, a, b, lex["kinship"],
                                                "kinship")
        if "OCCUPATION" in protected:
            groups["OCCUPATION"] += _find_all(text, a, b, lex["occ_role"],
                                              "role")
    # dedupe overlapping cues within each group, keep the longest
    for k, cues in groups.items():
        cues.sort(key=lambda c: (c["start"], -(c["end"] - c["start"])))
        kept, last_end = [], -1
        for c in cues:
            if c["start"] >= last_end:
                kept.append(c)
                last_end = c["end"]
        groups[k] = kept
    return groups
