"""Gold evaluation registry (plan §6.1) and canonical handles (§7).

The registry is the evaluator's private view. It is built once from the gold
file, written to runs/relaware/registry.json, and never shown to any method.
One entry per (subject, attribute) with certainty >= 3 and a non-empty gold
value, for the three core tags.

Canonical handles are `Subject A`, `Subject B`, ... in gold subject order
within each document -- they carry no name, sex, occupation, legal role or
family relationship (§7).
"""
import hashlib
import json
import re
import string

from . import profiles

CORE_TAGS = ("SEX", "RELATIONSHIP", "OCCUPATION")
MIN_CERTAINTY = 3


def _handle(idx):
    letters = string.ascii_uppercase
    return "Subject " + (letters[idx] if idx < 26
                         else letters[idx // 26 - 1] + letters[idx % 26])


def _surface_forms(subject, profile):
    """Public-ish surface forms for a subject: the gold NAME keyword plus any
    role phrase at the head of the description ('The applicant', 'Mrs X - the
    second applicant', ...). Used by the source-aware aligner and by the
    ORACLE cue detector -- both are allowed gold access; the automatic method
    (Phase 2) must never call this.

    First-person corpora have none: SynthPAI's description head is a Reddit
    username that appears in 0 of 281 texts. The first-person pronouns are
    returned instead so the aligner has something to ask about."""
    if profile.subject_is_author:
        return ["I", "me", "my"]
    forms = []
    for p in subject.get("PIIs", []):
        if p["tag"] == "NAME" and p.get("keyword"):
            forms.append(p["keyword"].strip())
    desc = subject.get("description") or ""
    head = desc.split(" - ")[0].strip()
    if head and head.lower() not in {f.lower() for f in forms}:
        forms.append(head)
    m = re.search(r"\b(the (?:\w+ )?applicants?|the accused|the victim|"
                  r"the defendant|the witness)\b", desc, re.I)
    if m:
        forms.append(m.group(1))
    return forms


def build(gold_path, profile=None):
    profile = profile or profiles.for_path(gold_path)
    docs = [json.loads(l) for l in open(gold_path)]
    subjects, entries = [], []
    for doc in docs:
        did = doc["metadata"]["data_id"]
        for i, s in enumerate(doc["subjects"]):
            sid = f"{did}:s{s['id']}"
            subjects.append({
                "eval_subject_id": sid, "doc_id": did,
                "canonical_handle": _handle(i),
                "gold_mention": _surface_forms(s, profile),
                "description": s.get("description", ""),
            })
            for p in s.get("PIIs", []):
                if p["tag"] not in CORE_TAGS:
                    continue
                if p.get("certainty", 0) < MIN_CERTAINTY or not p.get("keyword"):
                    continue
                v = p["keyword"].strip().lower()
                e = {"eval_subject_id": sid, "doc_id": did,
                     "canonical_handle": _handle(i),
                     "attribute": p["tag"], "gold_value": v,
                     "certainty": p["certainty"], "hardness": p.get("hardness"),
                     "policy": "GENERALIZE"}
                if p["tag"] == "OCCUPATION":
                    s4 = profiles.occ_stratum(v)
                    e["stratum"] = {lvl: (fn(s4) if s4 else None)
                                    for lvl, fn in profile.occ_ladder.items()}
                    # §6.1 maximum_disclosure is a CEILING coarser than the
                    # secret -- the plan's own example maps "solicitor" to
                    # "professional", not to "legal". "legal" would already
                    # disclose the stratum being protected.
                    e["maximum_disclosure"] = (
                        "professional" if e["stratum"]["OCC-4"] in
                        profiles._PROF
                        else "non-professional" if e["stratum"]["OCC-4"]
                        else None)
                elif p["tag"] == "RELATIONSHIP":
                    e["stratum"] = {lvl: fn(v)
                                    for lvl, fn in profile.rel_ladder.items()}
                    # for "married" the REL-V2 stratum IS the gold value, so
                    # using it as the ceiling would permit full disclosure;
                    # the ceiling for every REL secret is existence only
                    e["maximum_disclosure"] = "relationship status unspecified"
                else:
                    e["stratum"] = {"SEX": v}
                    e["maximum_disclosure"] = "person"
                entries.append(e)
    return {"gold_path": gold_path, "dataset": profile.name,
            "domain": profile.domain, "min_certainty": MIN_CERTAINTY,
            "subjects": subjects, "entries": entries}


def freeze_hash(reg):
    """Content hash recorded alongside every downstream result, so a silently
    rebuilt registry cannot masquerade as the frozen one."""
    blob = json.dumps(reg, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def load(path="runs/relaware/registry.json"):
    reg = json.load(open(path))
    stored = reg.pop("_freeze_hash")
    if freeze_hash(reg) != stored:
        raise RuntimeError(f"registry {path} does not match its freeze hash -- "
                           "it was edited after Phase 0 froze it")
    reg["_freeze_hash"] = stored
    return reg
