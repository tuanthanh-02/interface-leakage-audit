"""
The `match(t_i, g)` condition of INTACT (Pilan et al., arXiv:2412.12928, S3.2).

A replacement candidate is *risky* when at least one of the adversary's guesses
matches the original span. Matching is deliberately looser than string equality:
it fires on lemma overlap, on substring containment between long lemmas, and on
character n-gram overlap for named entities (so Turkey/Turkish counts as a hit).
DATETIME is stricter — a date only counts as guessed when the whole set of
lemmas matches — because the rule-based date ladder legitimately keeps part of
the original span.

Ported from guess.py in the reference implementation, with the module-level
globals it relied on turned into explicit arguments.
"""
import json
import re
from pathlib import Path

import spacy

_NLP = None


def nlp():
    """spaCy pipeline used for lemmas and NER (loaded once)."""
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_md")
    return _NLP


def load_allowed_words(path):
    """Frequent lemmas that may match without counting as a disclosure.

    The reference implementation ships the 200 most frequent lemmas of the TAB
    training set: in ECHR judgments words like "applicant" or "court" carry no
    identifying information, so a guess sharing them is not a leak.
    """
    p = Path(path)
    return set(json.loads(p.read_text())) if p.exists() else set()


def convert_to_acronym(tkn_list):
    """Coin an acronym from the title-cased tokens of a span."""
    acronym = ""
    for tkn in tkn_list:
        if tkn.istitle() and tkn not in ["A", "An", "The"]:
            acronym += tkn[0]
    return acronym if len(acronym) > 2 else None


def get_lemma_set(proc_text):
    return {t.lemma_.lower() for t in proc_text if not t.is_stop and not t.is_punct}


def get_ngrams(proc_text, ngram_size, allowed, only_ner=False):
    ngrams = set()
    for token in proc_text:
        if not only_ner or token.ent_type_ not in ("", "DATE"):
            if token.lemma_.lower() not in allowed:
                for i in range(len(token.text) - ngram_size + 1):
                    ngrams.add(token.text[i:i + ngram_size])
    return ngrams


_ORDINAL = re.compile(r"^(\d+)(st|nd|rd|th)$")


def ordinal_to_cardinal(s):
    m = _ORDINAL.match(str(s).strip().lower())
    return m.group(1) if m else s


def _date_key(text):
    """Very small date normaliser: returns (day, month, year) when parseable."""
    from datetime import datetime
    for fmt in ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
                "%Y-%m-%d", "%d/%m/%Y", "%B %Y", "%Y"):
        try:
            d = datetime.strptime(text.strip(), fmt)
            return (d.day, d.month, d.year)
        except ValueError:
            continue
    return None


def is_guessed(guesses, original_span, orig_ent_type, allowed, ngram_size=4):
    """True when any guess recovers the original span."""
    for guess in guesses:
        if guess.strip().lower() == original_span.strip().lower():
            return True, "exact"

    pipe = nlp()
    processed_orig = pipe(original_span)
    orig_lemmas = get_lemma_set(processed_orig)
    orig_acronym = convert_to_acronym([t.text for t in processed_orig])
    if orig_acronym:
        orig_lemmas.add(orig_acronym)

    for guess in guesses:
        processed_guess = pipe(guess)
        guess_lemmas = {l for l in get_lemma_set(processed_guess) if l not in allowed}
        guess_acronym = convert_to_acronym([t.text for t in processed_guess])
        if guess_acronym:
            guess_lemmas.add(guess_acronym)

        if orig_ent_type == "DATETIME":
            od, gd = _date_key(original_span), _date_key(guess)
            if od and gd and od == gd:
                return True, "date"
            # only a full lemma match counts, since the date ladder keeps part
            # of the original span by construction
            if sorted(ordinal_to_cardinal(l) for l in orig_lemmas) == \
               sorted(ordinal_to_cardinal(l) for l in guess_lemmas):
                return True, "date-lemmas"
            continue

        checked = {l for l in orig_lemmas if l not in allowed}
        if checked & guess_lemmas:
            return True, "lemma"
        for o in checked:
            if len(o) > 3:
                for g in guess_lemmas:
                    if len(g) > 3 and (o in g or g in o):
                        return True, "lemma-contained"
        if get_ngrams(processed_orig, ngram_size, allowed, only_ner=True) & \
           get_ngrams(processed_guess, ngram_size, allowed, only_ner=True):
            return True, "ngram"

    return False, None
