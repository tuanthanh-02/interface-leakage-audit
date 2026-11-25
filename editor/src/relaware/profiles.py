"""Dataset profiles: the ONE place that knows how a corpus differs.

Every other module takes a profile object rather than sniffing the dataset for
itself. That rule exists because this repository already paid for the
alternative: five separate `detect_dataset`-style helpers drifted apart and one
of them silently filed SynthPAI results under `evaluation/panorama/`.

What actually differs between the two corpora, measured rather than assumed:

                          TAB                    SynthPAI
  subjects / document     4.07                   1.00
  subject named in text   yes (gold NAME)        NO -- Reddit username, and
                                                 it appears in 0/281 texts
  reference person        third ("he", "Mr X")   first ("I", "my")
  RELATIONSHIP items      45, 3 values           108, 6 values
  OCCUPATION values       84, legal-heavy        134, ordinary jobs

The first-person case breaks the third-person machinery completely, not
partially: gating cue extraction on sentences containing a subject surface
form yields ZERO cues on all 281 SynthPAI documents, and every method arm then
returns the input unchanged while the pipeline reports a full table. The
`person` field is what prevents that.
"""
import re

# --------------------------------------------------------------- OCCUPATION
# One ladder for both corpora: a stratum has to mean the same thing everywhere
# or the endpoint is not comparable across datasets. Rules are ordered, first
# match wins, and Phase -1 reports whatever they cannot map as UNMAPPED.
OCC_RULES = [
    ("legal", r"\b(lawyer|judge|prosecutor|solicitor|barrister|notary|"
              r"attorney|magistrate|jurist|legal|bailiff)\b"),
    ("healthcare", r"\b(doctor|physician|nurse|surgeon|psychiatrist|dentist|"
                   r"medical|paramedic|midwife|anaesthesiologist|psychologist|"
                   r"therapist|healthcare|pharmacist|resident|"
                   r"lab(?:oratory)? technician|"
                   # MedQA is full of specialty names
                   r"\w*ologist|obstetrician|pediatrician|paediatrician|"
                   r"nutritionist|dental assistant|radiation technologist)\b"),
    ("non_professional", r"\b(unemployed|retired|farmer|housewife|homemaker|"
                         r"labourer|laborer|worker|driver|cleaner|shopkeeper|"
                         r"waiter|waitress|no occupation|pensioner|student|"
                         r"car park operator|pager operator|landlord|landowner|"
                         r"antique dealer|drug trafficker|barista|bartender|"
                         r"baker|chef|carpenter|fisherman|hairdresser|"
                         r"tour guide|clerk|part-time|freelancer|"
                         r"bookstore employee|gym trainer|tutor|bookkeeper|"
                         r"cashier|hairstylist|hotel staff|plumber|"
                         r"woodworker|sales|salesman|caregiver|caretaker|"
                         r"attendant|delivery|employee|foreman|picker|"
                         r"gardener|office assistant|retail|seamstress|"
                         r"shipbuilder|typist|welder|zookeeper|volunteer|"
                         r"administrative|assistant|dancer|athlete|"
                         r"bodybuilding|adult entertainment|"
                         r"tennis player)\b"),
    ("other_professional", r"\b(government|diplomat|politician|minister|"
                           r"officer|official|police|military|soldier|teacher|"
                           r"professor|engineer|journalist|accountant|banker|"
                           r"business|businessman|businesswoman|manager|"
                           r"director|civil servant|clergy|priest|architect|"
                           r"economist|scientist|researcher|consultant|"
                           r"representative|agent|translator|interpreter|"
                           r"pilot|mayor|governor|ambassador|entrepreneur|"
                           r"self-employed|academic|auditor|editor|publisher|"
                           r"administrator|inspector|chancellor|commissioner|"
                           r"expert|headmaster|ministerialdirigent|"
                           r"museum operator|political leader|preacher|"
                           r"television personality|writer|"
                           # SynthPAI's ordinary professions
                           r"analyst|anthropologist|archaeologist|curator|"
                           r"artist|astronomer|ceo|cryptographer|designer|"
                           r"developer|executive|activist|geologist|"
                           r"historian|technician|lecturer|librarian|"
                           r"biologist|specialist|critic|painter|physicist|"
                           r"planner|principal|sculptor|advisor|adviser|"
                           r"developer|owner|finance|financial|marketing|"
                           r"real estate|urban|author|firefighter|coordinator|"
                           r"it (?:professional|support)|technology|"
                           r"programmer|investigator|law enforcement|"
                           r"army instructor|poet)\b"),
]

# §11's ladder is written for TAB, where `legal` is the dominant and
# interesting stratum (210 of 391 items). On SynthPAI legal is a rounding
# error -- 3 items of 240 -- so a ladder that keeps forcing a `legal` stratum
# fails §11.1 at every level and blocks the corpus entirely. SynthPAI
# therefore coarsens in the other direction: merge legal upward, and let the
# professional/non-professional split be the coarse level. The stratum LABELS
# keep their meaning in both, which is what makes the endpoint comparable.
_PROF = {"legal", "healthcare", "other_professional"}

OCC_LADDERS = {
    "tab": {
        "OCC-4": lambda s: s,
        "OCC-3": lambda s: "other_professional" if s == "healthcare" else s,
        "OCC-2": lambda s: "legal" if s == "legal" else "other",
    },
    "synthpai": {
        "OCC-4": lambda s: s,
        "OCC-3": lambda s: "other_professional" if s == "legal" else s,
        "OCC-2": lambda s: "professional" if s in _PROF else "non_professional",
    },
}
# PANORAMA has two legal items in 132 -- the TAB ladder's forced `legal`
# stratum fails §11.1 at every level there, exactly as it did on SynthPAI
OCC_LADDERS["panorama"] = OCC_LADDERS["synthpai"]
# MedQA occupations are ordinary jobs mentioned in passing, with almost no
# legal ones -- same shape as SynthPAI, same ladder
OCC_LADDERS["medqa"] = OCC_LADDERS["synthpai"]
OCC_LADDER = OCC_LADDERS["tab"]        # default for pre-profile callers


def occ_stratum(value):
    v = value.strip().lower()
    for name, pat in OCC_RULES:
        if re.search(pat, v):
            return name
    return None


# ------------------------------------------------------------- RELATIONSHIP
# The gold is a marital/partnership status in both corpora -- NOT the six
# pair-relation types of plan §10, which need two subjects in one document and
# are therefore unavailable on SynthPAI by construction (1.00 subjects/doc).
_PARTNERED = {"married", "in relationship", "engaged"}
_FORMER = {"divorced", "widowed"}

# MedQA's RELATIONSHIP gold is heterogeneous: SPIA's annotator emitted marital
# status (no relation 434, in relation 145, married 131, widowed 14, divorced
# 3) AND bare kinship roles (father, mother, sister, ... 23 items) AND two
# non-statuses (deceased, dependent). Only the marital half is a relationship
# STATUS, so the kinship half maps to None -- excluded from the endpoint and
# counted as unstratified, never silently pooled. TAB's ladder must not be
# reused here: it maps everything-not-married to "formerly_married", which
# would relabel all 434 "no relation" subjects as formerly married.
_MEDQA_PARTNERED = {"married", "in relation", "in relationship", "engaged"}
_MEDQA_FORMER = {"divorced", "widowed"}
_MEDQA_SINGLE = {"no relation", "single"}


def _medqa_rel(v, coarse=False):
    if v in _MEDQA_PARTNERED:
        return "partnered"
    if v in _MEDQA_FORMER:
        return "not_partnered" if coarse else "formerly_partnered"
    if v in _MEDQA_SINGLE:
        return "not_partnered" if coarse else "single"
    return None                      # kinship role or non-status: unstratified


REL_LADDERS = {
    # finest first; check_strata picks the finest that clears §11.1
    "tab": {
        "REL-V3": lambda v: v,
        "REL-V2": lambda v: "married" if v == "married" else "formerly_married",
    },
    "medqa": {
        "REL-V3": lambda v: _medqa_rel(v),
        "REL-V2": lambda v: _medqa_rel(v, coarse=True),
    },
    "synthpai": {
        "REL-V6": lambda v: v,
        "REL-V3": lambda v: ("partnered" if v in _PARTNERED
                             else "formerly_partnered" if v in _FORMER
                             else "single"),
        "REL-V2": lambda v: "partnered" if v in _PARTNERED else "not_partnered",
    },
}

# §10.2's "Prior 95% CI width" is bootstrapped over canonical handles. The
# registry assigns "Subject A".."Subject G" -- six or seven strings differing
# by ONE letter -- so the measured width (.44-.50 on TAB and PANORAMA) is
# letter-position noise in the probe, not uncertainty about the prior, and it
# is currently the only criterion demoting REL on two of three corpora.
#
# These paraphrases vary the surface form properly while obeying §7: no name,
# sex, occupation, legal role or family relationship in any of them. A wider,
# genuinely varied set gives §10.3 a fair measurement -- which may confirm the
# demotion (if the probe is truly handle-unstable) or overturn it. Either
# outcome is informative; the current one is not.
CANONICAL_PARAPHRASES = (
    "Subject A", "Subject B", "Subject C",
    "the first individual", "the second individual",
    "Person 1", "Person 2", "Person 3",
    "this person", "that person",
    "the individual in question", "the individual described",
    "the person referred to", "the person concerned",
    "one of the people involved", "another of the people involved",
    "the party in question", "the other party",
    "an individual in this record", "a person in this record",
    "the first person mentioned", "the second person mentioned",
    "someone described here", "someone referred to here",
)

SEX_VALUES = ("male", "female")
REL_TYPES = ("spouse_or_partner", "parent", "child", "sibling",
             "other_family", "non_family_personal")

MIN_PAIR_N = 10
MIN_DOC_N = 8


# -------------------------------------------------------------- cue lexicons
# THIRD PERSON (TAB): the subject is named and written about.
THIRD = {
    "pronoun": r"\b(he|she|him|her|hers|his|himself|herself)\b",
    "title": r"\b(Mr|Mrs|Ms|Miss)\.?(?=\s)",
    "gendered_kin": r"\b(husband|wife|widow(?:er)?|father|mother|son|"
                    r"daughter|brother|sister)\b",
    "kinship": r"\b(wife|husband|spouse|widow(?:er)?|married|marriage|"
               r"divorced?|partner|fianc[eé]e?|the couple|wedding)\b",
    "occ_role": r"\b(lawyer|solicitor|barrister|attorney|judge|prosecutor|"
                r"notary|magistrate|law firm|the court|chambers|"
                r"legal (?:practice|profession|adviser)|doctor|physician|"
                r"nurse|surgeon|hospital|clinic|patient[s]? of|teacher|"
                r"professor|school|university|engineer|company|"
                r"business(?:man|woman)?|farm(?:er)?|unemploy(?:ed|ment)|"
                r"pension(?:er)?|retire[dment]*|police|military|army|officer|"
                r"minister|government|diplomat|politician|journalist|"
                r"represent(?:ed|ing) (?:the|his|her|a))\b",
}

# FIRST PERSON (SynthPAI): the subject is the author. "I" carries no gender,
# so SEX leaks through gendered self-reference and partner terms instead --
# which is exactly why this corpus is the interesting one for RELATIONSHIP.
FIRST = {
    "pronoun": r"\b(?:as a |i'?m a |being a )(guy|girl|man|woman|dude|"
               r"lady|mom|mum|dad|father|mother|wife|husband|bloke)\b",
    "title": r"(?!x)x",                      # no titles in first person
    "gendered_kin": r"\bmy (husband|wife|boyfriend|girlfriend|fianc[eé]e?)\b",
    "kinship": r"\b(my (?:husband|wife|partner|boyfriend|girlfriend|"
               r"fianc[eé]e?|ex|spouse)|single|divorced?|engaged|married|"
               r"marriage|widow(?:ed|er)?|in a relationship|"
               r"we(?:'ve| have) been together|my kids|my children)\b",
    "occ_role": r"\b(i work (?:as|at|in)|my (?:job|career|clients|patients|"
                r"students|colleagues|boss|office|shift|company)|"
                r"as a [a-z-]+(?:er|ist|or|ian|ant|ent)\b|at work|"
                r"day job|freelanc\w+|self-employed|"
                r"lawyer|nurse|doctor|teacher|engineer|designer|developer|"
                r"analyst|chef|barista|architect|librarian|curator|"
                r"therapist|technician|consultant|manager|artist|writer|"
                r"researcher|scientist|professor|lecturer|student|retired|"
                r"unemployed)\b",
}


# MIXED (PANORAMA): forum threads. 138/151 documents are speaker turns
# ("Christian : <first-person text>"); subjects are the speakers AND people
# they mention ("kid 1", a spouse). A subject's evidence therefore lives in
# two voices at once: first person inside their own turns, third person where
# others mention them. The lexicon is the union of both and the scope is
# own-turns ∪ mention-sentences; a mentioned-only subject that never appears
# by name yields no cues, and Detection Coverage reports that honestly
# rather than the profile pretending otherwise.
def _union(a, b):
    return {k: f"(?:{a[k]}|{b[k]})" if a[k] != b[k] else a[k] for k in a}


MIXED = _union(THIRD, FIRST)


def _add_alts(pattern, extra):
    """Insert alternatives before a pattern's closing `)\\b`.

    Slicing a fixed number of characters off the end silently produces an
    unbalanced group when the tail is not exactly what you assumed -- it did
    here, and only a compile check caught it."""
    tail = r")\b"
    if not pattern.endswith(tail):
        raise ValueError(f"pattern does not end with {tail!r}: ...{pattern[-20:]!r}")
    out = pattern[:-len(tail)] + "|" + extra + tail
    re.compile(out)
    return out


class Profile:
    def __init__(self, name, domain, person, lexicon, rel_ladder,
                 subject_is_author, turn_scope=False):
        self.name = name
        self.domain = domain
        self.person = person                 # "third" | "first"
        self.lex = lexicon
        self.rel_ladder = rel_ladder
        self.occ_ladder = OCC_LADDERS.get(name, OCC_LADDERS["tab"])
        self.subject_is_author = subject_is_author
        self.turn_scope = turn_scope

    def rel_values(self, entries):
        return sorted({e["gold_value"] for e in entries
                       if e["attribute"] == "RELATIONSHIP"})


PROFILES = {
    "tab": Profile(
        "tab", "a European Court of Human Rights judgment", "third",
        THIRD, REL_LADDERS["tab"], subject_is_author=False),
    "synthpai": Profile(
        "synthpai", "a Reddit discussion", "first",
        FIRST, REL_LADDERS["synthpai"], subject_is_author=True),
    # MedQA: USMLE vignettes. Third person, patient never named, demographics
    # stated formulaically in the opening clause ("A 39-year-old woman
    # presents with..."). Closest to TAB in grammar, closest to PANORAMA in
    # length. Uses the TAB lexicon plus the clinical role words a vignette
    # actually contains; the marital ladder is TAB's because the gold values
    # SPIA's annotator emits are the same closed set.
    "medqa": Profile(
        "medqa", "a clinical case vignette", "third",
        dict(THIRD,
             # MedQA states sex in the opening clause -- "A 39-year-old
             # woman presents with..." -- not through pronouns alone. Without
             # these the detector reaches 16% of SEX propositions and the
             # method is judged on a lexicon gap rather than on the register.
             pronoun=_add_alts(THIRD["pronoun"],
                               r"man|woman|men|women|male|female|boy|girl|"
                               r"gentleman|lady|mother|father"),
             occ_role=_add_alts(
            THIRD["occ_role"],
            r"patient|physician|resident|attending|intern|pharmacist|"
            r"technician|therapist|paramedic|surgeon|midwife")),
        REL_LADDERS["medqa"], subject_is_author=False),
    "panorama": Profile(
        "panorama", "a social-media discussion thread", "mixed",
        MIXED, REL_LADDERS["tab"], subject_is_author=False,
        turn_scope=True),
}


def for_path(gold_path):
    """The single dataset detector. Everything else takes the profile."""
    low = gold_path.lower()
    for name in ("synthpai", "panorama", "medqa", "tab"):
        if name in low:
            return PROFILES[name]
    raise ValueError(f"cannot tell which corpus {gold_path} is; name the file "
                     "so it contains tab, panorama, synthpai or medqa")


def for_registry(reg):
    """Profile of a loaded registry.

    Registries frozen before the profile layer existed carry no `dataset`
    field; they do carry `gold_path`, and re-freezing one to add the field
    would change its hash and invalidate every artifact chained to it. Fall
    back to the path detector rather than forcing a Phase 0 re-run."""
    name = reg.get("dataset")
    if name in PROFILES:
        return PROFILES[name]
    return for_path(reg["gold_path"])


def check_strata(items, ladder_fn, doc_of):
    """Per-stratum counts and whether every stratum clears §11.1."""
    per = {}
    for it, raw in items:
        s = ladder_fn(raw)
        if s is None:
            continue
        d = per.setdefault(s, {"n": 0, "docs": set()})
        d["n"] += 1
        d["docs"].add(doc_of(it))
    ok = all(v["n"] >= MIN_PAIR_N and len(v["docs"]) >= MIN_DOC_N
             for v in per.values()) and len(per) >= 2
    return ({k: {"n": v["n"], "docs": len(v["docs"])} for k, v in per.items()},
            ok)
