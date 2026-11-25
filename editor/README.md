# SACC: scoped attribute-cue closure (anonymised release)

Deterministic editor for residual attribute cues in anonymised text: it sweeps a document with closed lexicons, sets aside
the cues that dependency rules show to describe someone else or no person, and rewrites every other cue to a counterpart
that carries no value of the protected attribute, or redacts it. No model call; CPU only.

```bash
pip install spacy numpy nltk && python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet')"      # the MedQA upstream tagger uses WordNet occupation nouns
python reproduce_sacc.py --verify      # rebuilds every released text and compares it byte for byte (about 15 min)
python tables.py                       # recomputes the main privacy tables from the released scores (numpy only)
```

## 1. The editor: `reproduce_sacc.py`

| file in `released/arms/` | system |
|---|---|
| `{corpus}_C1v3.jsonl` | SACC-mask: selection, every selected cue redacted |
| `{corpus}_SACC_Nfix3.jsonl` | SACC (MedQA: full agreement setting; TAB: light; SynthPAI: equal to SACC-mask) |
| `medqa_SACC_Nfix3_light.jsonl` | SACC on MedQA in the light agreement setting |

The rules live in `scripts/dev_phase10_emit_m2.py` (sweep, scope, redaction), `scripts/dev_phase10_binding.py` (the
binder; rule R9 is applied in `reproduce_sacc.py`), `src/relaware/profiles.py` (the lexicons) and the rewrite step in
`scripts/dev_sacc_stack_rewrite_emit.py`, `scripts/dev_p22_n2_emit.py` and `scripts/dev_sacc_fix_emit.py`. File names
are those of the development repository and are kept so that the code runs unchanged (37 code files,
16 data files).

## 2. The scoring harness: `score_texts.py`

Scores any system's texts with the paper's adversary probe: priors without text, posteriors over eight seeded candidate
orders, the subject reference built from the scored text, entries at or above the frozen prior threshold left out. Use
`--adversary openai` (key in `.env`) or `--adversary vllm --base-url ...`. Run against the request cache of the paper's
development adversary, it reproduces the stored posterior of every one of 300 sampled entries exactly.

## 3. The scores: `scores/`

* `scores/passes/`: every scoring pass behind the tables, one gzipped JSONL per pass (system, entry, attribute, stratum,
  prior, posterior; a hash replaces the subject reference; gold values are not included).
* `scores/utility/`, `scores/tps/`: per-document U_nli and TPS.
* `scores/main/`: the development adversary's scores of every system as the main tables use them, and the paper's numbers
  (`expected.json`); `tables.py` recomputes Table 1's privacy and U_nli levels and Table 2 and checks every number.
* `scores/priors/`: the development adversary's priors.

## Data

The inputs under `data/` and `runs/` and the scores derive from the subject-level benchmark of Oh et al. (2026), which
builds on MedQA, the Text Anonymization Benchmark and SynthPAI; use them under those resources' licences.
`src/anonymize/` and `src/baseline/` are the benchmark's code (`LICENSE-SPIA`).
