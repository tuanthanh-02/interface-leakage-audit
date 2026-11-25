# Supplementary addendum: SWAP and the rule-aware test

This addendum completes the supplementary package for the parts a reviewer could not reproduce
from it: the construction of SWAP and its scores (Section 4, Appendix D). It is meant to be shipped together with:

| part | what it reproduces | entry point |
|---|---|---|
| editor release package (`sacc_release_package.zip`) | SACC and SACC-mask texts from the anonymised input, byte for byte: cue sweep, dependency binder (other-person and non-person rules), admissible rewriting, agreement repair | `python reproduce_sacc.py --verify` |
| table package (original supplementary) | every privacy number of the main tables from stored per-entry scores | `python scripts/emit_sacc_paper_tables.py` |
| **this addendum, `swap/`** | SWAP texts from SACC-mask, the reproduction gate, SWAP's scores and the rule-aware pass | `python swap/build_swap.py` |

## `swap/`

```bash
python swap/build_swap.py --corpus medqa --check swap/released/medqa_SWAP.jsonl   # 1249/1249 identical
python swap/build_swap.py --corpus tab   --check swap/released/tab_SWAP.jsonl     # 144/144 identical
python swap/build_swap.py --corpus medqa --gate    # flips written as [REDACTED] give SACC-mask back: 1245/1249 (99.7%)
```

Standard library only. `data/` holds the three inputs the construction reads: the text before cue
editing, the released SACC-mask text, and the editor's cue trace (offsets, surface form, syntactic
role, attribute of every selected cue).

| file in `swap/scores/` | content | paper |
|---|---|---|
| `e2_{medqa,tab}_SWAP_full_gpt4omini.json` | per-entry prior and posterior of SWAP, development adversary, 8 seeded orders | Table 2 |
| `e1_metric_family.json` | accuracy, Brier, log loss, AUC readings of the same entries | Table 6 |
| `e2_displacement.json` | LEAK / ANON / MISLED shares | App. O |
| `e2_medqa_SWAP_rules_gpt4omini.json`, `x4.json` | the rule-aware pass: the adversary told the swap rule (`rule_aware_prompt.txt`, prefixed to the standard question) | §4, App. D |
| `randcue_read_doc.json` | control with SWAP's sites, vocabulary and realiser and a seeded coin for direction | response only |

Per-entry scores are re-scored with the scoring harness of the editor release package
(`score_texts.py`, `--adversary openai` or `--adversary vllm`).

SWAP attributes false sex values to the people its texts describe; it is an evaluation construct,
not an anonymiser. Derived texts and scores keep the licences of MedQA, TAB and SynthPAI; code MIT.
