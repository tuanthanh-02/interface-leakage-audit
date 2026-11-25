"""Relation-Aware Joint Anonymization -- implementation of the frozen v0.9 plan.

Scope implemented here, in the plan's own execution order (plan §27-29):

  Phase -1   data feasibility, no LLM              scripts/rel_phase_m1.py
  Phase 0    evaluator + cost freeze               scripts/rel_phase0_*.py
  Phase 1    oracle static methods (exploratory)   scripts/rel_phase1_oracle.py

RAG placement (Phase 3), multi-query (Phase 4) and the confirmatory test
(Phase 5) are deliberately NOT here: the plan freezes the document until the
Phase -1/0 quantities exist (§29), and every later phase depends on artifacts
these scripts produce (frozen registry, priors, batch size, strata).

Evaluation-vs-method separation (§6): everything under `method/` sees only the
input text and public lexicons. The gold registry, evaluator candidate sets and
priors live in the top-level modules and are consumed only by evaluators.
"""
