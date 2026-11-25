# Reproduction package: the editor of Algorithm 1 and the SWAP experiments

This package contains the two things the review asked for: the complete editor implementation of
Algorithm 1, and the scripts and intermediate results needed to reproduce the SWAP experiments
independently.

`editor/` rebuilds every released SACC and SACC-mask text from the anonymised input, including the
dependency binder and the agreement repair, and compares it byte for byte:

    cd editor
    pip install spacy numpy nltk && python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('wordnet')"
    python reproduce_sacc.py --verify      # about 15 minutes; prints identical for every text
    python tables.py                       # recomputes the main privacy tables from the released scores

`swap_addendum/` rebuilds the released SWAP texts (standard library only), checks them against the
released files, and runs the reproduction gate:

    cd swap_addendum
    python swap/build_swap.py --corpus medqa --check swap/released/medqa_SWAP.jsonl   # 1249/1249 identical
    python swap/build_swap.py --corpus tab   --check swap/released/tab_SWAP.jsonl     # 144/144 identical
    python swap/build_swap.py --corpus medqa --gate                                   # 1245/1249 (99.7 per cent)
    python swap/build_swap.py --corpus tab   --gate                                   # 144/144 (100 per cent)

It also contains SWAP's per-entry scores and the prompt and pass of the rule-aware adversary.
