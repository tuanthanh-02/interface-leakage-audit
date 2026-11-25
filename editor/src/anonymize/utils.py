"""
Anonymization utilities
"""


def check_length_anomaly(original: str, anonymized: str, threshold: float = 0.4) -> dict:
    """
    Check length difference between original and anonymized text.

    Args:
        original: Original text
        anonymized: Anonymized text
        threshold: Ratio threshold for anomaly detection (default: 0.3 = 30%)

    Returns:
        dict with:
            - length_diff: Absolute character difference
            - length_ratio: anonymized/original ratio
            - length_anomaly: True if ratio deviates more than threshold from 1.0
    """
    orig_len = len(original)
    anon_len = len(anonymized)

    length_diff = abs(orig_len - anon_len)

    if orig_len == 0:
        length_ratio = 0.0 if anon_len == 0 else float('inf')
    else:
        length_ratio = anon_len / orig_len

    # Anomaly if ratio deviates more than threshold from 1.0
    # e.g., threshold=0.3 means anomaly if ratio < 0.7 or ratio > 1.3
    length_anomaly = abs(length_ratio - 1.0) > threshold

    return {
        "length_diff": length_diff,
        "length_ratio": round(length_ratio, 3),
        "length_anomaly": length_anomaly
    }


def run_batch(json_list, anonymize_one, output_file, workers=1,
              desc="Anonymizing", error_file=None):
    """
    Anonymize a list of SPIA records, appending each result as it completes.

    Two problems with the original per-script loops this replaces:

    * they were sequential, so a vLLM server that happily batches dozens of
      requests only ever saw one at a time;
    * they buffered every result in memory and wrote the file only at the very
      end, so an interrupted multi-hour run lost everything.

    Here each finished document is appended immediately and any document whose
    data_id is already in output_file is skipped, which makes runs resumable.
    The file is re-sorted into the input order once the run completes, so the
    output contract is unchanged.

    Args:
        json_list: input records (SPIA format, each with metadata/text)
        anonymize_one: callable(record) -> anonymized text ("" means failure)
        output_file: JSONL path to append to
        workers: number of documents in flight at once
        desc: progress-bar label
        error_file: where to record failures (default: <output>_errors.jsonl)

    Returns:
        (n_success, n_error)
    """
    import json
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    def _data_id(rec):
        return rec.get("metadata", {}).get("data_id", "unknown")

    done = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(_data_id(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        if done:
            print(f"Resuming: {len(done)} documents already in {output_file}")

    todo = [r for r in json_list if _data_id(r) not in done]
    if not todo:
        print("Nothing to do; output already complete.")
        return len(done), 0

    error_file = error_file or (os.path.splitext(output_file)[0] + "_errors.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    n_ok, n_err = len(done), 0
    out_fh = open(output_file, "a", encoding="utf-8")
    err_fh = open(error_file, "a", encoding="utf-8")

    def _work(rec):
        try:
            return rec, anonymize_one(rec), None
        except Exception as exc:                      # keep one bad doc from killing the run
            return rec, "", f"{type(exc).__name__}: {exc}"

    try:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(_work, r) for r in todo]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                rec, anon, err = fut.result()
                if err or not anon:
                    n_err += 1
                    err_fh.write(json.dumps(
                        {"data_id": _data_id(rec),
                         "error": err or "Anonymization returned empty result"},
                        ensure_ascii=False) + "\n")
                    err_fh.flush()
                    continue
                row = {"metadata": rec["metadata"], "text": rec["text"],
                       "anonymized_text": anon, **check_length_anomaly(rec["text"], anon)}
                out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_fh.flush()
                n_ok += 1
    finally:
        out_fh.close()
        err_fh.close()

    # Restore input order (concurrent completion scrambles it) and drop
    # duplicates: `done` is read once at start-up, so two processes resuming the
    # same unfinished file will both redo whatever was still missing.
    order = {_data_id(r): i for i, r in enumerate(json_list)}
    with open(output_file, "r", encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh if l.strip()]
    seen, deduped = set(), []
    for r in rows:
        rid = _data_id(r)
        if rid in seen:
            continue
        seen.add(rid)
        deduped.append(r)
    if len(deduped) != len(rows):
        print(f"Dropped {len(rows) - len(deduped)} duplicate record(s)")
    rows = deduped
    rows.sort(key=lambda r: order.get(_data_id(r), 1 << 30))
    with open(output_file, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nSaved {n_ok} documents to '{output_file}' ({n_err} errors)")
    return n_ok, n_err
