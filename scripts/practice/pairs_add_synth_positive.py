#!/usr/bin/env python3
# Додає у practice_pairs пари «оригінал ↔ обфускований» (label=1) з анти-дублем.
# python3 scripts/practice/pairs_add_synth_positive.py --suffix '#aug1'

import argparse
import os

from clickhouse_connect import get_client


def ch():
    return get_client(
        host=os.getenv("CH_HOST", "127.0.0.1"),
        port=int(os.getenv("CH_PORT", "8123")),
        username=os.getenv("CH_USER", "default"),
        password=os.getenv("CH_PASS", "1234"),
        database=os.getenv("CH_DB", "codebase"),
        interface=os.getenv("CH_IFACE", "http"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="javascript,go,python")
    ap.add_argument("--suffix", default="#aug1")
    ap.add_argument("--split", default="valid")
    args = ap.parse_args()

    client = ch()
    langs = tuple(x.strip() for x in args.langs.split(",") if x.strip())

    # 1) кандидатні пари (orig ↔ aug)
    q_candidates = """
      SELECT o.lang AS lang, o.uid AS uid_orig, a.uid AS uid_aug
      FROM practice_codes a
      JOIN practice_codes o
        ON a.lang=o.lang
       AND a.split='aug'
       AND o.split IN ('train','validation')
       AND o.uid = replaceAll(a.uid, %(suffix)s, '')
      WHERE a.lang IN %(langs)s
    """
    cand = client.query(
        q_candidates, parameters={"suffix": args.suffix, "langs": langs}
    ).result_rows
    if not cand:
        print("No synthetic candidates found")
        return

    # 2) уже існуючі пари (в обох орієнтаціях)
    q_existing = """
      SELECT a_lang, a_uid, b_lang, b_uid
      FROM practice_pairs
      WHERE split=%(split)s
    """
    existing = set()
    for a_lang, a_uid, b_lang, b_uid in client.query(
        q_existing, parameters={"split": args.split}
    ).result_rows:
        existing.add((a_lang, a_uid, b_lang, b_uid))
        existing.add((b_lang, b_uid, a_lang, a_uid))  # дзеркало

    # 3) відфільтруємо нові
    to_insert = []
    for lang, u_o, u_a in cand:
        key = (lang, u_o, lang, u_a)
        if key in existing:
            continue
        to_insert.append((args.split, lang, u_o, lang, u_a, 1, "synthetic:rename_ids"))

    if not to_insert:
        print("No new synthetic pairs to insert (all duplicates)")
        return

    client.insert(
        "practice_pairs",
        to_insert,
        column_names=["split", "a_lang", "a_uid", "b_lang", "b_uid", "label", "notes"],
    )
    print(f"Inserted synthetic positives: {len(to_insert)}")


if __name__ == "__main__":
    main()
