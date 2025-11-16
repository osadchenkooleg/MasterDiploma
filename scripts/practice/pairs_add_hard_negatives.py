#!/usr/bin/env python3
# Додає hard-negative (label=0) з обмеженням різниці довжин та лімітом.
# python3 scripts/practice/pairs_add_hard_negatives.py --limit 3000 --max-delta 20

import argparse
import math

#!/usr/bin/env python3
import os
import random
from typing import List, Tuple

from clickhouse_connect import get_client


def ch():
    # збільшуємо таймаути HTTP-клієнта
    return get_client(
        host=os.getenv("CH_HOST", "127.0.0.1"),
        port=int(os.getenv("CH_PORT", "8123")),
        username=os.getenv("CH_USER", "default"),
        password=os.getenv("CH_PASS", "1234"),
        database=os.getenv("CH_DB", "codebase"),
        interface=os.getenv("CH_IFACE", "http"),
        connect_timeout=30,
        send_receive_timeout=1200,  # 20 хв на великі відповіді
    )


def fetch_existing_pairs(client, split: str) -> set:
    rows = client.query(
        "SELECT a_lang, a_uid, b_lang, b_uid FROM practice_pairs WHERE split=%(s)s",
        parameters={"s": split},
        settings={"max_execution_time": 0},
    ).result_rows
    S = set()
    for aL, aU, bL, bU in rows:
        S.add((aL, aU, bL, bU))
        S.add((bL, bU, aL, aU))  # дзеркало
    return S


def fetch_anchor_sample(client, lang: str, per_lang: int):
    # Беремо випадкові "якорі" c1. MD5 віддаємо у hex, щоб не тягнути сирі байти.
    q = """
      SELECT
        uid,
        toInt32(code_len) AS code_len,
        lower(hex(code_norm_md5)) AS md5_hex
      FROM practice_codes
      WHERE lang = %(l)s AND split IN ('train','validation')
      ORDER BY rand()
      LIMIT %(n)s
    """
    return client.query(
        q, parameters={"l": lang, "n": per_lang}, settings={"max_execution_time": 0}
    ).result_rows


def fetch_candidates_for_lang(client, lang: str, anchors, max_delta: int, need: int):
    """
    anchors: список кортежів (uid, code_len, md5_hex)
    Повертаємо список кандидатів [(a_lang, a_uid, b_lang, b_uid), ...]
    """
    candidates = []
    step = 200  # не робимо надто довгі «пакети» анкорів

    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace("'", "\\'")

    for i in range(0, len(anchors), step):
        if len(candidates) >= need:
            break
        batch = anchors[i : i + step]

        # формуємо список кортежів для table function `values`
        tuples_str = ",".join(
            "('%s', %d, '%s')" % (esc(uid), int(code_len), esc(md5_hex))
            for (uid, code_len, md5_hex) in batch
        )

        q = f"""
        WITH anchors AS (
          SELECT *
          FROM values('uid String, code_len Int32, md5_hex String', {tuples_str})
        )
        SELECT
          '{lang}' AS a_lang, a.uid AS a_uid,
          '{lang}' AS b_lang, c2.uid AS b_uid
        FROM anchors a
        INNER JOIN practice_codes c2
          ON c2.lang = '{lang}'
         AND c2.split IN ('train','validation')
         AND c2.uid > a.uid
         AND abs(c2.code_len - a.code_len) <= %(d)s
         AND lower(hex(c2.code_norm_md5)) != a.md5_hex
        LIMIT %(lim)s
        """
        lim_local = max(need - len(candidates), 1000)
        rows = client.query(
            q,
            parameters={"d": max_delta, "lim": lim_local},
            settings={"max_execution_time": 0},
        ).result_rows
        candidates.extend(rows)

    return candidates[:need]


def main():
    ap = argparse.ArgumentParser(
        description="Add hard-negative pairs with close length (batched & sampled)"
    )
    ap.add_argument(
        "--langs", default="javascript,go,python", help="Comma-separated languages"
    )
    ap.add_argument("--max-delta", type=int, default=20, help="Max |len(a)-len(b)|")
    ap.add_argument(
        "--limit", type=int, default=3000, help="Total pairs to insert across languages"
    )
    ap.add_argument("--split", default="valid")
    ap.add_argument(
        "--anchors-per-lang",
        type=int,
        default=20000,
        help="How many anchor rows to sample per language",
    )
    args = ap.parse_args()

    client = ch()
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    per_lang_target = math.ceil(args.limit / max(1, len(langs)))

    print(f"Target per lang: ~{per_lang_target} pairs (total {args.limit})")

    existing = fetch_existing_pairs(client, args.split)
    print(f"Existing pairs cached: {len(existing)//2} (both directions counted once)")

    to_insert = []

    for lang in langs:
        anchors = fetch_anchor_sample(client, lang, args.anchors_per_lang)
        if not anchors:
            print(f"[{lang}] no anchors, skip")
            continue

        cand = fetch_candidates_for_lang(
            client, lang, anchors, args.max_delta, per_lang_target * 2
        )
        if not cand:
            print(f"[{lang}] no candidates, skip")
            continue

        # антидубль + ліміт
        new_lang_pairs = []
        random.shuffle(cand)
        for a_lang, a_uid, b_lang, b_uid in cand:
            key = (a_lang, a_uid, b_lang, b_uid)
            if key in existing:
                continue
            existing.add(key)
            existing.add((b_lang, b_uid, a_lang, a_uid))
            new_lang_pairs.append(
                (args.split, a_lang, a_uid, b_lang, b_uid, 0, "hard_neg:len_close")
            )
            if len(new_lang_pairs) >= per_lang_target:
                break

        print(
            f"[{lang}] anchors={len(anchors)} -> candidates={len(cand)} -> new={len(new_lang_pairs)}"
        )
        to_insert.extend(new_lang_pairs)

    if not to_insert:
        print("No new hard negatives to insert")
        return

    client.insert(
        "practice_pairs",
        to_insert,
        column_names=["split", "a_lang", "a_uid", "b_lang", "b_uid", "label", "notes"],
    )
    print(f"Inserted hard negatives: {len(to_insert)}")


if __name__ == "__main__":
    main()
