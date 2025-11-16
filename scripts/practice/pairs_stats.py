#!/usr/bin/env python3
# Показує баланс класів і покриття ембеддингами під заданий (model, pool, transform_ver).
import argparse
import os

from clickhouse_connect import get_client


def get_client_env():
    return get_client(
        host=os.getenv("CH_HOST", "127.0.0.1"),
        port=int(os.getenv("CH_PORT", "8123")),
        username=os.getenv("CH_USER", "default"),
        password=os.getenv("CH_PASS", "1234"),
        database=os.getenv("CH_DB", "codebase"),
        interface=os.getenv("CH_IFACE", "http"),
    )


def main():
    ap = argparse.ArgumentParser(
        description="Show practice_pairs stats and embedding coverage"
    )
    ap.add_argument(
        "--model", default=os.getenv("EMB_MODEL", "microsoft/codebert-base")
    )
    ap.add_argument("--pool", default=os.getenv("EMB_POOL", "mean"))
    ap.add_argument("--ver", type=int, default=int(os.getenv("EMB_TRANSFORM_VER", "2")))
    ap.add_argument("--split", default="valid")
    args = ap.parse_args()

    client = get_client_env()
    q1 = """
      SELECT label, count() AS n
      FROM practice_pairs
      WHERE split=%(split)s
      GROUP BY label ORDER BY label
    """
    print("Label distribution:")
    for row in client.query(q1, parameters={"split": args.split}).result_rows:
        print(row)

    q2 = f"""
      WITH
        (SELECT groupArray(uid) FROM practice_embeddings
         WHERE model=%(m)s AND pool=%(p)s AND transform_ver=%(v)s) AS emb_uids
      SELECT
        count() AS pairs_total,
        countIf(has(emb_uids, a_uid) AND has(emb_uids, b_uid)) AS pairs_with_vecs,
        countIf(label=1 AND has(emb_uids, a_uid) AND has(emb_uids, b_uid)) AS pos_with_vecs,
        countIf(label=0 AND has(emb_uids, a_uid) AND has(emb_uids, b_uid)) AS neg_with_vecs
      FROM practice_pairs
      WHERE split=%(split)s
    """
    print("\nCoverage with embeddings:")
    for row in client.query(
        q2,
        parameters={
            "m": args.model,
            "p": args.pool,
            "v": args.ver,
            "split": args.split,
        },
    ).result_rows:
        print(row)


if __name__ == "__main__":
    main()
