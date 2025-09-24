#!/usr/bin/env python3
# Видаляє пари певного типу (і чекає завершення мутації).
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
    ap = argparse.ArgumentParser(description="Delete practice_pairs by notes value")
    ap.add_argument(
        "--notes",
        required=True,
        help="Exact notes value to delete (e.g., synthetic:rename_ids)",
    )
    ap.add_argument("--split", default="valid")
    args = ap.parse_args()

    client = get_client_env()
    sql = """
      ALTER TABLE practice_pairs
      DELETE WHERE split=%(split)s AND notes=%(notes)s
      SETTINGS mutations_sync=1
    """
    r = client.command(sql, parameters={"split": args.split, "notes": args.notes})
    print("Delete done:", r or "OK")


if __name__ == "__main__":
    main()
