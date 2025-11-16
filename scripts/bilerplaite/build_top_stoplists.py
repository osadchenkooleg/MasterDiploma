#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from clickhouse_connect import get_client

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")

STOPLIST_TABLE = "codebase.stoplist_shingles"

DEFAULT_LANGS = ["java", "js", "go", "python"]
DEFAULT_TOP_N = 2000
DEFAULT_OUT_DIR = "data/stoplists"


def build_top_stoplists(langs, top_n: int, out_dir: str):
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    client = get_client(
        host=CH_HOST,
        username=CH_USER,
        password=CH_PASS,
        database=CH_DB,
    )

    for lang in langs:
        print(f"=== Building stoplist for lang={lang}, top_n={top_n} ===")

        # беремо найчастіші шингли для цієї мови
        rows = client.query(
            f"""
            SELECT shingle, df
            FROM {STOPLIST_TABLE}
            WHERE lang = %(lang)s
            ORDER BY df DESC
            LIMIT %(top_n)s
            """,
            {"lang": lang, "top_n": top_n},
        ).result_rows

        if not rows:
            print(f"  no rows for lang={lang}, skipping")
            continue

        out_file = out_dir_path / f"stoplist_{lang}.txt"
        with out_file.open("w", encoding="utf-8") as f:
            for shingle, df in rows:
                # пишемо лише текст шингла; df нам не потрібен під час фільтрації
                f.write(shingle.replace("\n", " ") + "\n")

        print(f"  wrote {len(rows)} shingles -> {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Build top-N stop shingle lists per language."
    )
    parser.add_argument(
        "--langs",
        type=str,
        default=",".join(DEFAULT_LANGS),
        help="Comma-separated languages (e.g. 'python,java,js,go')",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help=f"How many shingles per language to keep (default: {DEFAULT_TOP_N})",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for stoplist files (default: {DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    build_top_stoplists(langs, args.top_n, args.out_dir)


if __name__ == "__main__":
    main()
