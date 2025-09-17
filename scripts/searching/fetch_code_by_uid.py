from pathlib import Path

import duckdb


def fetch_code_by_uid(lang: str, split: str, uid: str) -> str | None:
    base = (
        Path(__file__).resolve().parents[2]
    )  # go 2 levels up from /scripts/searching/
    p = str(base / f"parquet/cleaned/{lang}/{split}/*.parquet")
    con = duckdb.connect()
    row = con.execute(
        "SELECT code FROM read_parquet(?) WHERE uid = ? LIMIT 1", [p, uid]
    ).fetchone()
    return row[0] if row else None


if __name__ == "__main__":
    code = fetch_code_by_uid("java", "train", "377116_2")
    print(code)

    print("-------------------------------------------------------------")

    code = fetch_code_by_uid("java", "train", "371290_2")
    print(code)
