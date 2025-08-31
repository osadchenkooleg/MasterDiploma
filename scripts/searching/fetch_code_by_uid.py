import duckdb


def fetch_code_by_uid(lang: str, split: str, uid: str) -> str | None:
    p = f"parquet/cleaned/{lang}/{split}/*.parquet"
    con = duckdb.connect()
    row = con.execute(
        "SELECT code FROM read_parquet(?) WHERE uid = ? LIMIT 1", [p, uid]
    ).fetchone()
    return row[0] if row else None


if __name__ == "__main__":
    code = fetch_code_by_uid("java", "train", "63228_1")
    print(code)

    print("-------------------------------------------------------------")

    code = fetch_code_by_uid("java", "train", "200_1")
    print(code)
