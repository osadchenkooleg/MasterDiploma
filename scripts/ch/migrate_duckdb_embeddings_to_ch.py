import numpy as np
from clickhouse_connect import get_client

import duckdb

DUCKDB_PATH = "duckdb/plag.db"
CH_HOST = "localhost"
CH_PORT = 8123
CH_USER = "default"
CH_PASS = "1234"
CH_DB = "codebase"

BATCH_SIZE = 50000
EXPECTED_DIM = 768


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v) + 1e-12
    return (v / n).astype(np.float32)


def main():
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    client = get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASS, database=CH_DB
    )

    total = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"Total rows = {total}")

    offset = 0
    inserted = 0
    while offset < total:
        rows = con.execute(
            f"""
            SELECT uid, lang, split, dim, embedding
            FROM embeddings
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """
        ).fetchall()

        out = []
        for uid, lang, split, dim, emb in rows:
            if emb is None:
                continue
            v = np.array(emb, dtype=np.float32)
            if v.size != EXPECTED_DIM:
                print(f"skip {uid} wrong dim {v.size}")
                continue
            v = normalize(v)
            out.append((str(uid), str(lang), str(split), int(dim), v.tolist()))

        if out:
            client.insert(
                "embeddings", out, column_names=["id", "lang", "split", "dim", "vector"]
            )
            inserted += len(out)
            print(f"Inserted {inserted}/{total}")

        offset += len(rows)

    print(f"Done. Inserted {inserted}/{total}")


if __name__ == "__main__":
    main()
