#!/usr/bin/env python3

# базово (корпус із bench_topk_vectors, виключаємо self)
# python3 scripts/practice/benchmark/benchmark_topk_ch.py \
#   --queries 300 --k 10 --ef 200 --exclude-self
#
# більша якість (еф вище) — трохи повільніше
# python3 scripts/practice/benchmark/benchmark_topk_ch.py \
#   --queries 300 --k 10 --ef 400 --exclude-self
#
# інший корпус через WHERE і ліміт
# python3 scripts/practice/benchmark/benchmark_topk_ch.py \
#   --where "lang IN ['go','java'] AND split='train'" \
#   --limit 10000 --queries 300 --k 10 --ef 300 --exclude-self


from __future__ import annotations

import argparse
import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from app.infrastructure.db.clickhouse.client import get_ch_client

table = "codebase.bench_topk_results"
corpus = "codebase.bench_topk_vectors"


# --- helpers ---
def l2_normalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return (X / n).astype("float32")


def cosine_topk_exact(
    q: np.ndarray, X: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    sims = X @ q  # X і q — L2-нормовані
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]


def recall_at_k(exact_idx: np.ndarray, approx_idx: np.ndarray) -> float:
    return len(set(exact_idx.tolist()) & set(approx_idx.tolist())) / float(
        len(exact_idx)
    )


def p50_p95_ms(values_ms: List[float]) -> Tuple[float, float, float]:
    arr = np.array(values_ms, dtype=np.float64)
    return (
        float(arr.mean()),
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
    )


def kendall_tau_topk(order_a: List[int], order_b: List[int]) -> Optional[float]:
    inter = [i for i in order_a if i in set(order_b)]
    if len(inter) < 2:
        return None
    pos_a = {v: i for i, v in enumerate(order_a)}
    pos_b = {v: i for i, v in enumerate(order_b)}
    concord = discord = 0
    for i in range(len(inter)):
        for j in range(i + 1, len(inter)):
            a_i, a_j = inter[i], inter[j]
            s_a = 1 if (pos_a[a_j] - pos_a[a_i]) > 0 else -1
            s_b = 1 if (pos_b[a_j] - pos_b[a_i]) > 0 else -1
            concord += s_a == s_b
            discord += s_a != s_b
    if concord + discord == 0:
        return None
    return (concord - discord) / (concord + discord)


# --- ClickHouse load ---
def load_bench_vectors(
    table="codebase.bench_topk_vectors",
    where: Optional[str] = None,
    limit: Optional[int] = None,
):
    client = get_ch_client()
    sql = f"""
      SELECT code_id, vector
      FROM {table}
      {"WHERE " + where if where else ""}
      {"LIMIT " + str(int(limit)) if limit else ""}
    """
    rows = client.query(sql).result_rows
    if not rows:
        raise RuntimeError("No rows fetched from ClickHouse. Check table/WHERE/limit.")
    ids = [r[0] for r in rows]
    vecs = np.stack([np.array(r[1], dtype="float32") for r in rows])
    vecs = l2_normalize(vecs)
    return vecs, ids


def load_query_ids(
    table="codebase.bench_topk_queries", limit: Optional[int] = None
) -> Optional[List[str]]:
    client = get_ch_client()
    sql = f"SELECT code_id FROM {table} ORDER BY code_id"
    if limit:
        sql += f" LIMIT {int(limit)}"
    try:
        rows = client.query(sql).result_rows
        return [r[0] for r in rows]
    except Exception:
        return None


# --- HNSW backend ---
def build_hnsw(
    X: np.ndarray, M: int = 16, ef_construction: int = 200, ef_search: int = 200
):
    import hnswlib

    N, D = X.shape
    index = hnswlib.Index(space="cosine", dim=D)  # cosine на L2-нормованих
    t0 = time.perf_counter()
    index.init_index(max_elements=N, ef_construction=ef_construction, M=M)
    index.add_items(X, np.arange(N, dtype=np.int32))
    index.set_ef(ef_search)  # параметр пошуку (якість/швидкість)
    build_ms = (time.perf_counter() - t0) * 1000.0
    return index, build_ms


def run_benchmark_hnsw(
    X: np.ndarray,
    ids: List[str],
    k: int = 10,
    n_queries: int = 300,
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 200,
    query_id_list: Optional[List[str]] = None,
    exclude_self: bool = True,
    seed: int = 42,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    N, D = X.shape
    hnsw, build_ms = build_hnsw(
        X, M=M, ef_construction=ef_construction, ef_search=ef_search
    )

    # prepare queries
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    if query_id_list:
        q_idx = [id_to_idx[cid] for cid in query_id_list if cid in id_to_idx][
            :n_queries
        ]
        if not q_idx:
            q_idx = np.random.choice(N, size=min(n_queries, N), replace=False).tolist()
    else:
        q_idx = np.random.choice(N, size=min(n_queries, N), replace=False).tolist()

    lat_exact, lat_hnsw = [], []
    recalls, taus = [], []

    for qi in tqdm(q_idx, desc="Queries"):
        q = X[qi]

        # exact cosine
        t0 = time.perf_counter()
        exact_idx, _ = cosine_topk_exact(q, X, k + (1 if exclude_self else 0))
        if exclude_self:
            exact_idx = exact_idx[exact_idx != qi][:k]
        else:
            exact_idx = exact_idx[:k]
        lat_exact.append((time.perf_counter() - t0) * 1000.0)

        # hnsw
        t0 = time.perf_counter()
        labels, dists = hnsw.knn_query(q, k=k + (1 if exclude_self else 0))
        approx_idx = labels[0].tolist()
        if exclude_self:
            approx_idx = [x for x in approx_idx if x != qi][:k]
        else:
            approx_idx = approx_idx[:k]
        lat_hnsw.append((time.perf_counter() - t0) * 1000.0)

        # quality
        rec = recall_at_k(exact_idx, np.array(approx_idx))
        recalls.append(rec)
        tau = kendall_tau_topk(exact_idx.tolist(), approx_idx)
        if tau is not None:
            taus.append(tau)

    exact_mean, exact_p50, exact_p95 = p50_p95_ms(lat_exact)
    hnsw_mean, hnsw_p50, hnsw_p95 = p50_p95_ms(lat_hnsw)

    return {
        "N": N,
        "D": D,
        "k": k,
        "n_queries": len(q_idx),
        "hnsw": {
            "M": M,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "build_ms": round(build_ms, 1),
        },
        "latency_ms": {
            "exact": {
                "mean": round(exact_mean, 2),
                "p50": round(exact_p50, 2),
                "p95": round(exact_p95, 2),
            },
            "hnsw": {
                "mean": round(hnsw_mean, 2),
                "p50": round(hnsw_p50, 2),
                "p95": round(hnsw_p95, 2),
            },
        },
        "quality": {
            "recall_at_k_mean": round(float(np.mean(recalls)), 4),
            "recall_at_k_p50": round(float(np.percentile(recalls, 50)), 4),
            "recall_at_k_p95": round(float(np.percentile(recalls, 95)), 4),
            "kendall_tau_mean": round(float(np.mean(taus)), 4) if len(taus) else None,
        },
    }


def parse_args():
    ap = argparse.ArgumentParser(
        description="Brute-force vs HNSW (ClickHouse corpus via project client)"
    )
    ap.add_argument("--table", default="codebase.bench_topk_vectors")
    ap.add_argument("--where", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--query-table", default="codebase.bench_topk_queries")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    # HNSW knobs
    ap.add_argument(
        "--M",
        type=int,
        default=16,
        help="graph connectivity (higher -> better recall, slower build)",
    )
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument(
        "--ef",
        type=int,
        default=200,
        help="ef at search time (higher -> better recall, slower search)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude-self",
        action="store_true",
        help="exclude the query itself from top-k",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    X, ids = load_bench_vectors(table=args.table, where=args.where, limit=args.limit)
    qids = load_query_ids(table=args.query_table, limit=args.queries)

    # діагностика
    norms = np.linalg.norm(X, axis=1)
    print(
        f"[DIAG] N={len(X)} D={X.shape[1]}  zero_norm={(norms<1e-8).sum()}  min_norm={norms.min():.3g} max_norm={norms.max():.3g} mean_norm={norms.mean():.3g}"
    )

    res = run_benchmark_hnsw(
        X,
        ids,
        k=args.k,
        n_queries=args.queries,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=args.ef,
        query_id_list=qids,
        exclude_self=args.exclude_self,
        seed=args.seed,
    )

    ch = get_ch_client()

    # exact row
    ch.query(
        """
    INSERT INTO codebase.bench_topk_results
    (engine, corpus_table, n, d, k, queries, params, p50_ms, p95_ms, mean_ms,
     recall_mean, recall_p50, recall_p95, tau_mean)
    VALUES
    ('exact', %(corpus)s, %(N)s, %(D)s, %(k)s, %(Q)s, %(params)s, %(p50)s, %(p95)s, %(mean)s,
     NULL, NULL, NULL, NULL)
    """,
        parameters={
            "corpus": corpus,
            "N": res["N"],
            "D": res["D"],
            "k": res["k"],
            "Q": res["n_queries"],
            "params": {"note": "brute_force_cosine"},
            "p50": res["latency_ms"]["exact"]["p50"],
            "p95": res["latency_ms"]["exact"]["p95"],
            "mean": res["latency_ms"]["exact"]["mean"],
        },
    )

    # hnsw row
    ch.query(
        """
    INSERT INTO codebase.bench_topk_results
    (engine, corpus_table, n, d, k, queries, params, p50_ms, p95_ms, mean_ms,
     recall_mean, recall_p50, recall_p95, tau_mean)
    VALUES
    ('hnsw', %(corpus)s, %(N)s, %(D)s, %(k)s, %(Q)s, %(params)s, %(p50)s, %(p95)s, %(mean)s,
     %(rmean)s, %(rp50)s, %(rp95)s, %(tau)s)
    """,
        parameters={
            "corpus": corpus,
            "N": res["N"],
            "D": res["D"],
            "k": res["k"],
            "Q": res["n_queries"],
            "params": res[
                "hnsw"
            ],  # {"M":..,"ef_construction":..,"ef_search":..,"build_ms":..}
            "p50": res["latency_ms"]["hnsw"]["p50"],
            "p95": res["latency_ms"]["hnsw"]["p95"],
            "mean": res["latency_ms"]["hnsw"]["mean"],
            "rmean": res["quality"]["recall_at_k_mean"],
            "rp50": res["quality"]["recall_at_k_p50"],
            "rp95": res["quality"]["recall_at_k_p95"],
            "tau": res["quality"]["kendall_tau_mean"],
        },
    )

    import json

    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
