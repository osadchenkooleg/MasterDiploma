#!/usr/bin/env python3
"""
Evaluate RAW vs MASKED Jaccard using a stop-list **from ClickHouse** and
pairs loaded **from ClickHouse by default** (no --pairs-sql needed).

By default, reads pairs from table `eval_pairs` with a simple WHERE clause.
You can override table and WHERE via flags.

Expected schema of pairs query result:
  id_a String, id_b String, label Int8/Int16, path_a String, path_b String, lang String (optional)

Examples:
  # Default: table=eval_pairs, where="split='test'"
  python scripts/practice/tools/eval_pairs_ch.py \
    --stop-lang java --stop-n 5 --stop-corpus cleaned_all_splits \
    --ch-host localhost --ch-db codebase --ch-user default --ch-pass 1234 \
    --t-raw 0.15 --t-mask 0.15 --out reports/java_eval.csv

  # Custom table/where and filter to one language
  python scripts/practice/tools/eval_pairs_ch.py \
    --pairs-table eval_pairs --pairs-where "split='test' AND lang='java'" \
    --stop-lang java --stop-n 5 --stop-corpus cleaned_all_splits \
    --ch-host localhost --ch-db codebase --ch-user default --ch-pass 1234 \
    --out reports/java_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set, Tuple

try:
    from clickhouse_connect import get_client
except Exception:
    raise SystemExit("clickhouse-connect is required. pip install clickhouse-connect")

# ---------- logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_pairs")

# ---------- tokenization & shingles ----------
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|==|!=|<=|>=|&&|\|\||[{}();,\[\].:+\-*/]")


def tokenize(code: str) -> List[str]:
    return TOKEN_RE.findall(code)


def shingles(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    L = len(tokens)
    if L < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(L - n + 1)]


# ---------- dataclasses ----------
@dataclass
class Pair:
    id_a: str
    id_b: str
    label: int
    path_a: Path
    path_b: Path
    lang: str | None = None


@dataclass
class Scores:
    raw: float
    masked: float
    label: int


# ---------- core ----------


def jaccard(sa: Set[Tuple[str, ...]], sb: Set[Tuple[str, ...]]) -> float:
    if not sa and not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


# ---------- IO helpers ----------


def ch_client(host: str, user: str, pwd: str, db: str):
    return get_client(host=host, username=user, password=pwd, database=db)


def load_stoplist_from_ch(
    client,
    lang: str,
    n: int,
    corpus_id: str | None,
    version: str | None,
    table: str = "stoplist_shingles",
) -> Set[Tuple[str, ...]]:
    conds = ["lang = %(lang)s", "n = %(n)s"]
    params = {"lang": lang, "n": n}
    if corpus_id:
        conds.append("corpus_id = %(corpus_id)s")
        params["corpus_id"] = corpus_id
    if version:
        conds.append("version = %(version)s")
        params["version"] = version
    sql = f"SELECT shingle FROM {table} WHERE " + " AND ".join(conds)
    rows = client.query(sql, parameters=params).result_rows
    stop = {tuple(r[0]) for r in rows}
    log.info(f"Loaded stoplist from CH: lang={lang} n={n} size={len(stop)}")
    return stop


def build_pairs_sql(table: str, where: str | None) -> str:
    base = f"SELECT id_a,id_b,label,path_a,path_b,lang FROM {table}"
    if where:
        base += f" WHERE {where}"
    return base


def load_pairs_from_ch(client, table: str, where: str | None) -> List[Pair]:
    sql = build_pairs_sql(table, where)
    rows = client.query(sql).result_rows
    out: List[Pair] = []
    for row in rows:
        if len(row) < 5:
            raise ValueError(
                "pairs SQL must return at least 5 columns: id_a,id_b,label,path_a,path_b[,lang]"
            )
        out.append(
            Pair(
                id_a=str(row[0]),
                id_b=str(row[1]),
                label=int(row[2]),
                path_a=Path(str(row[3])).expanduser(),
                path_b=Path(str(row[4])).expanduser(),
                lang=(str(row[5]) if len(row) > 5 and row[5] is not None else None),
            )
        )
    log.info(
        f"Loaded pairs from CH: {len(out)} rows | table={table} where={where or '—'}"
    )
    return out


# ---------- eval ----------


def evaluate(pairs: List[Pair], stop: Set[Tuple[str, ...]], n: int) -> List[Scores]:
    out: List[Scores] = []
    for p in pairs:
        try:
            a = p.path_a.read_text(encoding="utf-8", errors="ignore")
            b = p.path_b.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            out.append(Scores(0.0, 0.0, p.label))
            continue
        sa = set(shingles(tokenize(a), n))
        sb = set(shingles(tokenize(b), n))
        raw = jaccard(sa, sb)
        ma = sa - stop
        mb = sb - stop
        masked = jaccard(ma, mb)
        out.append(Scores(raw, masked, p.label))
    return out


def metrics(
    scores: List[Scores], threshold: float, which: str
) -> Tuple[float, float, float, Tuple[int, int, int, int]]:
    tp = fp = tn = fn = 0
    for s in scores:
        pred = 1 if (s.raw if which == "raw" else s.masked) >= threshold else 0
        if pred == 1 and s.label == 1:
            tp += 1
        elif pred == 1 and s.label == 0:
            fp += 1
        elif pred == 0 and s.label == 0:
            tn += 1
        else:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, (tp, fp, tn, fn)


def write_report(scores: List[Scores], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["raw_jaccard", "masked_jaccard", "label"])
        for s in scores:
            w.writerow([f"{s.raw:.6f}", f"{s.masked:.6f}", s.label])
    log.info(f"Per-pair report written to {out_csv}")


# ---------- CLI ----------


def main():
    ap = argparse.ArgumentParser()

    # Pairs source (defaults to ClickHouse table + where)
    ap.add_argument(
        "--pairs-table",
        type=str,
        default="eval_pairs",
        help="ClickHouse table with pairs",
    )
    ap.add_argument(
        "--pairs-where",
        type=str,
        default="split='test'",
        help="WHERE clause to filter pairs",
    )

    # Stop-list selection in CH
    ap.add_argument("--stop-lang", type=str, required=True)
    ap.add_argument("--stop-n", type=int, default=5)
    ap.add_argument("--stop-corpus", type=str, default=None)
    ap.add_argument("--stop-version", type=str, default=None)
    ap.add_argument("--stop-table", type=str, default="stoplist_shingles")

    # CH connection
    ap.add_argument("--ch-host", type=str, default="localhost")
    ap.add_argument("--ch-user", type=str, default="default")
    ap.add_argument("--ch-pass", type=str, default="")
    ap.add_argument("--ch-db", type=str, default="codebase")

    # Eval params
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--t-raw", type=float, default=0.15)
    ap.add_argument("--t-mask", type=float, default=0.15)
    ap.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()

    client = ch_client(args.ch_host, args.ch_user, args.ch_pass, args.ch_db)
    stop = load_stoplist_from_ch(
        client,
        args.stop_lang,
        args.stop_n,
        args.stop_corpus,
        args.stop_version,
        args.stop_table,
    )

    # Always load pairs from CH using defaults/overrides
    pairs = load_pairs_from_ch(client, args.pairs_table, args.pairs_where)

    scores = evaluate(pairs, stop, args.n)

    for which, t in [("raw", args.t_raw), ("masked", args.t_mask)]:
        p, r, f1, (tp, fp, tn, fn) = metrics(scores, t, which)
        log.info(f"=== {which.upper()} @ t={t:.3f} ===")
        log.info(f"TP={tp} FP={fp} TN={tn} FN={fn}")
        log.info(f"Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

    # Delta FP at given thresholds
    fp_raw = sum(1 for s in scores if (s.raw >= args.t_raw and s.label == 0))
    fp_mask = sum(1 for s in scores if (s.masked >= args.t_mask and s.label == 0))
    log.info(f"ΔFP = FP_masked - FP_raw = {fp_mask} - {fp_raw} = {fp_mask - fp_raw}")

    write_report(scores, args.out)


if __name__ == "__main__":
    main()
