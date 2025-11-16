#!/usr/bin/env python3
import os
import uuid
from typing import Dict, Tuple

import numpy as np
from clickhouse_connect import get_client
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_PORT = int(os.getenv("CH_PORT", "9000"))
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")

MODEL_NAME = os.getenv("EMB_MODEL", "microsoft/codebert-base")
POOL = os.getenv("EMB_POOL", "mean")
TRANSFORM_VER = int(os.getenv("EMB_TRANSFORM_VER", "1"))
SPLIT = os.getenv("EVAL_SPLIT", "valid")  # див. practice_pairs.split


def fetch_pairs(client):
    rows = client.query(
        """
        SELECT pair_id, a_lang, a_uid, b_lang, b_uid, label
        FROM practice_pairs
        WHERE split = %(split)s
    """,
        parameters={"split": SPLIT},
    ).result_rows
    return rows


def fetch_vecs(client, lang: str, uids: Tuple[str]):
    if not uids:
        return {}
    rows = client.query(
        """
        SELECT uid, vec
        FROM practice_embeddings
        WHERE lang = %(lang)s AND model=%(m)s AND pool=%(p)s AND transform_ver=%(tv)s
          AND uid IN %(uids)s
    """,
        parameters={
            "lang": lang,
            "m": MODEL_NAME,
            "p": POOL,
            "tv": TRANSFORM_VER,
            "uids": tuple(uids),
        },
    ).result_rows
    out = {}
    for uid, vec in rows:
        out[uid] = np.asarray(vec, dtype=np.float32)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    # захист від нульових векторів
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    client = get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASS, database=CH_DB
    )
    pairs = fetch_pairs(client)
    if not pairs:
        print("No pairs in practice_pairs for split=", SPLIT)
        return

    # Підтягнемо ембеддинги порціями по мові (щоб не робити N запитів)
    # 1) згрупуємо uids по мовах
    by_lang_a, by_lang_b = {}, {}
    for pair_id, a_lang, a_uid, b_lang, b_uid, label in pairs:
        by_lang_a.setdefault(a_lang, set()).add(a_uid)
        by_lang_b.setdefault(b_lang, set()).add(b_uid)

    cache: Dict[Tuple[str, str], np.ndarray] = {}

    # завантажимо для кожної мови всі потрібні uid
    for lang, uids in by_lang_a.items():
        cache.update(
            {(lang, uid): v for uid, v in fetch_vecs(client, lang, tuple(uids)).items()}
        )
    for lang, uids in by_lang_b.items():
        cache.update(
            {(lang, uid): v for uid, v in fetch_vecs(client, lang, tuple(uids)).items()}
        )

    y_true, y_score = [], []
    scored_rows = []
    for pair_id, a_lang, a_uid, b_lang, b_uid, label in pairs:
        va = cache.get((a_lang, a_uid))
        vb = cache.get((b_lang, b_uid))
        if va is None or vb is None:
            # пропускаємо пари без векторів (нема ембеддинга)
            continue
        s = cosine(va, vb)
        y_true.append(int(label))
        y_score.append(s)
        scored_rows.append((pair_id, a_lang, a_uid, b_lang, b_uid, int(label), s))

    if not y_true:
        print("No scored pairs (check embeddings present for these uids).")
        return

    # Метрики
    roc = roc_auc_score(y_true, y_score)
    pr = average_precision_score(y_true, y_score)  # PR-AUC
    # Вибір порогу за F1 на PR-кривій
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision/recall мають на 1 елемент більше за thresholds; узгодимо
    best_f1 = -1.0
    best_t = 0.5
    best_p = 0.0
    best_r = 0.0
    for p, r, t in zip(precision[:-1], recall[:-1], thresholds):
        f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
        if f1 > best_f1:
            best_f1, best_t, best_p, best_r = f1, float(t), float(p), float(r)

    run_id = str(uuid.uuid4())

    # Вставимо оцінки пар у practice_scores
    data = []
    for pair_id, a_lang, a_uid, b_lang, b_uid, label, s in scored_rows:
        decision = "Plagiarism" if s >= best_t else "OK"
        data.append(
            (
                run_id,
                MODEL_NAME,
                "light",
                "off",
                "cosine",
                best_t,
                pair_id,
                a_lang,
                a_uid,
                b_lang,
                b_uid,
                label,
                s,
                decision,
            )
        )
    client.insert(
        "practice_scores",
        data,
        column_names=[
            "run_id",
            "model",
            "normalization",
            "boilerplate",
            "metric",
            "threshold",
            "pair_id",
            "a_lang",
            "a_uid",
            "b_lang",
            "b_uid",
            "label",
            "score",
            "decision",
        ],
    )

    # І сам поріг + агреговані метрики у practice_thresholds
    client.insert(
        "practice_thresholds",
        [
            (
                run_id,
                MODEL_NAME,
                "mean",
                1,
                "valid",
                "cosine",
                best_t,
                roc,
                pr,
                best_p,
                best_r,
                best_f1,
            )
        ],
        column_names=[
            "run_id",
            "model",
            "pool",
            "transform_ver",
            "split",
            "metric",
            "threshold",
            "roc_auc",
            "pr_auc",
            "precision_at_t",
            "recall_at_t",
            "f1_at_t",
        ],
    )

    print(f"RUN {run_id}")
    print(f"ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")
    print(
        f"Best threshold T={best_t:.4f}  P={best_p:.3f}  R={best_r:.3f}  F1={best_f1:.3f}"
    )
    print("Done.")


if __name__ == "__main__":
    main()
