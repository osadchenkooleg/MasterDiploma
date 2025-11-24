#!/usr/bin/env python3
import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from clickhouse_connect import get_client

# ========= ENV / CONFIG =======================================================

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")

MODEL_NAME = os.getenv("EMB_MODEL", "microsoft/codebert-base")
TRANSFORM_VER = int(os.getenv("EMB_TRANSFORM_VER", "3"))  # наш EMB_TRANSFORM_VER=3
METRIC_NAME = os.getenv("EVAL_METRIC", "cosine")  # у practice_scores metric="cosine"

THRESHOLD_TABLE = "threshold_policies"  # у БД codebase
PRACTICE_SCORES_TABLE = "practice_scores"

# Фіксовані пороги в шкалі cosine similarity [0, 1]
# Можеш змінити їх у будь-який момент, наприклад 0.4 / 0.75
FIXED_T_LOW = 0.50
FIXED_T_HIGH = 0.70


# ========= DATA STRUCTURES ====================================================


@dataclass
class ChosenThresholds:
    t_low: float
    t_high: float
    prec_high: float
    rec_high: float
    f1_high: float


# ========= CORE THRESHOLD LOGIC ==============================================


def compute_chosen_thresholds_fixed(
    scores: np.ndarray,
    labels: np.ndarray,
    t_low: float = FIXED_T_LOW,
    t_high: float = FIXED_T_HIGH,
) -> ChosenThresholds:
    """
    Використовуємо фіксовані пороги t_low, t_high (наприклад, 0.50 і 0.70),
    а дані тільки для того, щоб порахувати precision/recall/F1 при t_high.
    """

    if scores.shape[0] == 0:
        raise ValueError("No data provided")

    labels = labels.astype(int)
    total_pos = int(labels.sum())

    # Прогнози для high-зони
    preds_high = scores >= t_high

    tp = int(((preds_high == 1) & (labels == 1)).sum())
    fp = int(((preds_high == 1) & (labels == 0)).sum())
    fn = int(((preds_high == 0) & (labels == 1)).sum())
    # tn можна порахувати для довідки, але він нам не потрібен для метрик, що ми пишемо в таблицю:
    # tn = int(((preds_high == 0) & (labels == 0)).sum())

    support_high = tp + fp

    precision = tp / support_high if support_high > 0 else 0.0
    recall = tp / total_pos if total_pos > 0 else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision > 0 and recall > 0)
        else 0.0
    )

    return ChosenThresholds(
        t_low=t_low,
        t_high=t_high,
        prec_high=precision,
        rec_high=recall,
        f1_high=f1,
    )


# ========= CLICKHOUSE IO ======================================================


def load_scores_for_lang(
    client,
    lang: str,
    model: str,
    metric: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Беремо score/label з practice_scores для:
      - a_lang = b_lang = lang (same-lang),
      - моделі, метрики,
      - normalization='light', boilerplate='off'.
    """
    query = f"""
        SELECT score, label
        FROM {PRACTICE_SCORES_TABLE}
        WHERE model = %(model)s
          AND metric = %(metric)s
          AND normalization = 'light'
          AND boilerplate = 'off'
          AND a_lang = %(lang)s
          AND b_lang = %(lang)s
    """

    result = client.query(
        query,
        parameters={
            "model": model,
            "metric": metric,
            "lang": lang,
        },
    ).result_rows

    if not result:
        raise ValueError(
            f"No rows in {PRACTICE_SCORES_TABLE} for "
            f"model={model}, metric={metric}, lang={lang}, normalization='light', boilerplate='off'"
        )

    scores = np.array([r[0] for r in result], dtype=float)
    labels = np.array([int(r[1]) for r in result], dtype=int)
    return scores, labels


def deactivate_old_policies(
    client,
    model: str,
    transform_ver: int,
    lang: str,
    metric: str,
) -> None:
    """
    Деактивуємо старі політики для (model, transform_ver, lang, metric):
      is_active=0, valid_to=now()
    """
    query = f"""
        ALTER TABLE {THRESHOLD_TABLE}
        UPDATE is_active = 0, valid_to = now()
        WHERE model = %(model)s
          AND transform_ver = %(transform_ver)s
          AND lang = %(lang)s
          AND metric = %(metric)s
          AND is_active = 1
    """
    client.query(
        query,
        parameters={
            "model": model,
            "transform_ver": transform_ver,
            "lang": lang,
            "metric": metric,
        },
    )


def insert_new_policy(
    client,
    model: str,
    transform_ver: int,
    lang: str,
    metric: str,
    thresholds: ChosenThresholds,
    comment: Optional[str] = None,
) -> None:
    """
    Вставляємо новий рядок у threshold_policies.
    Не заповнюємо policy_id/created_at/valid_from/is_active — вони мають DEFAULT.
    """
    data: List[Tuple] = [
        (
            model,
            transform_ver,
            lang,
            metric,
            thresholds.t_low,
            thresholds.t_high,
            thresholds.prec_high,
            thresholds.rec_high,
            thresholds.f1_high,
            comment,
        )
    ]

    client.insert(
        THRESHOLD_TABLE,
        data,
        column_names=[
            "model",
            "transform_ver",
            "lang",
            "metric",
            "t_low",
            "t_high",
            "target_precision_high_zone",
            "target_recall_high_zone",
            "f1_at_t_high",
            "policy_comment",
        ],
    )


# ========= CLI ================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute fixed t_low/t_high from practice_scores (all languages) and store into threshold_policies."
    )

    # Можна перевизначити список мов, але за замовчуванням java,js,go,python
    parser.add_argument(
        "--langs",
        default="java,js,go,python",
        help="Comma-separated list of languages to process (default: java,js,go,python)",
    )

    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--transform-ver", type=int, default=TRANSFORM_VER)
    parser.add_argument("--metric", default=METRIC_NAME)

    parser.add_argument(
        "--comment",
        default=None,
        help="Optional comment for threshold_policies (applied to all langs).",
    )

    # ClickHouse override (без порта – використовується дефолтний HTTP-порт 8123)
    parser.add_argument("--ch-host", default=CH_HOST)
    parser.add_argument("--ch-user", default=CH_USER)
    parser.add_argument("--ch-pass", default=CH_PASS)
    parser.add_argument("--ch-db", default=CH_DB)

    return parser.parse_args()


def main():
    args = parse_args()
    langs = [l.strip() for l in args.langs.split(",") if l.strip()]

    print(f"[INFO] Connecting to ClickHouse host={args.ch_host}, db={args.ch_db}")
    client = get_client(
        host=args.ch_host,
        username=args.ch_user,
        password=args.ch_pass,
        database=args.ch_db,
    )

    base_comment = args.comment or (
        f"Fixed thresholds t_low={FIXED_T_LOW}, t_high={FIXED_T_HIGH} "
        f"from practice_scores (normalization=light, boilerplate=off)"
    )

    for lang in langs:
        print("\n==============================")
        print(f"[LANG] {lang}")
        print("==============================")

        try:
            scores, labels = load_scores_for_lang(
                client,
                lang=lang,
                model=args.model,
                metric=args.metric,
            )
        except ValueError as e:
            print(f"[WARN] {e}")
            continue

        print(
            f"[INFO] Loaded {len(scores)} examples for lang={lang}. "
            f"Positives={int(labels.sum())}, "
            f"Negatives={len(scores) - int(labels.sum())}"
        )

        print("[INFO] Computing thresholds (fixed values)...")
        chosen = compute_chosen_thresholds_fixed(scores, labels)

        print(f"[RESULT] lang={lang} t_low  = {chosen.t_low:.2f}")
        print(f"[RESULT] lang={lang} t_high = {chosen.t_high:.2f}")
        print(
            f"[RESULT] lang={lang} high-zone precision={chosen.prec_high:.4f}, "
            f"recall={chosen.rec_high:.4f}, F1={chosen.f1_high:.4f}"
        )

        comment = f"{base_comment}; lang={lang}"

        print("[INFO] Deactivating old policies in threshold_policies...")
        deactivate_old_policies(
            client,
            model=args.model,
            transform_ver=args.transform_ver,
            lang=lang,
            metric=args.metric,
        )

        print("[INFO] Inserting new policy row...")
        insert_new_policy(
            client,
            model=args.model,
            transform_ver=args.transform_ver,
            lang=lang,
            metric=args.metric,
            thresholds=chosen,
            comment=comment,
        )

        print(f"[DONE] Policy stored for lang={lang}")

    print("\n[ALL DONE] Threshold policies updated for languages:", ", ".join(langs))


if __name__ == "__main__":
    main()
