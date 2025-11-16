#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import List, Tuple

import numpy as np
from clickhouse_connect import get_client

from app.domain.light_normalize import normalize_code
from app.infrastructure.embeddings.model_codebert import CodeEmbeddingModel

# ===============================
# Конфігурація через ENV
# ===============================
CH_HOST = os.getenv("CH_HOST", "127.0.0.1")
CH_PORT = int(os.getenv("CH_PORT", "8123"))  # HTTP порт за замовчуванням
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")
CH_IFACE = os.getenv("CH_IFACE", "http")  # лишаємо для сумісності

EMB_MODEL = os.getenv("EMB_MODEL", "microsoft/codebert-base")
EMB_POOL = os.getenv("EMB_POOL", "mean")  # фіксуємо як 'mean'
EMB_BATCH = int(os.getenv("EMB_BATCH", "128"))
EMB_LANGS = [
    x.strip()
    for x in os.getenv("EMB_LANGS", "javascript,go,python").split(",")
    if x.strip()
]
EMB_SPLITS = [
    x.strip() for x in os.getenv("EMB_SPLITS", "validation").split(",") if x.strip()
]
EMB_LIMIT = int(os.getenv("EMB_LIMIT", "0"))  # 0 = без ліміту

# Базова колонка тексту з ClickHouse
USE_NORMALIZED_TEXT = (
    os.getenv("EMB_USE_NORMALIZED", "0") == "1"
)  # 1 = брати code_norm, 0 = code
# Додаткова «легка» нормалізація перед токенізацією (на льоту)
CODE_NORM = os.getenv("EMB_CODE_NORM", "raw").lower()  # 'raw' | 'light'

# Якщо потрібно перезаписати існуючі ембеддинги для профілю (lang, model, pool, tver)
REPLACE_EXISTING = os.getenv("EMB_REPLACE", "0") == "1"

TABLE_CODES = "practice_codes"
TABLE_EMBS = "practice_embeddings"


# ===============================
# Обчислення transform_ver
# ===============================
def pick_transform_ver(code_norm: str) -> int:
    """
    2 = звичайний пайплайн (raw текст або code_norm з БД без додаткової "light" нормалізації)
    3 = додатково застосовано легкий нормалізатор (strip comments, whitespaces, id/literals)
    """
    return 3 if code_norm == "light" else 2


# Дозволяємо перевизначити через ENV (для відтворюваності експериментів)
EMB_TVER = int(os.getenv("EMB_TRANSFORM_VER", str(pick_transform_ver(CODE_NORM))))

# ===============================
# SQL-запити
# ===============================
SQL_SELECT_CODES = f"""
SELECT uid, {{col}}
FROM {TABLE_CODES}
WHERE lang = %(lang)s
  AND split IN %(splits)s
ORDER BY uid
{{limit_clause}}
"""

SQL_DELETE_EXISTING = f"""
ALTER TABLE {TABLE_EMBS}
DELETE WHERE lang = %(lang)s AND model=%(m)s AND pool=%(p)s AND transform_ver=%(tv)s
"""


# ===============================
# Допоміжні функції
# ===============================
def get_ch_client():
    """
    ClickHouse HTTP клієнт з розширеними таймаутами для великих вставок.
    """
    return get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASS,
        database=CH_DB,
        interface=CH_IFACE,
        connect_timeout=30,
        send_receive_timeout=1800,  # до 30 хв для великих батчів
    )


def fetch_rows(
    client, lang: str, splits: List[str], limit: int, use_code_norm_col: bool
) -> List[Tuple[str, str]]:
    """
    Завантажити (uid, текст) для вказаної мови та сплітів.
    Текст беремо з колонки 'code_norm' або 'code' (залежно від EMB_USE_NORMALIZED),
    а додаткова light-нормалізація застосовується пізніше, перед токенізацією.
    """
    col = "code_norm" if use_code_norm_col else "code"
    limit_clause = "" if limit <= 0 else f"LIMIT {limit}"
    q = SQL_SELECT_CODES.format(col=col, limit_clause=limit_clause)
    res = client.query(
        q,
        parameters={"lang": lang, "splits": tuple(splits)},
        settings={"max_execution_time": 0},
    )
    return res.result_rows  # [(uid, text), ...]


def insert_embeddings(client, rows):
    """
    rows: List[Tuple[lang, uid, model, pool, transform_ver, dim, vec(List[Float32])]]
    """
    if not rows:
        return
    client.insert(
        TABLE_EMBS,
        rows,
        column_names=["lang", "uid", "model", "pool", "transform_ver", "dim", "vec"],
    )


def maybe_delete_existing(client, lang: str):
    if not REPLACE_EXISTING:
        return
    print(
        f"[{lang}] Deleting existing embeddings for model={EMB_MODEL}, pool={EMB_POOL}, ver={EMB_TVER}"
    )
    client.command(
        SQL_DELETE_EXISTING + " SETTINGS mutations_sync=1",
        parameters={"lang": lang, "m": EMB_MODEL, "p": EMB_POOL, "tv": EMB_TVER},
    )


def apply_pre_token_norm(text: str, lang: str) -> str:
    """
    Додаткова легка нормалізація перед токенізацією (EMB_CODE_NORM='light').
    Якщо 'raw' — повертаємо як є.
    """
    if CODE_NORM == "light":
        return normalize_code(text or "", lang)
    return text or ""


# ===============================
# Основний сценарій
# ===============================
def main():
    # 1) ClickHouse
    client = get_ch_client()

    # 2) Модель
    model = CodeEmbeddingModel(model_name=EMB_MODEL)
    dim = model.encode("dummy").shape[0]
    print(
        f"Using transform_ver={EMB_TVER} (CODE_NORM={CODE_NORM}), base_col={'code_norm' if USE_NORMALIZED_TEXT else 'code'}"
    )
    print(f"Embedding dim = {dim}")

    # 3) Ітерація по мовах
    for lang in EMB_LANGS:
        maybe_delete_existing(client, lang)

        rows = fetch_rows(client, lang, EMB_SPLITS, EMB_LIMIT, USE_NORMALIZED_TEXT)
        total = len(rows)
        if total == 0:
            print(f"[{lang}] No rows found (splits={EMB_SPLITS}), skip")
            continue

        print(
            f"[{lang}] {total} snippets to embed | splits={EMB_SPLITS} | base_norm_col={'code_norm' if USE_NORMALIZED_TEXT else 'code'} | light_norm={CODE_NORM=='light'}"
        )
        inserted = 0
        buf = []

        # 4) Батч-енкодинг
        for i in range(0, total, EMB_BATCH):
            batch = rows[i : i + EMB_BATCH]
            uids = [u for (u, _) in batch]
            texts = [(t if isinstance(t, str) else "") for (_, t) in batch]

            # Легка нормалізація за потреби
            if CODE_NORM == "light":
                texts = [apply_pre_token_norm(t, lang) for t in texts]

            # Обережний батчинг: один за одним, щоб не з'їсти пам'ять (можна замінити на власний батч-encode)
            vecs: List[np.ndarray] = []
            for t in texts:
                vecs.append(model.encode(t))

            # Пакуємо до буфера
            for uid, v in zip(uids, vecs):
                # ClickHouse очікує масив Float32 — конвертуємо
                buf.append(
                    (
                        lang,
                        uid,
                        EMB_MODEL,
                        EMB_POOL,
                        EMB_TVER,
                        dim,
                        [float(x) for x in v.astype("float32").tolist()],
                    )
                )

            # Вставляємо частинами, щоб швидше бачити прогрес
            if len(buf) >= 1000 or i + EMB_BATCH >= total:
                insert_embeddings(client, buf)
                inserted += len(buf)
                print(f"[{lang}] inserted: {inserted}/{total}")
                buf.clear()

    print("Done.")


# ===============================
if __name__ == "__main__":
    main()
