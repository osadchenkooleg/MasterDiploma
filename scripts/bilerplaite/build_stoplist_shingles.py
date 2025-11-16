#!/usr/bin/env python3
"""
build_stoplist_shingles.py

Офлайн-скрипт для підрахунку document frequency шинглів
по корпусу codebase.codes_v4 і запису результату
в таблицю codebase.stoplist_shingles.

Кроки:
- читаємо codes_v4 батчами по code_id
- normalize_code(code, lang)
- будуємо шингли (k-грамми по токенах)
- рахуємо df для кожного шингла (по мові)
"""

import argparse
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Set, Tuple

from clickhouse_connect import get_client

CH_HOST = os.getenv("CH_HOST", "localhost")
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")

CODES_TABLE = "codebase.codes_v4"
STOPLIST_TABLE = "codebase.stoplist_shingles"

# Мови, які нас цікавлять
LANGS = ["java", "js", "go", "python"]

# Розмір шингла (k токенів)
SHINGLE_K = 5

# Розмір батча читання з ClickHouse
BATCH_SIZE = 10_000

#!/usr/bin/env python3

KW_JS = set(
    "break case catch class const continue debugger default delete do else export extends finally for function if import in instanceof let new return super switch this throw try typeof var void while with yield await async".split()
)
KW_GO = set(
    "break default func interface select case defer go map struct chan else goto package switch const fallthrough if range type continue for import return var".split()
)
KW_PY = set(
    "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split()
)

ID_RE = re.compile(r"\b[_A-Za-z][_A-Za-z0-9]*\b")
NUM_RE = re.compile(r"\b\d+(\.\d+)?\b")
# дуже просте виділення рядків: "…", '…' (без екранувань усередині — достатньо для light)
STR_RE = re.compile(r"(\"[^\n\"\\]*\"|'[^\n'\\]*')")


def strip_comments(code: str, lang: str) -> str:
    s = code
    if lang in ("javascript", "go"):
        s = re.sub(r"//[^\n]*", "", s)
        s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    if lang == "python":
        s = re.sub(r"#.*?$", "", s, flags=re.M)
    return s


def keywords(lang: str) -> Set[str]:
    return {"javascript": KW_JS, "go": KW_GO, "python": KW_PY}.get(lang, set())


def normalize_whitespace(s: str) -> str:
    # прибираємо порожні, стискаємо пробіли/переноси
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    one = " ".join(lines)
    return re.sub(r"\s+", " ", one).strip()


def replace_literals(s: str) -> str:
    # спочатку рядки, потім числа (щоб не чіпати <STR>)
    s = STR_RE.sub("<STR>", s)
    s = NUM_RE.sub("<NUM>", s)
    return s


def replace_identifiers(s: str, lang: str) -> str:
    kws = keywords(lang)
    mapping: Dict[str, str] = {}
    cnt = 1

    def repl(m: re.Match):
        nonlocal cnt
        tok = m.group(0)
        if tok in kws or tok in ("self", "this"):
            return tok
        # не замінюємо спеціальні маркери
        if tok in ("<STR>", "<NUM>"):
            return tok
        if tok.isupper() and len(tok) > 1:  # константи типу MAX_LEN залишимо
            return tok
        if tok not in mapping:
            mapping[tok] = f"v{cnt}"
            cnt += 1
        return mapping[tok]

    return ID_RE.sub(repl, s)


def normalize_code(code: str, lang: str) -> str:
    # порядок важливий: коментарі -> пробіли -> літерали -> ідентифікатори -> фінальне стиснення
    s = strip_comments(code or "", lang)
    s = normalize_whitespace(s)
    s = replace_literals(s)
    s = replace_identifiers(s, lang)
    return normalize_whitespace(s)


def map_lang_for_normalizer(lang: str) -> str:
    """
    В light_normalize, скоріше за все, використовується:
    - 'javascript' замість 'js'
    - 'python' (ok)
    - 'go' (ok)
    - 'java' (ok)
    """
    lang = (lang or "").lower()
    if lang == "js":
        return "javascript"
    if lang in ("py", "python"):
        return "python"
    return lang


def make_shingles(tokens: List[str], k: int) -> Iterable[str]:
    """Повертає всі k-грамми (шингли) як рядки 'tok_i ... tok_{i+k-1}'."""
    n = len(tokens)
    if n < k:
        return []
    return (" ".join(tokens[i : i + k]) for i in range(n - k + 1))


def process_lang(client, lang: str, batch_size: int, shingle_k: int) -> None:
    """
    Пройти по всіх кодах для даної мови, порахувати df-шинглів
    і зберегти в STOPLIST_TABLE.
    """
    print(f"=== Processing lang={lang} ===")
    norm_lang = map_lang_for_normalizer(lang)

    # document frequency для (lang, shingle)
    df_counter: Counter[str] = Counter()

    last_code_id = ""  # для "сканування" по ORDER BY code_id

    total_rows = 0
    while True:
        rows: List[Tuple[str, str]] = client.query(
            f"""
            SELECT code_id, code
            FROM {CODES_TABLE}
            WHERE lang = %(lang)s
              AND code_id > %(last_id)s
            ORDER BY code_id
            LIMIT %(limit)s
            """,
            parameters={"lang": lang, "last_id": last_code_id, "limit": batch_size},
        ).result_rows

        if not rows:
            break

        for code_id, code in rows:
            if not code:
                continue

            # 1) normalize
            norm = normalize_code(code, norm_lang)

            # 2) токенізація
            tokens = norm.split()
            if len(tokens) < shingle_k:
                continue

            # 3) унікальні шингли для цього документа
            unique_shingles = set(make_shingles(tokens, shingle_k))

            # 4) оновлюємо df
            for sh in unique_shingles:
                df_counter[sh] += 1

            total_rows += 1
            if total_rows % 10_000 == 0:
                print(f"  processed {total_rows} code rows for lang={lang}")

        last_code_id = rows[-1][0]

    print(
        f"Finished lang={lang}: processed {total_rows} code rows, "
        f"unique shingles = {len(df_counter)}"
    )

    if not df_counter:
        print(f"No shingles for lang={lang}, skipping insert.")
        return

    # Готуємо дані для вставки в ClickHouse
    # (lang, shingle, df)
    rows_to_insert = [(lang, shingle, int(df)) for shingle, df in df_counter.items()]

    # На всяк випадок можна очистити попередні дані для цієї мови
    client.command(
        f"ALTER TABLE {STOPLIST_TABLE} DELETE WHERE lang = %(lang)s",
        parameters={"lang": lang},
    )

    print(
        f"Inserting {len(rows_to_insert)} shingles into {STOPLIST_TABLE} for lang={lang}..."
    )
    client.insert(
        STOPLIST_TABLE,
        rows_to_insert,
        column_names=["lang", "shingle", "df"],
    )
    print(f"Done inserting for lang={lang}.")


def main():
    parser = argparse.ArgumentParser(
        description="Build DF statistics for code shingles and store in ClickHouse."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size for reading codes_v4 (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=SHINGLE_K,
        help=f"Shingle size k (default: {SHINGLE_K})",
    )
    parser.add_argument(
        "--langs",
        type=str,
        default=",".join(LANGS),
        help=f"Comma-separated languages to process (default: {','.join(LANGS)})",
    )
    args = parser.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    client = get_client(
        host=CH_HOST, username=CH_USER, password=CH_PASS, database=CH_DB
    )

    for lang in langs:
        process_lang(client, lang, args.batch_size, args.k)


if __name__ == "__main__":
    main()
