#!/usr/bin/env python3
# Цей скрипт бере вибірку з practice_codes (train+validation), перейменовує ідентифікатори (оминає ключові слова), і пише нові рядки назад у practice_codes зі split='aug' та uid з суфіксом
import os
import random
import re
from typing import Dict, Set

from clickhouse_connect import get_client

CH_HOST = os.getenv("CH_HOST", "127.0.0.1")
CH_PORT = int(os.getenv("CH_PORT", "8123"))
CH_USER = os.getenv("CH_USER", "default")
CH_PASS = os.getenv("CH_PASS", "1234")
CH_DB = os.getenv("CH_DB", "codebase")
LANGS = os.getenv("AUG_LANGS", "javascript,go,python").split(",")
PER_LANG = int(
    os.getenv("AUG_PER_LANG", "300")
)  # скільки оригіналів на мову обфускувати
SUFFIX = os.getenv("AUG_SUFFIX", "#aug1")

# дуже короткі списки ключових слів; за потреби розшириш
KW_JS = set(
    "break case catch class const continue debugger default delete do else export extends finally for function if import in instanceof let new return super switch this throw try typeof var void while with yield await async".split()
)
KW_GO = set(
    "break default func interface select case defer go map struct chan else goto package switch const fallthrough if range type continue for import return var".split()
)
KW_PY = set(
    "False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield".split()
)


def keywords(lang: str) -> Set[str]:
    return {"javascript": KW_JS, "go": KW_GO, "python": KW_PY}.get(lang, set())


ID_RE = re.compile(r"\b[_A-Za-z][_A-Za-z0-9]*\b")


def obfuscate(lang: str, code: str) -> str:
    kws = keywords(lang)
    mapping: Dict[str, str] = {}
    counter = 1

    def repl(m):
        nonlocal counter
        tok = m.group(0)
        if tok in kws:
            return tok
        if tok.isupper() and len(tok) > 1:  # константи/макроси — оминаємо
            return tok
        if tok in ("self", "this"):  # спец-ідентифікатори
            return tok
        if tok not in mapping:
            mapping[tok] = f"v{counter}"
            counter += 1
        return mapping[tok]

    return ID_RE.sub(repl, code)


def main():
    client = get_client(
        host=CH_HOST, port=CH_PORT, username=CH_USER, password=CH_PASS, database=CH_DB
    )
    for lang in LANGS:
        # вибірка оригіналів
        rows = client.query(
            """
            SELECT uid, code FROM practice_codes
            WHERE lang=%(l)s AND split IN ('train','validation')
            ORDER BY rand()
            LIMIT %(n)s
        """,
            parameters={"l": lang, "n": PER_LANG},
        ).result_rows
        if not rows:
            print(f"[{lang}] no source rows")
            continue
        to_ins = []
        for uid, code in rows:
            aug_uid = f"{uid}{SUFFIX}"
            aug_code = obfuscate(lang, code or "")
            to_ins.append((lang, "aug", aug_uid, aug_code))
        # вставляємо нові рядки
        client.insert(
            "practice_codes", to_ins, column_names=["lang", "split", "uid", "code"]
        )
        print(f"[{lang}] augmented {len(to_ins)} rows -> split='aug'")
    print("Done.")


if __name__ == "__main__":
    main()
