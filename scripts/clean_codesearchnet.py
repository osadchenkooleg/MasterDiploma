#!/usr/bin/env python
import argparse
import re
from pathlib import Path

from datasets import load_from_disk

# python
# .venv/bin/python scripts/clean_codesearchnet.py --lang python --splits train,validation --limit 200000
#
# # javascript (за потреби)
# .venv/bin/python scripts/clean_codesearchnet.py --lang javascript --splits train,validation --limit 200000

# -------- коментарі по мовах --------
LINE_CPP = re.compile(r"//.*?$", re.MULTILINE)
BLOCK_CPP = re.compile(r"/\*.*?\*/", re.DOTALL)

LINE_HASH = re.compile(r"#.*?$", re.MULTILINE)  # python, ruby, php підтримують '#'
BLOCK_PY1 = re.compile(r'"""[\s\S]*?"""', re.DOTALL)  # трик-лапки (спрощено)
BLOCK_PY2 = re.compile(r"'''[\s\S]*?'''", re.DOTALL)

BLOCK_RUBY = re.compile(r"=begin[\s\S]*?=end", re.DOTALL)  # ruby block comments


def strip_comments(code: str, lang: str) -> str:
    code = code or ""
    if lang in ("java", "javascript", "go", "php"):
        code = BLOCK_CPP.sub("", code)
        code = LINE_CPP.sub("", code)
        if lang == "php":
            code = LINE_HASH.sub("", code)  # php теж має '#'
    elif lang == "python":
        code = BLOCK_PY1.sub("", code)
        code = BLOCK_PY2.sub("", code)
        code = LINE_HASH.sub("", code)
    elif lang == "ruby":
        code = BLOCK_RUBY.sub("", code)
        code = LINE_HASH.sub("", code)
    return code.strip()


CANDIDATE_CODE_COLS = ["code", "func_code_string", "original_string", "function"]


def pick_code_column(ds):
    cols = set(ds.column_names)
    for c in CANDIDATE_CODE_COLS:
        if c in cols:
            return c
    raise KeyError(f"Не знайшов колонку з кодом. Є колонки: {sorted(cols)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--lang",
        required=True,
        choices=["python", "javascript", "java", "go", "php", "ruby"],
    )
    ap.add_argument(
        "--splits", default="train,validation", help="train,validation,test"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="взяти перші N прикладів (0=всі)"
    )
    ap.add_argument(
        "--num_proc", type=int, default=4, help="паралельних процесів для map()"
    )
    args = ap.parse_args()

    for sp in [x.strip() for x in args.splits.split(",") if x.strip()]:
        src = Path(f"data/raw/codesearchnet/{args.lang}/{sp}")
        if not src.exists():
            print(f"⏭ skip {args.lang}/{sp} (no raw dir)")
            continue

        ds = load_from_disk(src)
        if len(ds) == 0:
            print(f"⏭ skip {args.lang}/{sp} (empty dataset)")
            continue

        code_col = pick_code_column(ds)

        # додамо id, якщо його немає
        if "id" not in ds.column_names:
            ds = ds.add_column("id", list(range(len(ds))))

        # перейменуємо вибрану колонку в 'code' і приберемо зайві колонки
        if code_col != "code":
            ds = ds.rename_column(code_col, "code")
        keep = {"id", "code"}
        drop = [c for c in ds.column_names if c not in keep]
        if drop:
            ds = ds.remove_columns(drop)

        # очистка коментарів
        def _clean(batch):
            return {"code": [strip_comments(c, args.lang) for c in batch["code"]]}

        print(f"➡️  cleaning {args.lang}/{sp}: rows={len(ds):,}, src_col='{code_col}'")
        cleaned = ds.map(
            _clean, batched=True, batch_size=10_000, num_proc=args.num_proc
        )

        # обрізання за потреби
        if args.limit and args.limit < len(cleaned):
            cleaned = cleaned.select(range(args.limit))
            print(f"✂️  limited to {len(cleaned):,} rows")

        out = Path(f"data/cleaned/{args.lang}/{sp}")
        out.mkdir(parents=True, exist_ok=True)
        cleaned.save_to_disk(out)
        print(f"✔ saved → {out}  (rows={len(cleaned):,})")
