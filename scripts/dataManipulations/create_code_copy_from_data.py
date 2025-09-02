# .venv/bin/python - <<'PY'
import pathlib
import random

from datasets import load_from_disk

ds = load_from_disk("data/cleaned/java/train")
row = ds[random.randrange(len(ds))]
code = row["func1"] or row["func2"]
path = pathlib.Path("samples/similar.java")
path.write_text(code, encoding="utf-8")
print("Saved:", path)
# PY
