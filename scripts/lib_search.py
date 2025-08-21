#!/usr/bin/env python
import hashlib
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# уникнути форків-попереджень та OpenMP-конфліктів
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
import torch  # ІМПОРТУЄМО ПЕРШИМ (важливо для OpenMP на macOS)
from datasets import Dataset, load_from_disk
from transformers import AutoModel, AutoTokenizer

DIM = 768

JAVA_KEYWORDS = {
    "abstract",
    "assert",
    "boolean",
    "break",
    "byte",
    "case",
    "catch",
    "char",
    "class",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extends",
    "final",
    "finally",
    "float",
    "for",
    "goto",
    "if",
    "implements",
    "import",
    "instanceof",
    "int",
    "interface",
    "long",
    "native",
    "new",
    "package",
    "private",
    "protected",
    "public",
    "return",
    "short",
    "static",
    "strictfp",
    "super",
    "switch",
    "synchronized",
    "this",
    "throw",
    "throws",
    "transient",
    "try",
    "void",
    "volatile",
    "while",
    "true",
    "false",
    "null",
    "var",
    "record",
    "sealed",
    "permits",
    "non-sealed",
}
PY_KEYWORDS = {
    "False",
    "None",
    "True",
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "try",
    "while",
    "with",
    "yield",
}
IDENT_RE = re.compile(r"[A-Za-z_]\w+")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_index_and_meta(index_dir: str):
    idx = faiss.read_index(str(Path(index_dir) / "index.faiss"))
    meta = []
    with (Path(index_dir) / "meta.tsv").open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            row_idx, uid, label, split, lang = line.rstrip("\n").split("\t")
            meta.append((int(row_idx), uid, int(label), split, lang))
    return idx, meta


def load_model(model_name: str = "microsoft/codebert-base"):
    dev = pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(dev).eval()
    return tok, mdl, dev


def normalize_code(code: str) -> str:
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"//.*?$", "", code, flags=re.MULTILINE)
    code = re.sub(r"\s+", " ", code).strip()
    return code


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embed(code: str, tok, model, device) -> np.ndarray:
    with torch.no_grad():
        t = tok(code, return_tensors="pt", truncation=True, max_length=512).to(device)
        v = model(**t).last_hidden_state.mean(1)
        v = torch.nn.functional.normalize(v, p=2, dim=1)
        return v.cpu().numpy().astype("float32")  # [1, DIM]


# кеш об'єктів Dataset, щоб не відкривати Arrow кожного разу
_DS_CACHE: Dict[Tuple[str, str], Dataset] = {}


def fetch_code_by_uid(lang: str, split: str, uid: str) -> Optional[str]:
    pair_id_str, side_str = uid.split("_")
    pair_id, side = int(pair_id_str), int(side_str)
    key = (lang, split)
    if key not in _DS_CACHE:
        _DS_CACHE[key] = load_from_disk(f"data/cleaned/{lang}/{split}")
    ds = _DS_CACHE[key]
    hit = ds.filter(lambda r: r["id"] == pair_id)
    if len(hit) == 0:
        return None
    row = hit[0]
    return row["func1"] if side == 1 else row["func2"]


def identifiers(code: str, lang: str) -> set:
    toks = set(IDENT_RE.findall(code))
    if lang == "java":
        toks = {t for t in toks if t not in JAVA_KEYWORDS}
    elif lang == "python":
        toks = {t for t in toks if t not in PY_KEYWORDS}
    return toks


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def set_search_params(index, fast: bool):
    # HNSW
    if hasattr(index, "hnsw"):
        index.hnsw.efSearch = 48 if fast else 128
    # IVF (IVF*, зокрема IVFPQ)
    if hasattr(index, "nprobe"):
        index.nprobe = 8 if fast else 64
