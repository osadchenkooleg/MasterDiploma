#!/usr/bin/env python3
import re
from typing import Dict, Set

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
