import textwrap
from pathlib import Path

import pytest

from app.domain.boilerplate_filter import BoilerplateFilter


@pytest.fixture
def fake_stoplist_dir(tmp_path: Path) -> Path:
    """
    Створює тимчасовий каталог з маленьким стоп-списком для python.
    """
    stop_dir = tmp_path / "stoplists"
    stop_dir.mkdir(parents=True, exist_ok=True)

    # Беремо один шингл "import os" як бойлерплейт
    (stop_dir / "stoplist_python.txt").write_text(
        "import os\n",
        encoding="utf-8",
    )

    return stop_dir


@pytest.fixture
def boiler_with_fake_normalize(
    fake_stoplist_dir: Path, monkeypatch
) -> BoilerplateFilter:
    """
    Створює BoilerplateFilter з:
    - шинглами довжини 2 (k=2),
    - простим normalize_code, який просто уніфікує пробіли,
    - порогом 0.5, щоб і 'import', і 'os' вважались бойлерними.
    """

    def fake_normalize_code(code: str, lang: str) -> str:
        # Проста нормалізація: звести всі пробіли до одиночних
        return " ".join(code.split())

    # Патчимо normalize_code всередині модуля boilerplate_filter
    monkeypatch.setattr(
        "app.domain.boilerplate_filter.normalize_code",
        fake_normalize_code,
        raising=True,
    )

    boiler = BoilerplateFilter(
        default_lang="python",
        stoplist_dir=str(fake_stoplist_dir),
        langs=["python"],
        shingle_k=2,  # важливо: співпадає з тим, як ми підбираємо шингли
        boiler_ratio_thresh=0.5,  # 👈 знижуємо поріг
        min_keep_ratio=0.3,
    )

    return boiler


def test_filter_removes_import_boilerplate(
    boiler_with_fake_normalize: BoilerplateFilter,
):
    """
    Перевіряємо, що типовий шаблон імпорту видаляється,
    а корисний код (print) залишається.
    """

    boiler = boiler_with_fake_normalize

    original_code = textwrap.dedent(
        """
        import os

        print("hello")
        """
    )

    # Для нашого fake_normalize_code це стане:
    # "import os print("hello")"
    # Токени: ["import", "os", 'print("hello")']
    #
    # Шингли при k=2:
    # 0: "import os"
    # 1: "os print("hello")"
    #
    # STOP = {"import os", "os print"} (з файлу)
    # → перші два токени будуть переважно покриті бойлерними шинглами і видаляться

    filtered = boiler.filter_for_embedding(original_code, "python")

    # Маємо переконатися, що імпорт зник
    assert "import" not in filtered
    assert "os" not in filtered

    # Але корисний виклик print залишився
    assert "print" in filtered
    assert "hello" in filtered


def test_filter_no_stoplist_keeps_code(monkeypatch, tmp_path: Path):
    """
    Якщо для мови немає стоп-списку, фільтр має просто повернути
    нормалізований код (без вирізання).
    """

    def fake_normalize_code(code: str, lang: str) -> str:
        return " ".join(code.split())

    # Патчимо normalize_code
    monkeypatch.setattr(
        "app.domain.boilerplate_filter.normalize_code",
        fake_normalize_code,
        raising=True,
    )

    # Стоп-списки є тільки для python, але ми будемо перевіряти lang="go"
    stop_dir = tmp_path / "stoplists"
    stop_dir.mkdir(parents=True, exist_ok=True)
    (stop_dir / "stoplist_python.txt").write_text("import os\n", encoding="utf-8")

    boiler = BoilerplateFilter(
        default_lang="python",
        stoplist_dir=str(stop_dir),
        langs=["python"],  # немає "go" у списку мов зі стоп-списком
        shingle_k=2,
    )

    original_code = 'package main\n\nfunc main() { println("hi") }'
    filtered = boiler.filter_for_embedding(original_code, "go")

    # Має бути просто нормалізація, а не видалення
    assert "package main" in filtered
    assert "println" in filtered
    # normalize_code зведе пробіли, тож рядок точно не пустий
    assert filtered.strip() != ""
