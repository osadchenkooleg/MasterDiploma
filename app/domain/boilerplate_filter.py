import os
from pathlib import Path
from typing import Dict, List, Optional, Set

from app.domain.light_normalize import normalize_code


class BoilerplateFilter:
    """
    Фільтрація бойлерплейту для коду перед векторизацією.

    Кроки:
    1) Легка нормалізація (normalize_code).
    2) Токенізація (split по пробілах).
    3) Побудова шинглів довжини k.
    4) Для кожного шингла перевіряємо, чи він у STOP[lang].
    5) Для кожного токена рахуємо:
       - скільки шинглів його покривають загалом,
       - скільки з них бойлерні.
       Якщо частка бойлерних >= boiler_ratio_thresh → токен вважаємо бойлерним.
    6) Вирізаємо бойлерні токени; якщо залишилось занадто мало,
       повертаємо просто нормалізований код (fallback).
    """

    def __init__(
        self,
        default_lang: str = "python",
        stoplist_dir: Optional[str] = None,
        langs: Optional[list[str]] = None,
        shingle_k: int = 5,
        boiler_ratio_thresh: float = 0.6,
        min_keep_ratio: float = 0.3,
    ):
        """
        :param default_lang: мова за замовчуванням для normalize_code
        :param stoplist_dir: директорія з файлами stoplist_{lang}.txt
        :param langs: мови, для яких будемо намагатися завантажити STOP
        :param shingle_k: довжина шингла по токенах (має збігатися з офлайн-скриптом)
        :param boiler_ratio_thresh: поріг частки бойлерних шинглів для токена
        :param min_keep_ratio: мінімальна частка токенів, які мають залишитися
                               після фільтрації; інакше fallback на norm
        """
        self.default_lang = default_lang
        self.stoplist_dir = Path(
            stoplist_dir or os.getenv("STOPLIST_DIR", "data/stoplists")
        )
        self.langs = langs or ["java", "js", "go", "python"]
        self.shingle_k = shingle_k
        self.boiler_ratio_thresh = boiler_ratio_thresh
        self.min_keep_ratio = min_keep_ratio

        # lang (java/js/go/python) -> set(shingle)
        self.stop_shingles: Dict[str, Set[str]] = {}

        self._load_stoplists()

    # ─────────────────────────────────────────────────────────────────────
    # Lang mapping
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _map_lang_for_normalizer(lang: Optional[str]) -> Optional[str]:
        """
        Відображення значень lang у те, що очікує normalize_code.
        """
        if not lang:
            return None
        lang = lang.lower()
        if lang == "js":
            return "javascript"
        if lang in ("py", "python"):
            return "python"
        return lang

    @staticmethod
    def _map_lang_for_stoplist(lang: Optional[str]) -> Optional[str]:
        """
        Мапінг для STOP-списків: тут ми очікуємо "java/js/go/python",
        як вони зберігаються в codes_v4 і в stoplist-файлах.
        """
        if not lang:
            return None
        return lang.lower()

    # ─────────────────────────────────────────────────────────────────────
    # Loading stoplists
    # ─────────────────────────────────────────────────────────────────────
    def _load_stoplists(self) -> None:
        """
        Завантажує файли stoplist_{lang}.txt з self.stoplist_dir
        і кладе їх у self.stop_shingles[lang] як set(...)
        """
        if not self.stoplist_dir.exists():
            print(f"[BoilerplateFilter] stoplist dir not found: {self.stoplist_dir}")
            return

        for lang in self.langs:
            filename = self.stoplist_dir / f"stoplist_{lang}.txt"
            if not filename.exists():
                print(
                    f"[BoilerplateFilter] no stoplist file for lang={lang}: {filename}"
                )
                continue

            with filename.open("r", encoding="utf-8") as f:
                shingles = [line.strip() for line in f if line.strip()]

            self.stop_shingles[lang] = set(shingles)
            print(
                f"[BoilerplateFilter] loaded {len(shingles)} shingles for lang={lang} "
                f"from {filename}"
            )

    # ─────────────────────────────────────────────────────────────────────
    # Shingling helpers
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def _make_shingles(tokens: List[str], k: int) -> List[str]:
        n = len(tokens)
        if n < k:
            return []
        return [" ".join(tokens[i : i + k]) for i in range(n - k + 1)]

    # ─────────────────────────────────────────────────────────────────────
    # Core filtering logic
    # ─────────────────────────────────────────────────────────────────────
    def _filter_tokens_by_stop_shingles(
        self, tokens: List[str], stop_set: Set[str]
    ) -> List[str]:
        """
        Основна логіка вирізання бойлерплейт-токенів на основі STOP-шинглів.

        :param tokens: список токенів нормалізованого коду
        :param stop_set: множина бойлерних шинглів для відповідної мови
        :return: список токенів після фільтрації
        """
        n = len(tokens)
        k = self.shingle_k

        if n == 0 or k <= 1 or n < k or not stop_set:
            # Немає що шинглювати або STOP-порожній → нічого не видаляємо
            return tokens

        shingles = self._make_shingles(tokens, k)
        m = len(shingles)
        if m == 0:
            return tokens

        # Для кожного токена рахуємо:
        # - total_cover[i]: у скількох шинглах він взагалі зустрічається
        # - boiler_cover[i]: у скількох "бойлерних" шинглах він зустрічається
        total_cover = [0] * n
        boiler_cover = [0] * n

        for start_idx, sh in enumerate(shingles):
            is_boiler = sh in stop_set
            # вікно токенів [start_idx, start_idx + k - 1]
            end_idx = start_idx + k
            for i in range(start_idx, min(end_idx, n)):
                total_cover[i] += 1
                if is_boiler:
                    boiler_cover[i] += 1

        # Визначаємо, які токени видалити
        keep_mask = [True] * n
        for i in range(n):
            if total_cover[i] == 0:
                # токен не потрапив у жоден шингл (може бути, якщо n == k)
                continue
            ratio = boiler_cover[i] / total_cover[i]
            if ratio >= self.boiler_ratio_thresh:
                keep_mask[i] = False

        kept_tokens = [tok for tok, keep in zip(tokens, keep_mask) if keep]

        # Захисний механізм: якщо ми видалили майже все — відкат.
        if not kept_tokens:
            return tokens

        keep_ratio = len(kept_tokens) / n
        if keep_ratio < self.min_keep_ratio:
            # надто агресивна фільтрація → краще повернути оригінальні токени
            return tokens

        return kept_tokens

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def filter_for_embedding(self, code: str, lang: Optional[str]) -> str:
        """
        Публічний метод, який використовує APIшний код:

        - викликається з /codes та /uniqueness/check,
        - повертає текст, який йде в model.encode().

        На цьому етапі:
        1) робимо light-normalize
        2) застосовуємо STOP-шингли, якщо вони є для цієї мови
        3) повертаємо очищений текст
        """
        if not code:
            return ""

        norm_lang = self._map_lang_for_normalizer(lang) or self.default_lang
        norm = normalize_code(code, norm_lang)
        tokens = norm.split()
        if not tokens:
            return norm

        stop_lang = self._map_lang_for_stoplist(lang) or self.default_lang
        stop_set = self.stop_shingles.get(stop_lang)

        if not stop_set:
            # немає стоп-списку для цієї мови → тільки normalize
            return norm

        filtered_tokens = self._filter_tokens_by_stop_shingles(tokens, stop_set)
        return " ".join(filtered_tokens)
