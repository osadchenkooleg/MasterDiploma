# tests/conftest.py
import sys
from pathlib import Path

# Шукаємо корінь репозиторію (папка Master)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
