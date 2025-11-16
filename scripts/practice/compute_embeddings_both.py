#!/usr/bin/env python3
import os
import subprocess

LANGS = os.getenv("EMB_LANGS", "javascript,go,python")
SPLITS = os.getenv("EMB_SPLITS", "validation")


def run_embeddings(mode: str, ver: int):
    env = os.environ.copy()
    env["EMB_CODE_NORM"] = mode
    env["EMB_TRANSFORM_VER"] = str(ver)
    env["EMB_REPLACE"] = "1"  # перезаписати старі ембеддинги
    print(f"\n=== Running embeddings for {mode.upper()} (ver={ver}) ===")
    subprocess.run(
        ["python3", "scripts/practice/compute_embeddings.py"], env=env, check=True
    )


def run_eval(ver: int):
    env = os.environ.copy()
    env["EMB_MODEL"] = "microsoft/codebert-base"
    env["EMB_POOL"] = "mean"
    env["EMB_TRANSFORM_VER"] = str(ver)
    print(f"\n=== Evaluating embeddings (ver={ver}) ===")
    subprocess.run(
        ["python3", "scripts/practice/eval_pairs_baseline.py"], env=env, check=True
    )


def main():
    print(f"Embedding for langs={LANGS}, splits={SPLITS}")

    # 1) RAW
    run_embeddings("raw", 2)
    run_eval(2)

    # 2) LIGHT
    run_embeddings("light", 3)
    run_eval(3)

    print("\nAll embeddings computed and evaluated successfully.")


if __name__ == "__main__":
    main()
