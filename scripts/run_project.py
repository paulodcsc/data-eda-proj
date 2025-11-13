#!/usr/bin/env python3
"""Automatiza as etapas principais definidas no projeto."""

from __future__ import annotations

import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"


def run_step(script: str) -> None:
    location = SCRIPTS_DIR / script
    if not location.exists():
        raise FileNotFoundError(f"Script {location} não encontrado")
    print(f"\n→ Executando `{script}`")
    subprocess.run([sys.executable, str(location)], check=True)


def main() -> None:
    steps = [
        "build_clean_data.py",
        "train_models.py",
    ]
    for step in steps:
        run_step(step)
    print("\nPipeline concluída. Os artefatos estão em data/clean/ e src/models/trained/")


if __name__ == "__main__":
    main()
