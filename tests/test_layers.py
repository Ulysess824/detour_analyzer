"""Layering: engine imports no other layer, evaluation imports only engine, experiments may import both, app (the desktop app) may import all."""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
ALLOWED = {"engine": {"engine"}, "evaluation": {"engine", "evaluation"}, "experiments": {"engine", "evaluation", "experiments"},
           "app": {"engine", "evaluation", "experiments", "app"}}


def test_each_layer_imports_only_the_layers_below_it():
    bad = []
    for layer, allowed in ALLOWED.items():
        for path in (SRC / layer).glob("*.py"):
            for used in re.findall(r"^\s*(?:from|import) src\.(\w+)", path.read_text(encoding="utf-8"), flags=re.M):
                if used not in allowed:
                    bad.append(f"{path.name} ({layer}) imports src.{used}")
    assert not bad, bad


def test_every_module_lives_in_a_layer():
    assert not (SRC / "utils").exists()
    assert {p.name for p in SRC.iterdir() if p.is_dir() and p.name != "__pycache__"} == set(ALLOWED)
