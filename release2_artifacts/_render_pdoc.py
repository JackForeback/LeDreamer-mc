"""Render pdoc HTML for the LeDreamer-mc authored modules.

Stubs every eval-only runtime dependency (minerl, gym3, gym) with
``unittest.mock.MagicMock`` so pdoc can introspect the modules without a
full MineRL install. pdoc only reads docstrings / signatures / module
attributes — it never actually executes the code — so a MagicMock stub
is sufficient.

Run from the repo root:
    env/bin/python release2_artifacts/_render_pdoc.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO = Path(__file__).resolve().parent.parent

# Stubs for the minerl / gym stack — attribute access on a MagicMock
# auto-returns more MagicMocks, so even nested imports resolve.
for name in [
    "gym3", "gym3.types", "gym", "gym.spaces",
    "minerl", "minerl.herobraine", "minerl.herobraine.env_specs",
    "minerl.herobraine.env_specs.human_survival_specs",
    "minerl.herobraine.hero", "minerl.herobraine.hero.mc",
    "minerl.env", "minerl.env.malmo",
]:
    sys.modules[name] = MagicMock(name=name)

# Some modules use ``from gym import spaces`` — give the fake ``gym``
# a ``spaces`` attribute that points to our fake submodule.
sys.modules["gym"].spaces = sys.modules["gym.spaces"]

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "Video-Pre-Training"))

import pdoc  # noqa: E402

OUT_DIR = REPO / "release2_artifacts" / "docs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODULES = [
    str(REPO / "minecraft_vpt_dataset.py"),
    str(REPO / "train_dreamer4_minecraft.py"),
    str(REPO / "download" / "download_vpt.py"),
    str(REPO / "Video-Pre-Training" / "dreamer4_minecraft_agent.py"),
    str(REPO / "Video-Pre-Training" / "evaluate_dreamer4_minecraft.py"),
]

pdoc.render.configure(docformat="google")
pdoc.pdoc(*MODULES, output_directory=OUT_DIR)
print(f"pdoc wrote HTML to {OUT_DIR}")
