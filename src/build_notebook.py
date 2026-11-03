"""Convert a `# %%` script in src/ into a Kaggle notebook in notebooks/.

    python src/build_notebook.py sokoban_cwm_transcript_v4    # one
    python src/build_notebook.py --all                         # every script in src/
"""

import json
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "notebooks"


def build(name):
    source = (SRC / f"{name}.py").read_text()
    cells = []
    for chunk in re.split(r"^# %%", source, flags=re.MULTILINE)[1:]:
        header, _, body = chunk.partition("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.strip().splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.strip().splitlines(keepends=True)})
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{name}.ipynb").write_text(json.dumps(notebook, indent=1))
    print(f"notebooks/{name}.ipynb: {len(cells)} cells")


if __name__ == "__main__":
    names = [p.stem for p in sorted(SRC.glob("sokoban_*.py"))] if sys.argv[1:] == ["--all"] else sys.argv[1:]
    for n in names:
        build(n)
