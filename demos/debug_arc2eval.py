# ruff: noqa
"""Debug script: test algebraic spaces on arc2eval tasks."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scipy.ndimage import label

from src.orchestration.arc_math_solver import SymmetryQuotient, TranslationPeriod


def load_all_arc_tasks(arc_dir: Path) -> dict:  # type: ignore[type-arg]
    tasks = {}
    for f in sorted(arc_dir.glob("*.json")):
        data = json.loads(f.read_text())
        tasks[f.stem] = data
    return tasks


def analyze_shapes(grid, bg_color=0):  # type: ignore[no-untyped-def]
    """Extract connected components as topological invariants."""
    mask = (grid != bg_color)
    labeled, n = label(mask)
    shapes = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        colors = set(grid[labeled == i].tolist())
        shapes.append({
            "id": i,
            "size": len(coords),
            "dim": (r_max - r_min + 1, c_max - c_min + 1),
            "pos": (int(r_min), int(c_min)),
            "colors": colors,
        })
    return shapes


arc_dir = Path(__file__).resolve().parents[1] / "data" / "training" / "arc2eval"
tasks = load_all_arc_tasks(arc_dir)

for _tid, task_data in sorted(tasks.items())[:40]:
    t0 = task_data["train"][0]
    inp = np.array(t0["input"])
    out = np.array(t0["output"])
    same = inp.shape == out.shape
    has_8 = 8 in set(inp.flatten().tolist())

    inp_shapes = analyze_shapes(inp)  # type: ignore[no-untyped-call]
    out_shapes = analyze_shapes(out)  # type: ignore[no-untyped-call]

    extras = []

    # Check SymmetryQuotient on mask-8 tasks
    if has_8:
        rect = SymmetryQuotient.find_mask_rect(inp, 8)
        if rect:
            r_min, r_max, c_min, c_max = rect
            extras.append(f"mask8={r_max-r_min+1}x{c_max-c_min+1}")
            if same:
                recon = SymmetryQuotient.reconstruct_from_symmetry(inp, 8)
                if recon is not None:
                    extras.append("sym=" + ("✅" if np.array_equal(recon, out) else "❌"))
                else:
                    extras.append("sym=None")
            else:
                ext = SymmetryQuotient.extract_masked_content(inp, 8)
                if ext is not None and ext.shape == out.shape:
                    extras.append("ext=" + ("✅" if np.array_equal(ext, out) else "❌"))

    # Check TranslationPeriod
    if same:
        period = TranslationPeriod.detect_period(inp)
        if period:
            repaired = TranslationPeriod.repair_periodic(inp)
            if repaired is not None:
                match = np.array_equal(repaired, out)
                diff = np.sum(repaired != out)
                extras.append(f"period={period} repair={'✅' if match else f'❌({diff}px)'}")

    dim = "SAME" if same else f"{inp.shape[0]}x{inp.shape[1]}->{out.shape[0]}x{out.shape[1]}"
    ex = " | " + " ".join(extras) if extras else ""
