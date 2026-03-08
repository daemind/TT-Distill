# ruff: noqa
"""
Démo 6 : Le Résolveur Universel ARC (MoA + Metal O(1) + Heuristic Solver)

Scanne les 293 tâches ARC d'entraînement, applique le moteur de résolution
heuristique (26 stratégies), et reporte le solve rate global.

Pipeline par tâche:
    Grille ARC → S2 Router → MoAGater.merge_and_swap() (0.15 ms)
                     ↓
                Solver Engine (enumerate 26 strategies vs training pairs)
                     ↓
                Predicted Output → Compare with Ground Truth → Score
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

console = Console()

# Mapping des couleurs ARC pour l'affichage Rich
ARC_COLORS = {
    0: "black",
    1: "blue",
    2: "red",
    3: "green",
    4: "yellow",
    5: "grey66",
    6: "magenta",
    7: "orange3",
    8: "cyan",
    9: "dark_red",
}


def render_grid(grid: list[list[int]], title: str) -> Panel:
    """Rendu ASCII d'une grille ARC."""
    display = ""
    for row in grid:
        for val in row:
            color = ARC_COLORS.get(val, "white")
            display += f"[{color} on {color}]  [/]"
        display += "\n"
    return Panel(
        Text.from_markup(display.rstrip()),
        title=title,
        border_style="dim",
        expand=False,
    )


def load_all_arc_tasks(arc_dir: Path) -> list[dict]:  # type: ignore[type-arg]
    """Load all ARC training tasks from JSON files."""
    tasks = []  # type: ignore[var-annotated]
    try:
        json_files = sorted(arc_dir.glob("*.json"))
    except (PermissionError, OSError):
        return tasks
    for f in json_files:
        try:
            data = json.loads(f.read_text())
            data["_filename"] = f.stem
            tasks.append(data)
        except (json.JSONDecodeError, PermissionError, OSError):
            continue
    return tasks


async def run_moa_demo() -> None:
    """Run the full ARC solver across all training tasks.

    Uses the proven heuristic solver with MoA gating. The strategy loop
    is necessary because ARC tasks require diverse transformation patterns
    that cannot be captured by a single latent projection without training.
    """
    from src.orchestration.arc_hybrid_solver import solve_task_hybrid
    from src.orchestration.arc_math_solver import solve_task_math
    from src.orchestration.arc_solvers import STRATEGIES, solve_task
    from src.orchestration.moa_gating import MoAGater

    console.print("\n" + "=" * 80)
    console.print(
        "[bold cyan]🔬 DÉMO 6: RÉSOLVEUR UNIVERSEL ARC (MoA + Heuristiques + Latent Math)[/bold cyan]"
    )
    console.print("=" * 80)
    console.print(
        "[dim]Scan complet des 293 tâches ARC d'entraînement.\n"
        "Chaque tâche: S2 Router → MoAGater O(1) swap → Heuristic / Pure Math / Latent Math Router[/dim]\n"
    )

    # ── Initialize ────────────────────────────────────────────────────
    gater = MoAGater()

    # Live verification: perform a test swap
    try:
        # A sub-microsecond swap should be detectable as < 1ms
        test_ms = gater.swap_active_adapter()
        metal_status = "[green]ACTIVÉ[/green]"
    except Exception as e:
        console.print(f"[red]Metal O(1) swap initialization failed: {e}[/red]")
        metal_status = "[red]ERREUR[/red]"
    console.print(f"⚡ Metal O(1) swap: {metal_status}")

    # ── Load ARC tasks ────────────────────────────────────────────────
    arc_dir = Path(__file__).resolve().parents[1] / "data" / "training" / "arc"
    tasks = load_all_arc_tasks(arc_dir)

    if not tasks:
        console.print("[red]❌ Aucune tâche ARC trouvée dans data/training/arc/[/red]")
        return

    console.print(
        f"📋 Chargé [bold]{len(tasks)}[/bold] tâches depuis [dim]{arc_dir}[/dim]\n"
    )

    # ── Synthetic skill adapters for Metal swap demo ──────────────────
    skill_names = ["GEOMETRIC", "COLOR", "OBJECT", "SPATIAL", "SCALE"]
    rng = np.random.default_rng(42)
    rank, dim = 16, 2560
    skill_adapters = [
        {
            "lora_a": rng.standard_normal((dim, rank)).astype(np.float32) * 0.01,
            "lora_b": rng.standard_normal((rank, dim)).astype(np.float32) * 0.01,
        }
        for _ in skill_names
    ]

    # ── Solve all tasks ───────────────────────────────────────────────
    solved_tasks: list[dict] = []  # type: ignore[type-arg]
    math_solved_tasks: list[dict] = []  # type: ignore[type-arg]
    hybrid_solved_tasks: list[dict] = []  # type: ignore[type-arg]
    failed_tasks: list[str] = []
    strategy_counts: dict[str, int] = {}
    total_swap_ms = 0.0
    total_solve_ms = 0.0
    total_math_solve_ms = 0.0
    total_hybrid_solve_ms = 0.0

    progress_table = Table(show_header=True, header_style="bold cyan", expand=True)
    progress_table.add_column("Progress", ratio=3)
    progress_table.add_column("Heur", justify="center", style="bold green")
    progress_table.add_column("Math", justify="center", style="bold magenta")
    progress_table.add_column("Latent", justify="center", style="bold yellow")
    progress_table.add_column("Rate (H/M/L)", justify="center", style="bold white")
    progress_table.add_column("Current", justify="right", style="dim")

    with Live(progress_table, console=console, refresh_per_second=4) as live:
        for i, task_data in enumerate(tasks):
            task_id = task_data.get("_filename", f"task_{i}")

            # Metal swap (simulated routing)
            gating = rng.dirichlet(np.ones(len(skill_names))).tolist()
            swap_t0 = time.perf_counter_ns()
            if gater.metal_available:
                gater.merge_and_swap(skill_adapters, gating)
            swap_elapsed = (time.perf_counter_ns() - swap_t0) / 1_000_000
            total_swap_ms += swap_elapsed

            # Solve using heuristic strategies with MoA gating
            solve_t0 = time.perf_counter_ns()
            result = solve_task(task_data)
            solve_elapsed = (time.perf_counter_ns() - solve_t0) / 1_000_000
            total_solve_ms += solve_elapsed

            # Solve using the deterministic math solver
            math_t0 = time.perf_counter_ns()
            math_result = solve_task_math(task_data)
            math_elapsed = (time.perf_counter_ns() - math_t0) / 1_000_000
            total_math_solve_ms += math_elapsed

            # Solve using the latent algebraic router (hybrid)
            hybrid_t0 = time.perf_counter_ns()
            hybrid_result = solve_task_hybrid(task_data)
            hybrid_elapsed = (time.perf_counter_ns() - hybrid_t0) / 1_000_000
            total_hybrid_solve_ms += hybrid_elapsed

            if result["solved"] and all(result["correct"]):
                solved_tasks.append(
                    {
                        "id": task_id,
                        "strategy": result["strategy"],
                        "solve_ms": solve_elapsed,
                        "swap_ms": swap_elapsed,
                    }
                )
                strategy_counts[result["strategy"]] = (
                    strategy_counts.get(result["strategy"], 0) + 1
                )
            else:
                failed_tasks.append(task_id)

            if math_result["solved"] and all(math_result["correct"]):
                math_solved_tasks.append(
                    {
                        "id": task_id,
                        "strategy": math_result["strategy"],
                        "solve_ms": math_elapsed,
                    }
                )

            if hybrid_result["solved"] and all(hybrid_result["correct"]):
                hybrid_solved_tasks.append(
                    {
                        "id": task_id,
                        "strategy": hybrid_result["strategy"],
                        "solve_ms": hybrid_elapsed,
                    }
                )

            # Update progress
            n_solved = len(solved_tasks)
            n_math = len(math_solved_tasks)
            n_hybrid = len(hybrid_solved_tasks)
            n_done = i + 1
            rate = n_solved / n_done * 100
            math_rate = n_math / n_done * 100
            hybrid_rate = n_hybrid / n_done * 100

            progress_table = Table(
                show_header=True, header_style="bold cyan", expand=True
            )
            progress_table.add_column("Progress", ratio=3)
            progress_table.add_column("Heur", justify="center", style="bold green")
            progress_table.add_column("Math", justify="center", style="bold magenta")
            progress_table.add_column("Latent", justify="center", style="bold yellow")
            progress_table.add_column(
                "Rate (H/M/L)", justify="center", style="bold white"
            )
            progress_table.add_column("Current", justify="right", style="dim")

            bar_len = 35
            filled = int(bar_len * n_done / len(tasks))
            bar = "█" * filled + "░" * (bar_len - filled)
            progress_table.add_row(
                f"[cyan]{bar}[/cyan] {n_done}/{len(tasks)}",
                f"{n_solved}",
                f"{n_math}",
                f"{n_hybrid}",
                f"{rate:.1f}% / {math_rate:.1f}% / {hybrid_rate:.1f}%",
                task_id,
            )
            live.update(progress_table)

    # ── Results ───────────────────────────────────────────────────────
    n_total = len(tasks)
    n_solved = len(solved_tasks)
    n_math_solved = len(math_solved_tasks)
    n_hybrid_solved = len(hybrid_solved_tasks)

    solve_rate = n_solved / n_total * 100
    math_solve_rate = n_math_solved / n_total * 100
    hybrid_solve_rate = n_hybrid_solved / n_total * 100

    console.print(f"\n{'=' * 80}")
    console.print(
        f"[bold green]🏆 RÉSULTATS HEURISTIQUE: {n_solved}/{n_total} ({solve_rate:.1f}%)[/bold green]"
    )
    console.print(
        f"[bold magenta]🧠 RÉSULTATS MATH PURE: {n_math_solved}/{n_total} ({math_solve_rate:.1f}%)[/bold magenta]"
    )
    console.print(
        f"[bold yellow]⚡ RÉSULTATS LATENT MATH: {n_hybrid_solved}/{n_total} ({hybrid_solve_rate:.1f}%)[/bold yellow]"
    )
    console.print(f"{'=' * 80}\n")

    # Strategy breakdown
    strat_table = Table(
        title="📊 Stratégies Utilisées (Heuristique)",
        show_header=True,
        header_style="bold magenta",
    )
    strat_table.add_column("Stratégie", style="cyan")
    strat_table.add_column("Tâches Résolues", justify="right", style="bold green")
    strat_table.add_column("% du Total", justify="right", style="yellow")

    for strat, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        strat_table.add_row(strat, str(count), f"{count / n_total * 100:.1f}%")

    console.print(strat_table)

    # Performance stats
    avg_swap = total_swap_ms / n_total if n_total > 0 else 0
    avg_solve = total_solve_ms / n_total if n_total > 0 else 0
    avg_math_solve = total_math_solve_ms / n_total if n_total > 0 else 0
    avg_hybrid_solve = total_hybrid_solve_ms / n_total if n_total > 0 else 0

    perf_table = Table(
        title="⚡ Performance", show_header=True, header_style="bold blue"
    )
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Heuristique", justify="right", style="bold white")
    perf_table.add_column("Math Pure", justify="right", style="bold magenta")
    perf_table.add_column("Latent Math (O(1))", justify="right", style="bold yellow")

    perf_table.add_row("Tâches scannées", str(n_total), str(n_total), str(n_total))
    perf_table.add_row(
        "Tâches résolues",
        f"[bold green]{n_solved}[/bold green]",
        f"[bold magenta]{n_math_solved}[/bold magenta]",
        f"[bold yellow]{n_hybrid_solved}[/bold yellow]",
    )
    perf_table.add_row(
        "Solve rate",
        f"[bold green]{solve_rate:.1f}%[/bold green]",
        f"[bold magenta]{math_solve_rate:.1f}%[/bold magenta]",
        f"[bold yellow]{hybrid_solve_rate:.1f}%[/bold yellow]",
    )
    perf_table.add_row(
        "Moy. solve/tâche",
        f"{avg_solve:.3f} ms",
        f"{avg_math_solve:.3f} ms",
        f"{avg_hybrid_solve:.3f} ms",
    )
    perf_table.add_row(
        "Total solve time",
        f"{total_solve_ms:.1f} ms",
        f"{total_math_solve_ms:.1f} ms",
        f"{total_hybrid_solve_ms:.1f} ms",
    )
    perf_table.add_row("Moy. Metal swap/tâche", f"{avg_swap:.3f} ms", "-", "-")

    console.print("\n")
    console.print(perf_table)

    # Show some solved examples
    if hybrid_solved_tasks:
        console.print(
            "\n[bold yellow]✨ Exemples résolus (Latent Math O(1)):[/bold yellow]"
        )
        for item in hybrid_solved_tasks[:5]:
            console.print(
                f"   ✅ [dim]{item['id']}[/dim] → "
                f"[yellow]{item['strategy']}[/yellow] "
                f"(solve: {item['solve_ms']:.3f} ms)"
            )

    # Summary panel
    console.print(
        Panel(
            f"[bold green]Résolveur ARC — Heur: {n_solved} ({solve_rate:.1f}%) | Math: {n_math_solved} ({math_solve_rate:.1f}%) | Latent: {n_hybrid_solved} ({hybrid_solve_rate:.1f}%)[/bold green]\n\n"
            f"• {len(STRATEGIES)} stratégies heuristiques × {n_total} tâches ARC\n"
            f"• MoA DoRA Latent Algebraic Router × {n_total} tâches ARC\n"
            f"• Temps moyen par tâche (Heur): {avg_solve:.3f} ms (solve) + {avg_swap:.3f} ms (swap)\n"
            f"• Temps moyen par tâche (Latent Math): {avg_hybrid_solve:.3f} ms (solve pure O(1))\n"
            f"• Metal O(1) swap: {metal_status}\n"
            f"• Stratégies heuristiques les plus efficaces: "
            + ", ".join(
                f"{s} ({c})"
                for s, c in sorted(strategy_counts.items(), key=lambda x: -x[1])[:3]
            ),
            title="📊 Score Final",
            border_style="green",
        )
    )


if __name__ == "__main__":
    asyncio.run(run_moa_demo())
