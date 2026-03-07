# ruff: noqa
#!/usr/bin/env python3
"""TT-Distill Demo CLI.

Interface CLI interactive pour exécuter les 7 démonstrations de l'architecture TT-Distill.

Usage:
    uv run demos/main.py --list          # Lister les démonstrations disponibles
    uv run demos/main.py --demo 1        # Exécuter la démo 1 (Stress-Test)
    uv run demos/main.py --demo 2        # Exécuter la démo 2 (Autopsie Algébrique)
    uv run demos/main.py --demo 3        # Exécuter la démo 3 (Reality Filter)
    uv run demos/main.py --demo 4        # Exécuter la démo 4 (Post-Silicon)
    uv run demos/main.py --demo 5        # Exécuter la démo 5 (ARC-AGI Resolution)
    uv run demos/main.py --demo 6        # Exécuter la démo 6 (MoA Resolver)
    uv run demos/main.py --demo 7        # Exécuter la démo 7 (Metal O(1) Swap)
    uv run demos/main.py --all           # Exécuter toutes les démonstrations
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports BEFORE any other imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / 'src'))

from demo_arc_agi import run_arc_demo
from demo_maca_visualizer import main as demo_maca_dora_main
from demo_metal_swap import run_demo as run_metal_swap_demo
from demo_moa_resolver import run_moa_demo
from demo_post_silicon import main as demo_post_silicon_main
from demo_reality_filter import main as demo_reality_filter_main
from demo_stress_test import main as demo_stress_test_main
from rich.console import Console
from rich.table import Table

console = Console()

# Wrappers synchrones pour les démos 5 et 6
def demo_arc_agi_main() -> None:
    asyncio.run(run_arc_demo())

def demo_moa_resolver_main() -> None:
    asyncio.run(run_moa_demo())


DEMO_REGISTRY: dict[int, dict[str, Any]] = {
    1: {
        "name": "Stress-Test de Fréquence",
        "description": "S1 Reflex vs RAG Classique - Écrasante supériorité du Système 1",
        "function": demo_stress_test_main,
        "icon": "🏎️",
    },
    2: {
        "name": "Autopsie Algébrique",
        "description": "MACA & DoRA Visualizer - Compression mathématique pure",
        "function": demo_maca_dora_main,
        "icon": "🧠",
    },
    3: {
        "name": "Reality Filter",
        "description": "Compliance AGENT.md - Zéro Hallucination garantie",
        "function": demo_reality_filter_main,
        "icon": "🛡️",
    },
    4: {
        "name": "Symbiose Post-Silicon",
        "description": "Hardware Profiling - Harmonie Mac Studio (M2 Max/ANE)",
        "function": demo_post_silicon_main,
        "icon": "🔬",
    },
    5: {
        "name": "Résolution ARC-AGI",
        "description": "Transductive Learning - Du Test-Time Compute à l'Instinct",
        "function": demo_arc_agi_main,
        "icon": "🧩",
    },
    6: {
        "name": "Résolveur Universel (MoA)",
        "description": "Mixture of Adapters - Hot-Swap dynamique du Système 1",
        "function": demo_moa_resolver_main,
        "icon": "🔀",
    },
    7: {
        "name": "Metal O(1) DoRA Swap",
        "description": "C++ Metal Backend — Benchmark du Swap O(1) sub-milliseconde",
        "function": run_metal_swap_demo,
        "icon": "🔧",
    },
}


def print_demo_list() -> None:
    """Afficher la liste des démonstrations disponibles."""
    console.print("\n" + "=" * 70)
    console.print("[bold blue]🚀 TT-Distill Demo Suite[/bold blue]")
    console.print("=" * 70)

    table = Table(title="Démonstrations Disponibles")
    table.add_column("#", style="cyan", width=3)
    table.add_column("Démonstration", style="green")
    table.add_column("Description", style="yellow")

    for demo_id, demo_info in sorted(DEMO_REGISTRY.items()):
        table.add_row(
            f"[bold]{demo_id}[/bold]",
            f"{demo_info['icon']} {demo_info['name']}",
            demo_info["description"],
        )

    console.print(table)

    console.print("\n[bold cyan]Usage:[/bold cyan]")
    console.print("  uv run demos/main.py --list          # Lister les démonstrations")
    for i in range(1, 7):
        console.print(f"  uv run demos/main.py --demo {i}        # Exécuter la démo {i}")
    console.print("  uv run demos/main.py --all           # Exécuter toutes les démonstrations")
    console.print()


def run_demo(demo_id: int) -> None:
    """Exécuter une démonstration spécifique."""
    if demo_id not in DEMO_REGISTRY:
        console.print(f"[bold red]❌ Démonstration #{demo_id} non trouvée![/bold red]")
        console.print("[dim]Utilisez --list pour voir les démonstrations disponibles[/dim]")
        sys.exit(1)

    demo_info = DEMO_REGISTRY[demo_id]
    console.print("\n" + "=" * 70)
    console.print(f"[bold blue]{demo_info['icon']} {demo_info['name']}[/bold blue]")
    console.print("=" * 70)
    console.print(f"[dim]{demo_info['description']}[/dim]\n")

    try:
        demo_info["function"]()
    except Exception as e:
        console.print(f"[bold red]❌ Erreur: {e}[/bold red]")
        raise


def run_all_demos() -> None:
    """Exécuter toutes les démonstrations."""
    import subprocess

    console.print("\n" + "=" * 70)
    console.print("[bold blue]🚀 Exécution de toutes les démonstrations[/bold blue]")
    console.print("=" * 70)

    for demo_id in sorted(DEMO_REGISTRY.keys()):
        try:
            # We spawn a pristine subprocess for each demo to ensure Llama.cpp and Metal
            # completely free memory between executions. This avoids malloc tracking errors.
            console.print(f"\n[bold yellow]🔄 Isolation VRAM: Démarrage de la Démo #{demo_id} dans un nouveau processus...[/bold yellow]")
            subprocess.run([sys.executable, __file__, "--demo", str(demo_id)], check=True)
            console.print(f"\n[bold green]✅ Démo #{demo_id} terminée avec succès![/bold green]\n")
        except subprocess.CalledProcessError as e:
            console.print(f"\n[bold red]❌ Démo #{demo_id} a échoué (Exit: {e.returncode})[/bold red]")
            console.print("[dim]Continuation vers la démo suivante...[/dim]\n")
        except Exception as e:
            console.print(f"\n[bold red]❌ Erreur inattendue pour la Démo #{demo_id}: {e}[/bold red]")
            console.print("[dim]Continuation vers la démo suivante...[/dim]\n")


def main() -> None:
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="TT-Distill Demo Suite - Démonstrations interactives",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  uv run demos/main.py --list          Lister les démonstrations
  uv run demos/main.py --demo 5        Exécuter la Résolution ARC-AGI
  uv run demos/main.py --demo 6        Exécuter le Résolveur Universel
  uv run demos/main.py --all           Exécuter toutes les démonstrations
        """,
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les démonstrations disponibles",
    )
    parser.add_argument(
        "--demo",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        help="Exécuter une démonstration spécifique (1-6)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Exécuter toutes les démonstrations",
    )

    args = parser.parse_args()

    if args.list:
        print_demo_list()
    elif args.demo:
        run_demo(args.demo)
    elif args.all:
        run_all_demos()
    else:
        print_demo_list()


if __name__ == "__main__":
    main()
