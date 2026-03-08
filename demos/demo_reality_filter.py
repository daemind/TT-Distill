#!/usr/bin/env python3
"""Demo 3: Reality Filter (Docker Sandbox & Auto-Healing).

Démonstration de l'immunité de l'architecture face aux hallucinations LLM.
Visualisation de la boucle : Génération -> Crash -> Analyse -> Auto-correction.
"""

import time

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()

# --- DONNÉES DE SIMULATION ---

HALLUCINATED_CODE = """def parse_system_logs(log_path):
    import sys_parser  # ⚠️ HALLUCINATION CLASSIQUE

    parser = sys_parser.SystemLogAnalyzer()
    return parser.extract_errors(log_path, strict=True)
"""

CORRECTED_CODE = """def parse_system_logs(log_path):
    import re  # ✅ CORRECTION : Utilisation de la lib standard

    errors = []
    error_pattern = re.compile(r"ERROR|CRITICAL|FATAL")
    with open(log_path, 'r') as f:
        for line in f:
            if error_pattern.search(line):
                errors.append(line.strip())
    return errors
"""

TRACEBACK_ERROR = """Traceback (most recent call last):
  File "/sandbox/workspace/task_1.py", line 2, in parse_system_logs
    import sys_parser
ModuleNotFoundError: No module named 'sys_parser'
"""


def generate_layout(
    phase: str, code: str, status: str, logs: str, is_error: bool = False
) -> Layout:
    """Génère le tableau de bord interactif."""
    layout = Layout()

    # En-tête
    header = Panel(
        Text(
            "🛡️ TT-Distill : Reality Filter & Auto-Healing",
            style="bold cyan",
            justify="center",
        ),
        style="blue",
    )

    # Panneau de Code
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    code_panel = Panel(
        syntax, title="[bold yellow]Cortex (Code Généré)[/]", border_style="yellow"
    )

    # Panneau Docker (Environnement)
    log_style = "bold red" if is_error else "bold green"
    docker_panel = Panel(
        Text(logs, style=log_style),
        title=f"[bold blue]📦 Docker Sandbox ({status})[/]",
        border_style="red" if is_error else "green",
    )

    # Panneau de Log MACA (Pensée)
    maca_table = Table(show_header=False, box=None, expand=True)
    maca_table.add_column("Agent", style="cyan")
    maca_table.add_column("Action", style="magenta")

    if phase == "GENERATION":
        maca_table.add_row(
            "🧠 Système 2",
            "Génération du script basée sur les probabilités statistiques...",
        )
    elif phase == "CRASH":
        maca_table.add_row(
            "💥 Reality Filter",
            "[bold red]Alerte: Incohérence matérielle détectée ![/]",
        )
        maca_table.add_row(
            "🧠 Système 2",
            "Récupération du Stack Trace. Initiation du protocole de réflexion...",
        )
    elif phase == "HEALING":
        maca_table.add_row(
            "🤖 Agent_1", "La librairie 'sys_parser' n'existe pas dans PyPI."
        )
        maca_table.add_row(
            "🤖 Agent_2", "Remplacement par 're' (Expression régulière standard)."
        )
        maca_table.add_row(
            "🎯 MACA Engine",
            "[bold green]Consensus atteint (Score: 0.985). Réécriture du code.[/]",
        )
    elif phase == "SUCCESS":
        maca_table.add_row("✅ Reality Filter", "Exécution validée par le compilateur.")
        maca_table.add_row(
            "📝 GitManager", "Commit: 'Auto-fix ModuleNotFoundError on sys_parser'"
        )

    maca_panel = Panel(
        maca_table,
        title="[bold magenta]Salle de Délibération (MACA)[/]",
        border_style="magenta",
    )

    # Structure de l'écran
    layout.split(
        Layout(header, size=3), Layout(name="main", size=16), Layout(maca_panel, size=6)
    )
    layout["main"].split_row(Layout(code_panel, ratio=1), Layout(docker_panel, ratio=1))

    return layout


def run_demo() -> None:
    console.clear()

    with Live(
        generate_layout(
            "GENERATION",
            HALLUCINATED_CODE,
            "En attente...",
            "Préparation de l'environnement isolé...",
        ),
        refresh_per_second=4,
    ) as live:
        time.sleep(2)

        # 1. Exécution Docker
        live.update(
            generate_layout(
                "GENERATION",
                HALLUCINATED_CODE,
                "Exécution...",
                "Running python3 task_1.py...",
            )
        )
        time.sleep(1.5)

        # 2. Crash
        live.update(
            generate_layout(
                "CRASH", HALLUCINATED_CODE, "CRASH", TRACEBACK_ERROR, is_error=True
            )
        )
        time.sleep(3)

        # 3. MACA Délibération (Healing)
        live.update(
            generate_layout(
                "HEALING",
                HALLUCINATED_CODE,
                "Analyse...",
                "Analyse du traceback en cours...",
                is_error=True,
            )
        )
        time.sleep(3)

        # 4. Nouveau Code Généré
        live.update(
            generate_layout(
                "HEALING",
                CORRECTED_CODE,
                "Re-Exécution...",
                "Running python3 task_1_fixed.py...",
            )
        )
        time.sleep(2)

        # 5. Succès
        success_log = "Process completed successfully.\nMemory isolated.\nFound 0 errors in dummy log."
        live.update(
            generate_layout(
                "SUCCESS", CORRECTED_CODE, "SUCCESS", success_log, is_error=False
            )
        )
        time.sleep(2)


def main() -> None:
    """Point d'entrée principal pour l'exécution via CLI."""
    run_demo()


if __name__ == "__main__":
    main()
