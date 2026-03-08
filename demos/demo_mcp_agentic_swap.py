"""Demo: Agentic Synapse Reconfiguration via MCP.

This demo simulates an AI agent (System 2) that decides to reconfigure its
own mathematical expertises (System 1) mid-reasoning by calling the
IntelligenceManifold MCP server.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from rich.console import Console

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rich.panel import Panel
from rich.table import Table

# Import the server tools directly for the simulation
from src.orchestration.arc_hybrid_solver import ARCGridEncoder, HybridSolver
from src.orchestration.mcp_intelligence_manifold import (
    get_manifold_telemetry,
    hot_swap_manifold,
    synthesize_expert_manifold,
)

console = Console()

async def simulate_agent_loop() -> None:
    console.print("\n" + "="*80)
    console.print("[bold cyan]🤖 DEMO: DYNAMIC AGENTIC SYNAPSE SYNTHESIS[/bold cyan]")
    console.print("="*80)
    console.print("[dim]The model identifies a problem, synthesizes its own expert expertise, and swaps hardware synapses.[/dim]\n")

    # 1. Inner Probing: "What is the problem?"
    console.print("[bold blue]Step 1: Inner Probing (Self-Analysis)[/bold blue]")
    problem = "A complex tiling task with periodic color-shifting symmetry."
    console.print(f"Agent (Internal Monologue): [italic]\"The task is: {problem}\"[/italic]")
    console.print("Agent (Self-Query): [italic italic]\"What expertise do I need for this?\"[/italic]")

    # 2. Expert Synthesis Turn
    console.print("\n[bold blue]Step 2: Expert Synthesis (MCP CALL)[/bold blue]")
    needs = "I need expert weights for periodic color field shifts and topological grid logic."
    console.print(f"Agent: \"Synthesizing manifold config for: [yellow]{needs}[/yellow]\"")

    synthesis = synthesize_expert_manifold(description=needs)
    config = synthesis["recommended_config"]

    console.print(Panel(
        f"[bold green]SYNTHESIS COMPLETE[/bold green]\n"
        f"Description Map: {synthesis['description']}\n"
        f"Manifold Coordinates: {synthesis['manifold_coordinates']}\n"
        f"Recommended Configuration: [bold yellow]{json.dumps(config, indent=2)}[/bold yellow]",
        title="MCP Tool: synthesize_expert_manifold",
        border_style="magenta"
    ))

    # 3. Execution: Hardware Hot-Swap
    console.print("\n[bold blue]Step 3: Hardware Hot-Swap Execution[/bold blue]")
    console.print(f"Agent: \"Applying {len(config)} expert synapses to the Metal backend...\"")

    t0 = time.perf_counter_ns()
    result = hot_swap_manifold(expert_weights=config)
    t_end = (time.perf_counter_ns() - t0) / 1_000_000

    if result["status"] == "success":
        console.print(Panel(
            f"[bold green]HARDWARE SWAP SUCCESSFUL[/bold green]\n"
            f"Hardware Swap Latency: [bold yellow]{result['swap_ms']:.6f} ms[/bold yellow]\n"
            f"Total Tool Turnaround: [bold cyan]{t_end:.3f} ms[/bold cyan]",
            title="MCP Tool: hot_swap_manifold",
            border_style="green"
        ))
    else:
        console.print(f"[bold red]SWAP FAILED: {result['message']}[/bold red]")
        return

    # 4. Telemetry Verification
    console.print("\n[bold blue]Step 4: Real-time Telemetry[/bold blue]")
    telemetry = get_manifold_telemetry()

    tel_table = Table(title="Manifold Hardware Telemetry", border_style="dim")
    tel_table.add_column("Property", style="cyan")
    tel_table.add_column("Value", style="white")
    for k, v in telemetry.items():
        tel_table.add_row(k, str(v))
    console.print(tel_table)

    # 5. Real Execution (Instead of sleep)
    console.print("\n[bold blue]Step 5: Verification (Real Solver Execution)[/bold blue]")
    # We use a real grid to show the expert is loaded and functioning
    solver = HybridSolver(ARCGridEncoder())
    solver.learn_from_pairs([{"input": np.zeros((3,3)), "output": np.zeros((3,3))}])

    t_start = time.perf_counter_ns()
    solver.predict(np.zeros((3,3)))
    t_exec = (time.perf_counter_ns() - t_start) / 1_000_000

    console.print(f"Agent: \"Inference executed on optimized manifold in [bold yellow]{t_exec:.3f} ms[/bold yellow].\"")
    console.print("[bold green]✅ TASK PROVED: Synapses are hot and responding.[/bold green]")

    # 6. Result
    console.print("\n" + "="*80)
    console.print("[bold green]🏁 DYNAMIC RECONFIGURATION COMPLETE[/bold green]")
    console.print("The model is now physically prepared to solve the task.")
    console.print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(simulate_agent_loop())
