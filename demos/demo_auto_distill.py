import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from rich.console import Console
from src.agent_factory import AgentFactory
from src.agent_spawner import AgentSpawner
from src.db_manager import DBManager
from src.llm_client import LLMClient
from src.mcp_manager import MCPManager

console = Console()

async def run_auto_distill_demo() -> None:
    console.print("[bold blue]🚀 Starting Auto-Distillation Demo (35B Model)[/bold blue]")

    # 1. Initialize Infrastructure
    db_path = "data/distill_demo.db"
    db = DBManager(db_path)
    await db.initialize()

    mcp = MCPManager(project_root=str(Path.cwd()))
    # LLMClient will use env vars: LLM_MODEL and LLM_SERVER_URL
    llm = LLMClient()

    spawner = AgentSpawner(persistence=db, llm_client=llm)

    # 2. Create the Agent (Strategist has the 'crystallize_weights' right)
    agent = AgentFactory.create_agent("Strategist")
    agent.name = "Oracle-35B"

    console.print(f"🤖 Spawning Agent: [bold]{agent.name}[/bold] ([italic]{agent.role}[/italic])")
    console.print(f"🔒 Rights: [yellow]{agent.rights}[/yellow]")

    # 3. Define the Task
    task = (
        "OBJECTIVE: Perform an autonomous hardware reconfiguration.\n"
        "1. Analyze the context: We need to optimize for geometric tiling tasks.\n"
        "2. Identify the required adapters: 'geometric' and 'tiling'.\n"
        "3. Trigger a hardware weight crystallization using the 'crystallize_to_weights' tool.\n"
        "   - Use 0.7 weight for 'geometric'\n"
        "   - Use 0.3 weight for 'tiling'\n"
        "4. Confirm the success of the manifold merger."
    )

    # 4. Execute the loop
    project_id = 1
    result = await spawner.spawn_single(
        agent=agent,
        task=task,
        project_id=project_id,
        mcp_manager=mcp
    )

    console.print("\n" + "="*50)
    console.print("✅ [bold green]Distillation Loop Complete[/bold green]")
    console.print(f"Status: {'[bold green]SUCCESS[/bold green]' if result['success'] else '[bold red]FAILED[/bold red]'}")
    console.print("="*50)

    if result['success']:
        console.print("\n[bold]Final Response Extract:[/bold]")
        console.print(result['response'][-500:])
    else:
        console.print(f"[bold red]Error:[/bold red] {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    # Ensure environment is set up for the 35B model
    if not os.getenv("LLM_SERVER_URL"):
        os.environ["LLM_SERVER_URL"] = "http://localhost:8001/v1"
    if not os.getenv("LLM_MODEL"):
        os.environ["LLM_MODEL"] = "unsloth/Qwen3-Coder-Next"

    asyncio.run(run_auto_distill_demo())
