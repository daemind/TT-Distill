# ruff: noqa
"""Demo 2: Autopsie Algébrique - MACA & DoRA Visualizer.

Cette démo présente l'architecture tensorielle de MACA (Multi-Agent Consensus Alignment)
et visualise le processus de délibération latente.

Architecture:
    Agent S2 → Latent Rollout → Hidden States → Sinkhorn Alignment → Barycentre
                                                                    ↓
                                                            TT-Distill Bridge
                                                                    ↓
                                                            DoRA Adapter (15 Mo)

Objectifs:
- Visualiser les trajectoires latentes des agents S2
- Montrer l'alignement Sinkhorn-Wasserstein
- Démontrer l'efficacité du DoRA adapter
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.logger import get_logger
from src.orchestration.maca import (
    LatentTrajectory,
    SinkhornBarycenter,
)

logger = get_logger(__name__)
console = Console()


@dataclass
class MACAVizState:
    """État de visualisation pour MACA."""

    agent_trajectories: list[LatentTrajectory]
    current_barycentre: np.ndarray | None
    consensus_score: float
    convergence_iteration: int
    elapsed_time: float
    alignment_matrix: np.ndarray | None


class MACADemo:
    """Démonstration interactive de MACA avec visualisation."""

    def __init__(self, num_agents: int = 3, seq_len: int = 10, latent_dim: int = 64):
        """Initialiser la démo MACA.

        Args:
            num_agents: Nombre d'agents S2 pour le consensus
            seq_len: Longueur des séquences latentes
            latent_dim: Dimension de l'espace latent
        """
        self.num_agents = num_agents
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.sinkhorn = SinkhornBarycenter()
        self.trajectories: list[LatentTrajectory] = []
        self.current_barycentre: np.ndarray | None = None
        self.consensus_score = 0.0
        self.convergence_iteration = 0
        self.start_time = 0.0
        self.is_running = False

    def _generate_agent_trajectory(self, agent_id: str) -> LatentTrajectory:
        """Générer une trajectoire latente pour un agent.

        Args:
            agent_id: Identifiant de l'agent

        Returns:
            Trajectoire latente générée
        """
        # Simuler des hidden states avec bruit contrôlé
        # Les agents divergent légèrement pour montrer le processus de consensus
        np.random.seed(hash(agent_id) % (2**32))
        hidden_states = np.random.randn(self.seq_len, self.latent_dim) * 0.5

        # Ajouter une tendance commune (pour montrer que le consensus est possible)
        common_signal = np.sin(np.linspace(0, 4 * np.pi, self.seq_len)).reshape(-1, 1)
        hidden_states += common_signal * 0.3

        return LatentTrajectory(
            agent_id=agent_id,
            hidden_states=hidden_states,
            confidence=np.random.uniform(0.8, 1.0),
            metadata={"generation_time": np.random.uniform(0.001, 0.01)},
        )

    def _compute_sinkhorn_alignment(
        self, trajectories: list[LatentTrajectory]
    ) -> tuple[np.ndarray, float, int]:
        """Calculer l'alignement Sinkhorn pour les trajectoires.

        Args:
            trajectories: Liste des trajectoires des agents

        Returns:
            Tuple (barycentre, consensus_score, convergence_iterations)
        """
        if not trajectories:
            return np.zeros(self.latent_dim), 0.0, 0

        # Utiliser la méthode compute_barycenter de SinkhornBarycenter
        consensus_result = self.sinkhorn.compute_barycenter(
            trajectories,
            weights=[t.confidence for t in trajectories],
        )

        # Extraire le barycentre (shape: (seq_len, latent_dim))
        # Pour la visualisation, utiliser la dernière couche
        barycentre = consensus_result.barycentre[-1] if len(consensus_result.barycentre) > 1 else consensus_result.barycentre[0]

        return barycentre, consensus_result.consensus_score, consensus_result.convergence_iterations

    def run_demo(self, iterations: int = 5) -> MACAVizState:
        """Exécuter la démo MACA avec visualisation.

        Args:
            iterations: Nombre d'itérations de consensus

        Returns:
            État final de visualisation
        """
        self.start_time = time.perf_counter()
        self.is_running = True

        # Générer les trajectoires initiales
        self.trajectories = [
            self._generate_agent_trajectory(f"agent_{i}")
            for i in range(self.num_agents)
        ]

        # Exécuter les itérations de consensus
        for _i in range(iterations):
            barycentre, consensus_score, _convergence_iter = (
                self._compute_sinkhorn_alignment(self.trajectories)
            )

            self.current_barycentre = barycentre
            self.consensus_score = consensus_score
            self.convergence_iteration = _i + 1

            # Simuler l'évolution des trajectoires vers le consensus
            if barycentre is not None:
                for traj in self.trajectories:
                    traj.hidden_states[-1] = (
                        traj.hidden_states[-1] * 0.7 + barycentre * 0.3
                    )
                    traj.confidence = min(1.0, traj.confidence * 1.02)  # Augmenter la confiance, max 100%

        self.is_running = False
        elapsed_time = time.perf_counter() - self.start_time

        # Calculer la matrice d'alignement pour la visualisation
        alignment_matrix = self._compute_alignment_matrix()

        return MACAVizState(
            agent_trajectories=self.trajectories,
            current_barycentre=self.current_barycentre,
            consensus_score=self.consensus_score,
            convergence_iteration=self.convergence_iteration,
            elapsed_time=elapsed_time,
            alignment_matrix=alignment_matrix,
        )

    def _compute_alignment_matrix(self) -> np.ndarray | None:
        """Calculer la matrice d'alignement pour la visualisation.

        Returns:
            Matrice de similarité entre agents
        """
        min_trajectories = 2
        if len(self.trajectories) < min_trajectories:
            return None

        # Calculer les similarités cosinus entre les trajectoires
        n_agents = len(self.trajectories)
        alignment_matrix = np.zeros((n_agents, n_agents))

        for i, traj_i in enumerate(self.trajectories):
            for j, traj_j in enumerate(self.trajectories):
                if i == j:
                    alignment_matrix[i, j] = 1.0
                else:
                    # Similarité cosinus sur les dernières hidden states
                    sim = np.dot(
                        traj_i.hidden_states[-1], traj_j.hidden_states[-1]
                    ) / (
                        np.linalg.norm(traj_i.hidden_states[-1])
                        * np.linalg.norm(traj_j.hidden_states[-1])
                        + 1e-8
                    )
                    alignment_matrix[i, j] = (sim + 1) / 2  # Normaliser à [0, 1]

        return alignment_matrix

    def print_visualization(self, state: MACAVizState) -> None:
        """Afficher la visualisation MACA.

        Args:
            state: État de visualisation
        """
        console.print("\n" + "=" * 70)
        console.print("[bold magenta]🧮 Autopsie Algébrique - MACA & DoRA Visualizer[/bold magenta]")
        console.print("=" * 70)

        # Tableau des agents
        table = Table(title="Trajectoires des Agents S2")
        table.add_column("Agent", style="cyan")
        table.add_column("Confiance", style="green")
        table.add_column("Latence", style="yellow")
        table.add_column("Dimension", style="blue")

        for traj in state.agent_trajectories:
            table.add_row(
                traj.agent_id,
                f"{traj.confidence:.2%}",
                f"{traj.metadata.get('generation_time', 0)*1000:.2f} ms",
                f"{traj.latent_dim}D",
            )

        console.print(table)

        # Tableau du consensus
        consensus_table = Table(title="Résultat du Consensus MACA")
        consensus_table.add_column("Métrique", style="cyan")
        consensus_table.add_column("Valeur", style="green")

        consensus_table.add_row(
            "Score de consensus",
            f"{state.consensus_score:.4f}",
        )
        consensus_table.add_row(
            "Itérations de convergence",
            f"{state.convergence_iteration}",
        )
        consensus_table.add_row(
            "Temps écoulé",
            f"{state.elapsed_time:.3f} s",
        )
        consensus_table.add_row(
            "Dimension du barycentre",
            f"{state.current_barycentre.shape[0] if state.current_barycentre is not None else 'N/A'}D",
        )

        console.print(consensus_table)

        # Matrice d'alignement
        if state.alignment_matrix is not None:
            console.print("\n[bold yellow]Matrice d'Alignement (Similarité Cosinus)[/bold yellow]")
            console.print(
                "[dim]Valeurs: 0.0 (divergent) → 1.0 (parfaitement aligné)[/dim]"
            )

            # Afficher la matrice sous forme de tableau
            matrix_table = Table(show_header=False, box=None)
            matrix_table.add_column("Agent", style="cyan")
            for i in range(state.alignment_matrix.shape[1]):
                matrix_table.add_column(f"Agent {i}", justify="right", style="green")

            for i, traj in enumerate(state.agent_trajectories):
                row = [traj.agent_id]
                for j in range(state.alignment_matrix.shape[1]):
                    val = state.alignment_matrix[i, j]
                    # Colorer selon la valeur
                    if val > 0.8:
                        row.append(Text(f"{val:.2f}", style="bold green"))
                    elif val > 0.5:
                        row.append(Text(f"{val:.2f}", style="yellow"))
                    else:
                        row.append(Text(f"{val:.2f}", style="red"))
                matrix_table.add_row(*row)

        # Information sur DoRA
        console.print("\n[bold cyan]🔧 DoRA Adapter (15 Mo)[/bold cyan]")
        console.print("[dim]Bridge TT-Distill: Injection directe d'inputs_embeds[/dim]")
        console.print("[dim]Objectif: 87 Hz (~12 ms) pour Qwen2.5-VL-3B-Instruct[/dim]")

        # Résumé
        console.print("\n" + "=" * 70)
        console.print(
            f"[bold green]✅ Consensus atteint: {state.consensus_score:.2%}[/bold green]"
        )
        console.print(
            "[dim]Alignement tensoriel via Sinkhorn-Wasserstein barycenter[/dim]"
        )
        console.print("=" * 70)


def main() -> None:
    """Point d'entrée principal pour l'exécution via CLI."""
    _main()


def _main() -> None:
    """Point d'entrée principal."""
    console.print("\n" + "=" * 70)
    console.print("[bold magenta]🧠 Autopsie Algébrique[/bold magenta]")
    console.print("[dim]MACA & DoRA Visualizer - Compression mathématique pure[/dim]")
    console.print("=" * 70)

    # Initialiser la démo MACA
    demo = MACADemo(num_agents=4, seq_len=10, latent_dim=64)

    # Exécuter la démo
    state = demo.run_demo(iterations=5)

    # Afficher la visualisation
    demo.print_visualization(state)


if __name__ == "__main__":
    main()
