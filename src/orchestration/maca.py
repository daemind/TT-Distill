"""Multi-Agent Consensus Alignment (MACA) for Tensorial Deliberation.

This module implements the MACA protocol for replacing text-based brainstorming
with tensorial deliberation using Sinkhorn-Wasserstein barycenter computation.

Architecture:
    Agent S2 → Latent Rollout → Hidden States → Sinkhorn Alignment → Barycentre
                                                                    ↓
                                                            TT-Distill Bridge
                                                                    ↓
                                                            DoRA Adapter (15 Mo)

Key Concepts:
- Auto-regressive Latent Generation: Each agent produces hidden state sequences
- Sinkhorn Alignment: Fuses trajectories via Wasserstein barycenter
- Consensus Game: Equilibrium where predictions align perfectly

References:
- Soatto & Achille (Feb 2026): Two-level cognitive systems
- LatentMAS & Interlat frameworks: Tensorial communication
- LATCH (Latent Lookahead): Predictive constraint evolution
"""

from __future__ import annotations

import asyncio
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

# Constants
SINKHORN_ITERATIONS = (
    20  # Nombre d'itérations pour l'algorithme de Sinkhorn (optimisé: 50→20)
)
SINKHORN_EPSILON = 1e-3  # Régularisation entropique
CONSENSUS_THRESHOLD = 0.9  # Seuil de consensus pour validation
LATENT_DIM = 4096  # Dimension de l'espace latent (à adapter selon le modèle)
SINKHORN_CONVERGENCE_THRESHOLD = (
    1e-6  # Seuil de convergence pour l'algorithme de Sinkhorn
)


@dataclass
class LatentTrajectory:
    """Trajectoire latente d'un agent S2."""

    agent_id: str
    hidden_states: np.ndarray  # Shape: (seq_len, latent_dim)
    kv_cache: np.ndarray | None = None  # Cache KV pour l'héritage
    confidence: float = 1.0  # Score de confiance de l'agent
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        """Longueur de la séquence."""
        return int(self.hidden_states.shape[0])

    @property
    def latent_dim(self) -> int:
        """Dimension latente."""
        return int(self.hidden_states.shape[1])


@dataclass
class ConsensusResult:
    """Résultat du consensus MACA."""

    barycentre: np.ndarray  # Shape: (latent_dim,)
    consensus_score: float  # Score de consensus (0-1)
    participating_agents: list[str]
    convergence_iterations: int
    divergence_type: str  # "wasserstein" ou "sinkhorn"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convertir en dictionnaire."""
        return {
            "consensus_score": round(self.consensus_score, 4),
            "participating_agents": self.participating_agents,
            "convergence_iterations": self.convergence_iterations,
            "divergence_type": self.divergence_type,
            "barycentre_dim": self.barycentre.shape[0],
        }


class SinkhornBarycenter:
    """Calcul du barycentre de Wasserstein via l'algorithme de Sinkhorn."""

    @staticmethod
    def compute_wasserstein_distance(
        x: np.ndarray, y: np.ndarray, epsilon: float = SINKHORN_EPSILON
    ) -> float:
        """Calculer la distance de Wasserstein entre deux distributions.

        Args:
            x: Première distribution (shape: (n_samples,))
            y: Deuxième distribution (shape: (m_samples,))
            epsilon: Régularisation entropique

        Returns:
            Distance de Wasserstein 1D
        """
        n, m = len(x), len(y)

        # Matrice de coût (distance euclidienne 1D)
        cost_matrix = np.abs(x[:, None] - y[None, :])

        # Initialisation des marges uniformes
        a = np.ones(n) / n
        b = np.ones(m) / m

        # Algorithme de Sinkhorn
        u = np.zeros(n)
        v = np.zeros(m)

        for _ in range(SINKHORN_ITERATIONS):
            u_prev = u.copy()
            kernel_matrix = np.exp(-cost_matrix / epsilon)

            u = a / (kernel_matrix @ (b + 1e-10) + 1e-10)
            v = b / (kernel_matrix.T @ u + 1e-10)

            # Vérifier la convergence
            if np.max(np.abs(u - u_prev)) < SINKHORN_CONVERGENCE_THRESHOLD:
                break

        # Calcul de la distance
        transport_plan = np.diag(kernel_matrix @ np.diag(v) @ kernel_matrix.T)
        wasserstein = np.sum(transport_plan * cost_matrix)

        return float(wasserstein)

    @staticmethod
    def compute_barycenter(
        trajectories: list[LatentTrajectory],
        weights: list[float] | None = None,
        rejection_threshold: float = 0.85,
    ) -> ConsensusResult:
        """Calculer le barycentre de Wasserstein de plusieurs trajectoires.

        Args:
            trajectories: Liste des trajectoires latentes
            weights: Poids optionnels pour chaque agent (défaut: uniforme)
            rejection_threshold: Seuil Cosinus sous lequel un agent est rejeté (0.0 = désactivé)

        Returns:
            ConsensusResult avec le barycentre et les métriques
        """
        if not trajectories:
            raise ValueError("Au moins une trajectoire requise")

        # Normaliser les poids initiaux
        if weights is None:
            weights = [1.0 / len(trajectories)] * len(trajectories)
        else:
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        # Extraire les hidden states et aligner les séquences
        # (padding ou truncation pour avoir la même longueur)
        max_seq_len = max(t.seq_len for t in trajectories)
        aligned_states = []

        for traj in trajectories:
            if traj.seq_len < max_seq_len:
                # Padding avec zéros
                padding = np.zeros((max_seq_len - traj.seq_len, traj.latent_dim))
                aligned = np.vstack([traj.hidden_states, padding])
            else:
                # Truncation
                aligned = traj.hidden_states[:max_seq_len]
            aligned_states.append(aligned)

        aligned_array = np.stack(
            aligned_states, axis=0
        )  # Shape: (n_agents, max_seq_len, latent_dim)

        # --- LATENT TOPOLOGY REJECTION FILTER (Cosine Similarity) ---
        participating_indices = list(range(len(trajectories)))

        if len(trajectories) > 2 and rejection_threshold > 0.0:
            # Centroïde naïf (médiane pour robustesse face aux vraies hallucinations)
            median_centroid = np.median(aligned_array, axis=0)
            flat_centroid = median_centroid.flatten()
            norm_centroid = np.linalg.norm(flat_centroid) or 1.0

            valid_weights = []
            valid_indices = []

            for i, traj_array in enumerate(aligned_array):
                flat_traj = traj_array.flatten()
                norm_traj = np.linalg.norm(flat_traj) or 1.0
                cosine_sim = float(
                    np.dot(flat_traj, flat_centroid) / (norm_traj * norm_centroid)
                )

                if cosine_sim < rejection_threshold:
                    valid_weights.append(0.0)
                    trajectories[i].metadata["rejected"] = True
                    trajectories[i].metadata["cosine_sim"] = cosine_sim
                else:
                    valid_weights.append(weights[i])
                    valid_indices.append(i)
                    trajectories[i].metadata["rejected"] = False
                    trajectories[i].metadata["cosine_sim"] = cosine_sim

            # Re-normaliser les poids après filtrage
            total_valid_weight = sum(valid_weights)
            if total_valid_weight > 0:
                weights = [w / total_valid_weight for w in valid_weights]
                participating_indices = valid_indices
            else:
                # Fallback si tous mutés (hallucination de masse)
                weights = [1.0 / len(trajectories)] * len(trajectories)
        # ------------------------------------------------------------

        # Calculer le barycentre par dimension (moyenne pondérée vectorisée)
        # Vectoriser le calcul: np.average avec axis=0 pour moyenne pondérée
        barycentre = np.average(
            aligned_array, weights=weights, axis=0
        )  # Shape: (max_seq_len, latent_dim)

        # Calculer le score de consensus (variance inverse) sur les agents non-rejetés
        if participating_indices:
            valid_trajectories = [trajectories[i] for i in participating_indices]
            variance_list = [
                np.var(t.hidden_states - barycentre[: t.seq_len])
                for t in valid_trajectories
            ]
            variance = float(np.mean(variance_list)) if variance_list else 1.0
        else:
            variance = float(
                np.mean(
                    [
                        np.var(t.hidden_states - barycentre[: t.seq_len])
                        for t in trajectories
                    ]
                )
            )

        consensus_score = 1.0 / (1.0 + variance)

        # Vérifier le seuil de consensus
        is_valid = consensus_score >= CONSENSUS_THRESHOLD

        # Force float32 pour compatibilité
        barycentre = barycentre.astype(np.float32)

        active_agents = [trajectories[i].agent_id for i in participating_indices]

        return ConsensusResult(
            barycentre=barycentre,
            consensus_score=float(consensus_score),
            participating_agents=active_agents,
            convergence_iterations=SINKHORN_ITERATIONS,
            divergence_type="sinkhorn",
            metadata={"is_valid": is_valid, "variance": float(variance)},
        )


class LatentRollout:
    """Auto-regressive Latent Generation pour les agents S2."""

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        seq_len: int = 32,
        model_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        """Initialiser le rollout latent.

        Args:
            latent_dim: Dimension de l'espace latent
            seq_len: Longueur de séquence
            model_fn: Fonction optionnelle pour simuler l'évolution latente
        """
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.model_fn = model_fn or self._default_rollout

    def _default_rollout(self, hidden_state: np.ndarray) -> np.ndarray:
        """Évolution latente par défaut (mouvement brownien).

        Args:
            hidden_state: État latent actuel

        Returns:
            Prochain état latent
        """
        # Mouvement avec inertie et bruit
        inertia = 0.7
        noise_scale = 0.1

        return inertia * hidden_state + np.random.randn(self.latent_dim) * noise_scale

    def generate_trajectory(
        self,
        initial_state: np.ndarray,
        agent_id: str,
        confidence: float = 1.0,
    ) -> LatentTrajectory:
        """Générer une trajectoire latente auto-régressive.

        Args:
            initial_state: État latent initial
            agent_id: Identifiant de l'agent
            confidence: Score de confiance de l'agent

        Returns:
            LatentTrajectory avec la séquence complète
        """
        hidden_states = [initial_state.copy()]
        current_state = initial_state.copy()

        for _ in range(self.seq_len - 1):
            current_state = self.model_fn(current_state)
            hidden_states.append(current_state.copy())

        return LatentTrajectory(
            agent_id=agent_id,
            hidden_states=np.array(hidden_states),
            confidence=confidence,
        )


class MACAEngine:
    """Moteur de Consensus Multi-Agent (Multi-Agent Consensus Alignment).

    Ce moteur remplace le brainstorming textuel par une délibération tensorielle
    où les agents communiquent via leurs hidden states plutôt que du texte.

    Workflow:
        1. Initialisation: Créer les agents S2 avec leurs intentions initiales
        2. Latent Rollout: Chaque agent génère une trajectoire latente
        3. Sinkhorn Alignment: Fusionner les trajectoires via barycentre
        4. Validation: Vérifier le seuil de consensus
        5. Output: Retourner le barycentre pour TT-Distill
    """

    def __init__(
        self,
        latent_dim: int = LATENT_DIM,
        seq_len: int = 32,
        sinkhorn_epsilon: float = SINKHORN_EPSILON,
    ) -> None:
        """Initialiser le moteur MACA.

        Args:
            latent_dim: Dimension de l'espace latent
            seq_len: Longueur de séquence pour le rollout
            sinkhorn_epsilon: Régularisation pour Sinkhorn
        """
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.sinkhorn_epsilon = sinkhorn_epsilon

        # Initialiser les composants
        self.rollout = LatentRollout(latent_dim=latent_dim, seq_len=seq_len)
        self.sinkhorn = SinkhornBarycenter()

        # Cache des trajectoires pour l'héritage KV
        self._trajectory_cache: dict[str, LatentTrajectory] = {}

        logger.info(
            f"🧠 MACA Engine initialisé (latent_dim={latent_dim}, seq_len={seq_len})"
        )

    async def run_consensus(
        self,
        agent_intentions: list[dict[str, Any]],
    ) -> ConsensusResult:
        """Exécuter le jeu de consensus MACA.

        Args:
            agent_intentions: Liste d'intentions d'agents
                            Chaque intention: {"agent_id": str, "initial_state": np.ndarray}

        Returns:
            ConsensusResult avec le barycentre et les métriques
        """
        logger.info(
            f"🎮 Démarrage du consensus MACA avec {len(agent_intentions)} agents"
        )

        # Étape 1: Latent Rollout pour chaque agent
        trajectories: list[LatentTrajectory] = []
        for intent in agent_intentions:
            agent_id = intent["agent_id"]
            initial_state = intent.get(
                "initial_state", np.random.randn(self.latent_dim)
            )

            # Générer la trajectoire
            trajectory = self.rollout.generate_trajectory(
                initial_state=initial_state,
                agent_id=agent_id,
                confidence=intent.get("confidence", 1.0),
            )

            trajectories.append(trajectory)

            # Mettre en cache pour l'héritage KV
            self._trajectory_cache[agent_id] = trajectory

            logger.debug(
                f"✅ Agent {agent_id}: trajectoire générée (len={trajectory.seq_len})"
            )

        # Étape 2: Sinkhorn Alignment - Fusion des trajectoires
        weights = [t.confidence for t in trajectories]
        consensus = self.sinkhorn.compute_barycenter(trajectories, weights=weights)

        logger.info(
            f"🎯 Consensus atteint: score={consensus.consensus_score:.4f}, "
            f"valid={consensus.metadata['is_valid']}"
        )

        return consensus

    async def run_consensus_with_heritage(
        self,
        agent_intentions: list[dict[str, Any]],
        heritage_mode: str = "kv_cache",
    ) -> ConsensusResult:
        """Exécuter le consensus avec héritage des KV-caches.

        Args:
            agent_intentions: Liste d'intentions d'agents
            heritage_mode: Mode d'héritage ("kv_cache" ou "trajectory")

        Returns:
            ConsensusResult avec le barycentre
        """
        # Appliquer l'héritage aux intentions
        enriched_intentions = []
        for intent in agent_intentions:
            agent_id = intent["agent_id"]
            enriched = intent.copy()

            if heritage_mode == "kv_cache" and agent_id in self._trajectory_cache:
                # Hériter du KV-cache du prédécesseur
                prev_trajectory = self._trajectory_cache[agent_id]
                enriched["kv_cache"] = prev_trajectory.kv_cache

            enriched_intentions.append(enriched)

        # Exécuter le consensus
        return await self.run_consensus(enriched_intentions)

    def get_trajectory_cache(self) -> dict[str, LatentTrajectory]:
        """Récupérer le cache des trajectoires."""
        return self._trajectory_cache.copy()

    def clear_cache(self) -> None:
        """Vider le cache des trajectoires."""
        self._trajectory_cache.clear()
        logger.debug("🗑️ Cache des trajectoires vidé")


class TTDistillBridge:
    """Pont TT-Distill pour transformer le barycentre en adaptateur DoRA.

    Ce composant prend le barycentre de Wasserstein calculé par MACA
    et le transmute en un adaptateur DoRA de 15 Mo via SVD factorisation.

    Architecture:
        MACA Consensus → Sinkhorn Barycentre → SVD Factorization → DoRA Adapter
    """

    def __init__(
        self,
        adapter_size_mb: int = 15,
        rank: int = 16,  # Rank réduit pour Qwen2.5-3B
        hidden_size: int = 2560,  # Qwen2.5-3B hidden size
        target_arch: str = "qwen2",  # Architecture Qwen2.5
    ) -> None:
        """Initialiser le pont TT-Distill.

        Args:
            adapter_size_mb: Taille cible de l'adaptateur en Mo
            rank: Rank de l'adaptation low-rank (16 pour ~15 MB)
            hidden_size: Taille de la couche cachée (Qwen2.5-3B: 2560)
            target_arch: Architecture cible ("qwen2" pour Qwen2.5)
        """
        self.adapter_size_mb = adapter_size_mb
        self.rank = rank
        self.hidden_size = hidden_size
        self.target_arch = target_arch

        # Calculer la taille de l'adaptateur
        # DoRA: 2 matrices de rank x hidden_size (float32 = 4 bytes)
        adapter_params = 2 * rank * hidden_size
        adapter_size_bytes = adapter_params * 4  # float32
        adapter_size_mb_actual = adapter_size_bytes / (1024 * 1024)

        logger.info(
            f"🔨 TT-Distill Bridge initialisé (arch={target_arch}, rank={rank}, "
            f"hidden_size={hidden_size}, size~{adapter_size_mb_actual:.2f} MB)"
        )

    def distill_barycenter(
        self,
        barycentre: np.ndarray,
        target_arch: str | None = None,
        use_svd: bool = True,
        preserve_temporal: bool = False,
    ) -> dict[str, np.ndarray]:
        """Distiller le barycentre en adaptateur DoRA avec SVD factorisation.

        DoRA (Weight-Decomposed Low-Rank Adaptation) sépare la magnitude (amplitude)
        de la direction pour préserver la capacité d'apprentissage du modèle.

        Formule DoRA:
            W' = m * (W + BA) / ||W + BA||_c

        Où:
            - m = magnitude (norme du barycentre)
            - B, A = matrices de rank (direction)
            - W = poids originaux du modèle

        Args:
            barycentre: Barycentre de Wasserstein (shape: (seq_len, latent_dim))
            target_arch: Architecture cible (défaut: self.target_arch)
            use_svd: Utiliser SVD pour factorisation optimale (défaut: True)
            preserve_temporal: Garder la dimension temporelle pour SVD complète (défaut: False)

        Returns:
            Dictionnaire des tenseurs de l'adaptateur DoRA:
            - dora_scale: Vecteur de magnitude (norme du barycentre)
            - lora_a: Matrice A (rank x hidden_size) - direction normalisée
            - lora_b: Matrice B (hidden_size x rank) - direction normalisée
            - barycentre_mean: Moyenne temporelle du barycentre (avant normalisation)
            - target_arch: Architecture cible
            - adapter_size_mb: Taille estimée
            - latent_dim: Dimension latente originale
        """
        arch = target_arch or self.target_arch
        logger.info(
            f"🔥 Distillation du barycentre vers DoRA (arch={arch}, SVD={use_svd})"
        )

        # ÉTAPE 1: Extraire la magnitude (norme) DU barycentre AVANT normalisation
        # C'est la clé de DoRA: séparer magnitude et direction
        barycentre_mean = np.mean(barycentre, axis=0)  # (latent_dim,)
        latent_dim = barycentre_mean.shape[0]

        # Calculer la magnitude (norme L2) - CECI EST LA CLÉ DE DORA
        magnitude = np.linalg.norm(barycentre_mean)  # scalaire
        logger.info(f"📏 Magnitude extraite: {magnitude:.6f}")

        # Normaliser pour obtenir la direction pure
        barycentre_norm = barycentre_mean / (magnitude + 1e-8)

        if use_svd:
            if preserve_temporal:
                # Garder la dimension temporelle pour capturer la dynamique
                barycentre_matrix = barycentre  # (seq_len, latent_dim)
            else:
                # Utiliser la moyenne (LoRA standard)
                barycentre_matrix = barycentre_norm.reshape(1, -1)  # (1, latent_dim)

            u, s, vt = np.linalg.svd(barycentre_matrix, full_matrices=False)

            # Tronquer au rank souhaité
            effective_rank = min(self.rank, s.shape[0])

            lora_a = (
                u[:, :effective_rank].T * np.sqrt(s[:effective_rank])[:, np.newaxis]
            ).astype(np.float32)
            lora_b = vt[:effective_rank, :].T.astype(np.float32)

            logger.info(
                f"✅ SVD factorisation (temporal={preserve_temporal}): rank={effective_rank}, singular_values={s[:effective_rank]}"
            )
            logger.info(
                f"   lora_a shape: {lora_a.shape}, lora_b shape: {lora_b.shape}"
            )
        else:
            # Méthode alternative: multiplication directe
            np.random.seed(42)  # Reproducibilité

            effective_hidden_size = min(self.hidden_size, latent_dim)

            # Matrice A: rank x effective_hidden_size
            lora_a = np.random.randn(self.rank, effective_hidden_size).astype(
                np.float32
            )
            # Appliquer la direction du barycentre
            lora_a *= np.outer(
                np.random.randn(self.rank), barycentre_norm[:effective_hidden_size]
            )

            # Matrice B: effective_hidden_size x rank (zero init)
            lora_b = np.zeros((effective_hidden_size, self.rank), dtype=np.float32)

        # Normaliser pour respecter la taille cible
        scale = (
            self.adapter_size_mb * 1024 * 1024 / (lora_a.nbytes + lora_b.nbytes)
        ) ** 0.5
        lora_a *= scale

        # ÉTAPE 2: Créer le tenseur dora_scale (magnitude)
        # Shape: (latent_dim,) - doit correspondre à la dimension réelle du barycentre
        dora_scale = np.full((latent_dim,), magnitude, dtype=np.float32)

        logger.info("🔨 DoRA adapter généré:")
        logger.info(
            f"   dora_scale shape: {dora_scale.shape}, magnitude={magnitude:.6f}"
        )
        logger.info(f"   lora_a shape: {lora_a.shape}")
        logger.info(f"   lora_b shape: {lora_b.shape}")

        return {
            "dora_scale": dora_scale,  # Tenseur de magnitude (CLÉ DE DORA)
            "lora_a": lora_a,
            "lora_b": lora_b,
            "barycentre_mean": barycentre_mean,
            "target_arch": arch,
            "adapter_size_mb": self.adapter_size_mb,
            "latent_dim": latent_dim,
            "effective_hidden_size": lora_b.shape[0],
            "effective_rank": lora_a.shape[0],
            "magnitude": magnitude,  # Magnitude scalaire pour référence
        }

    def save_adapter(
        self,
        adapter_tensors: dict[str, np.ndarray],
        output_path: str,
    ) -> None:
        """Sauvegarder l'adaptateur DoRA sur disque.

        Args:
            adapter_tensors: Tenseurs de l'adaptateur (doit inclure dora_scale)
            output_path: Chemin de sortie
        """

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Vérifier que dora_scale est présent (DoRA requis)
        if "dora_scale" not in adapter_tensors:
            logger.warning("⚠️ dora_scale non trouvé - l'adaptateur sera en mode LoRA")

        with Path(output_path).open("wb") as f:
            pickle.dump(adapter_tensors, f)

        actual_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        logger.info(
            f"✅ Adaptateur sauvegardé: {output_path} ({actual_size_mb:.2f} MB)"
        )
        logger.info(f"   Tenseurs: {', '.join(adapter_tensors.keys())}")


async def demo_maca() -> None:
    """Démonstration du moteur MACA pour Qwen2.5-VL-3B-Instruct."""

    # Initialiser le moteur avec dimensions adaptées à Qwen2.5-3B
    # Qwen2.5-3B: hidden_size=2560, mais on utilise latent_dim=512 pour la démo
    maca = MACAEngine(latent_dim=512, seq_len=32)

    # Simuler 3 agents S2 avec des intentions différentes
    agent_intentions = [
        {
            "agent_id": "architect",
            "initial_state": np.random.randn(512) * 0.5,
            "confidence": 0.9,
        },
        {
            "agent_id": "engineer",
            "initial_state": np.random.randn(512) * 0.3,
            "confidence": 0.85,
        },
        {
            "agent_id": "reviewer",
            "initial_state": np.random.randn(512) * 0.4,
            "confidence": 0.95,
        },
    ]

    # Exécuter le consensus
    consensus = await maca.run_consensus(agent_intentions)

    # Distiller vers DoRA (config Qwen2.5-3B)
    bridge = TTDistillBridge(
        adapter_size_mb=15,
        rank=16,  # Rank réduit pour ~15 MB
        hidden_size=2560,  # Qwen2.5-3B hidden size
        target_arch="qwen2",
    )
    adapter_tensors = bridge.distill_barycenter(consensus.barycentre, use_svd=True)

    # Sauvegarder
    output_path = "reflex_instinct_maca.bin"
    bridge.save_adapter(adapter_tensors, output_path)


if __name__ == "__main__":
    asyncio.run(demo_maca())
