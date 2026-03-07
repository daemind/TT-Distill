"""TT-Distill End-to-End Integration Module.

This module provides the complete integration of the TT-Distill architecture:
- MACA Engine (Multi-Agent Consensus Alignment)
- TTDistillBridge (DoRA adapter generation)
- Reflex Engine (S1 Product/Cerebellum)
- Vector Memory (CocoIndex RAG)

Architecture:
    S2 Resolver → MACA → TTDistillBridge → DoRA Adapter
                                  ↓
    S1 Product → Reflex Engine → Vector Memory → Action
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from reflex_engine import MultimodalReflexEngine
from src.logger import get_logger
from src.orchestration.maca import ConsensusResult, MACAEngine
from src.orchestration.maca_salon_bridge import MACASalonBridge
from src.orchestration.moa_gating import MoAGater
from src.orchestration.post_silicon import PostSiliconController
from src.vector_memory import VectorMemory

logger = get_logger(__name__)

# Magic value constants
MIN_METRICS_FOR_STATS = 10


@dataclass
class TTDistillConfig:
    """Configuration pour l'intégration TT-Distill."""

    # Modèle S1 (Produit/Cervelet)
    model_path: str = "Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
    lora_path: str | None = "reflex_instinct_15MB.bin"

    # MACA Engine
    maca_n_agents: int = 4
    maca_rank: int = 16
    maca_sinkhorn_iterations: int = 100

    # Vector Memory (CocoIndex)
    vector_db_path: str = "data/vector_memory.db"
    vector_embedding_dim: int = 2560  # Qwen2.5-3B

    # Post-Silicon
    post_silicon_interval_ms: float = 8.0

    # Latence cible
    target_latency_ms: float = 12.0  # 87 Hz
    target_jitter_ms: float = 0.2


class TTDistillIntegration:
    """Intégration end-to-end TT-Distill.

    Cette classe orchestre l'ensemble du pipeline:
    1. MACA consensus pour générer le barycentre tensoriel
    2. TTDistillBridge pour distiller le barycentre dans DoRA
    3. Reflex Engine pour exécuter le réflexe S1
    4. Vector Memory pour le RAG sub-millisecond
    5. Post-Silicon pour l'optimisation hardware
    """

    def __init__(self, config: TTDistillConfig | None = None):
        """Initialiser l'intégration TT-Distill.

        Args:
            config: Configuration TT-Distill
        """
        self.config = config or TTDistillConfig()

        # Initialiser les composants
        self.maca_engine = MACAEngine(
            latent_dim=self.config.vector_embedding_dim,
            seq_len=64,
            sinkhorn_epsilon=1e-3,
        )

        self.maca_bridge = MACASalonBridge()

        # Check if lora_path exists before passing to ReflexEngine
        lora_path_value = None
        if self.config.lora_path and Path(self.config.lora_path).exists():
            lora_path_value = self.config.lora_path

        self.reflex_engine = MultimodalReflexEngine(
            model_path=self.config.model_path,
            lora_path=lora_path_value,
        )

        self.vector_memory = VectorMemory(
            VectorMemory.Config(
                db_path=self.config.vector_db_path,
                embedding_dim=self.config.vector_embedding_dim,
            )
        )

        self.post_silicon = PostSiliconController()

        # MoA Gater with Metal O(1) swap (graceful degradation if dylib missing)
        self.moa_gater = MoAGater(enable_metal=True)

        # État de l'intégration
        self._running = False
        self._consensus_history: list[ConsensusResult] = []
        self._reflex_metrics: list[dict[str, Any]] = []
        self._optimization_task: asyncio.Task[None] | None = None
        self._pending_swap = False  # True when a DoRA preload is staged

        metal_status = "Metal O(1)" if self.moa_gater.metal_available else "numpy-only"
        logger.info("✅ TT-Distill Integration initialisée (adapter swap: %s)", metal_status)

    async def initialize(self) -> None:
        """Initialiser l'intégration (chargement des modèles, etc.)."""
        logger.info("🚀 Initialisation TT-Distill...")

        # Charger le modèle Vector Memory (initialisation automatique dans __init__)
        pass

        # Démarrer la boucle Post-Silicon
        self._running = True
        self._optimization_task = asyncio.create_task(
            self.post_silicon.run_optimization_loop(self.config.post_silicon_interval_ms)
        )

        logger.info("✅ TT-Distill prêt")

    async def run_consensus(self, intents: list[str]) -> ConsensusResult:
        """Exécuter le consensus MACA sur une liste d'intentions.

        Args:
            intents: Liste d'intentions des agents S2

        Returns:
            ConsensusResult: Résultat du consensus tensoriel
        """
        start_time = time.perf_counter()

        # Générer les trajectoires latentes pour chaque agent
        trajectories = []
        for i, intent in enumerate(intents):
            # Simuler la génération de trajectoire (à remplacer par vrai LLM S2)
            trajectory = self._generate_latent_trajectory(intent)
            trajectories.append({"agent_id": f"agent_{i}", "initial_state": trajectory})

        # Exécuter le consensus MACA
        consensus_result = await self.maca_engine.run_consensus(trajectories)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        self._consensus_history.append(consensus_result)

        logger.info(f"✅ Consensus atteint en {latency_ms:.2f} ms")

        return consensus_result

    def distill_to_dora(self, consensus: ConsensusResult) -> np.ndarray:
        """Distiller le barycentre MACA dans un adaptateur DoRA.

        After computing the DoRA adapter via SVD, stages it in the Metal
        backend's preload buffer if available. The actual O(1) swap fires
        at the next ``execute_reflex()`` call.

        Args:
            consensus: Résultat du consensus MACA

        Returns:
            np.ndarray: Matrice de l'adaptateur DoRA (lora_a @ lora_b)
        """
        # Récupérer le barycentre du consensus
        barycentre = consensus.barycentre  # Shape: [hidden_size]

        # Normaliser le barycentre
        barycentre_norm = barycentre / np.linalg.norm(barycentre)

        # SVD factorization pour DoRA
        u, s, vt = np.linalg.svd(barycentre_norm.reshape(1, -1), full_matrices=False)

        # Appliquer le rank et l'échelle
        rank = self.config.maca_rank
        scale = 1.0

        lora_a = (u[:, :rank] * s[:rank]) * scale
        lora_b = vt[:rank, :]

        # Combiner en un seul adaptateur
        dora_adapter = lora_a @ lora_b

        logger.info(f"✅ DoRA adapter généré: shape={dora_adapter.shape}, rank={rank}")

        # Stage the fused adapter into the Metal preload buffer
        if self.moa_gater.metal_available:
            adapter_dict = {"lora_a": lora_a.astype(np.float32), "lora_b": lora_b.astype(np.float32)}
            preload_ms = self.moa_gater.preload_to_metal(adapter_dict)
            self._pending_swap = True
            logger.info(f"⚡ DoRA adapter staged for Metal O(1) swap (preload: {preload_ms:.4f} ms)")

        return dora_adapter  # type: ignore[no-any-return]

    async def execute_reflex(
        self,
        prompt: str,
        use_rag: bool = True,
        context: str | None = None,
    ) -> tuple[float, str]:
        """Exécuter un réflexe S1 avec injection de contexte RAG.

        Args:
            prompt: Prompt de base
            use_rag: Utiliser le RAG Vector Memory
            context: Contexte optionnel (si None, recherche dans Vector Memory)

        Returns:
            tuple: (latency_ms, response_text)
        """
        start_time = time.perf_counter()

        # Fire the O(1) Metal swap if a preloaded adapter is staged
        if self._pending_swap and self.moa_gater.metal_available:
            swap_ms = self.moa_gater.swap_active_adapter()
            self._pending_swap = False
            logger.info(f"⚡ Metal O(1) swap fired before inference ({swap_ms:.6f} ms)")

        # Injecter le contexte RAG si demandé
        if use_rag and context is None:
            # Rechercher dans Vector Memory (sync API)
            results = self.vector_memory.search(prompt, top_k=3)
            if results:
                context = "\n".join([r[0]["text"] for r in results])
                prompt = f"Contexte: {context}\n\n{prompt}"

        # Exécuter le réflexe
        latency_ms, response_text = self.reflex_engine.query_reflex(prompt)

        # Enregistrer les métriques
        self._reflex_metrics.append({
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "response_length": len(response_text),
            "use_rag": use_rag,
        })

        # Appliquer les ajustements Post-Silicon
        await self._apply_post_silicon_adjustments()

        end_time = time.perf_counter()
        total_latency_ms = (end_time - start_time) * 1000

        return total_latency_ms, response_text

    async def execute_multimodal_reflex(
        self,
        prompt: str,
        image_path: str | None = None,
        video_path: str | None = None,
    ) -> tuple[float, str]:
        """Exécuter un réflexe multimodal.

        Args:
            prompt: Prompt de texte
            image_path: Chemin vers l'image (optionnel)
            video_path: Chemin vers la vidéo (optionnel)

        Returns:
            tuple: (latency_ms, response_text)
        """
        time.perf_counter()

        if image_path and Path(image_path).exists():  # noqa: ASYNC240
            latency_ms, response_text = self.reflex_engine.query_with_image(prompt, image_path)
        elif video_path and Path(video_path).exists():  # noqa: ASYNC240
            latency_ms, response_text = self.reflex_engine.query_with_video(prompt, video_path)
        else:
            latency_ms, response_text = self.reflex_engine.query_reflex(prompt)

        # Enregistrer les métriques
        self._reflex_metrics.append({
            "timestamp": time.time(),
            "latency_ms": latency_ms,
            "response_length": len(response_text),
            "modalities": ["image"] if image_path else ["video"] if video_path else ["text"],
        })

        return latency_ms, response_text

    async def add_document(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        """Ajouter un document au Vector Memory (CocoIndex).

        Args:
            text: Texte à indexer
            metadata: Métadonnées optionnelles
        """
        # Utiliser l'API sync (add_document au lieu de add_document_async)
        doc_id = int(time.time() * 1000)
        self.vector_memory.add_document(doc_id, text, metadata or {})
        logger.info(f"✅ Document ajouté au Vector Memory: {text[:50]}...")

    def get_reflex_metrics(self) -> dict[str, Any]:
        """Récupérer les métriques du réflexe.

        Returns:
            dict: Métriques statistiques
        """
        if not self._reflex_metrics:
            return {"count": 0}

        latencies = [m["latency_ms"] for m in self._reflex_metrics]

        return {
            "count": len(self._reflex_metrics),
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "frequency_hz": 1000 / np.mean(latencies),
            "target_latency_ms": self.config.target_latency_ms,
            "target_frequency_hz": 1000 / self.config.target_latency_ms,
        }

    def get_consensus_history(self) -> list[ConsensusResult]:
        """Récupérer l'historique des consensus.

        Returns:
            list[ConsensusResult]: Historique des consensus
        """
        return self._consensus_history

    async def _apply_post_silicon_adjustments(self) -> None:
        """Appliquer les ajustements Post-Silicon basés sur les métriques."""
        metrics = self.get_reflex_metrics()

        if metrics["count"] < MIN_METRICS_FOR_STATS:
            return  # Attendre suffisamment de métriques

        # Proposer un ajustement si nécessaire
        if metrics["mean_latency_ms"] > self.config.target_latency_ms:
            # Latence trop élevée → augmenter la fréquence du NPU
            adjustment = self.post_silicon.propose_adjustment(
                reason="Latence élevée",
                estimated_impact="latency",
                confidence=0.8,
            )

            if adjustment:
                await self.post_silicon.apply_adjustment(adjustment)

        # Mettre à jour l'état hardware
        await self.post_silicon.update_hardware_state()

    def _generate_latent_trajectory(self, intent: str) -> np.ndarray:
        """Générer une trajectoire latente pour un agent S2.

        Args:
            intent: Intent de l'agent

        Returns:
            np.ndarray: Trajectoire latente (shape: [seq_len, hidden_size])
        """
        # Simulation: générer une trajectoire aléatoire
        # Dans une implémentation réelle, utiliser un LLM S2 pour générer
        hidden_size = self.config.vector_embedding_dim
        seq_len = 64

        # Générer une embedding basée sur le hash de l'intent
        intent_hash = hash(intent) & 0xFFFFFFFF
        np.random.seed(intent_hash)

        return np.random.randn(seq_len, hidden_size).astype(np.float32)


    async def shutdown(self) -> None:
        """Arrêter l'intégration et libérer les ressources."""
        self._running = False

        # Arrêter la boucle Post-Silicon
        self.post_silicon._running = False

        # Libérer les ressources
        self.reflex_engine.close()  # type: ignore[no-untyped-call]
        await self.vector_memory.close()

        logger.info("🛑 TT-Distill Integration arrêtée")


async def main() -> None:
    """Point d'entrée principal pour tester l'intégration end-to-end."""
    config = TTDistillConfig(
        model_path="Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
        lora_path="test_adapter.bin",
        maca_n_agents=4,
        maca_rank=16,
    )

    integration = TTDistillIntegration(config)

    try:
        # Initialiser
        await integration.initialize()

        # Ajouter un document au Vector Memory
        await integration.add_document(
            "Le système TT-Distill sépare le Resolver (S2) du Produit (S1).",
            {"category": "architecture", "tags": ["TT-Distill", "S2", "S1"]},
        )

        # Exécuter un consensus MACA
        intents = [
            "Optimiser la latence du réflexe",
            "Améliorer la précision du RAG",
            "Réduire le jitter",
            "Maximiser le throughput",
        ]
        consensus = await integration.run_consensus(intents)

        # Distiller le barycentre dans DoRA
        integration.distill_to_dora(consensus)

        # Exécuter un réflexe avec RAG
        _latency, _response = await integration.execute_reflex(
            "Quel est le modèle S1 utilisé?",
            use_rag=True,
        )

        # Afficher les métriques
        integration.get_reflex_metrics()

    finally:
        await integration.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
