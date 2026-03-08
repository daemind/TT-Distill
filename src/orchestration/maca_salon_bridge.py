"""MACA Salon Bridge - Intégration MACA avec BrainstormingSalon existant.

Ce module crée une couche de compatibilité entre le brainstorming_salon.py
(textuel) et le MACA engine (tensoriel), permettant une transition progressive.

Architecture:
    BrainstormingSalon (S2 textuel) → MACASalonBridge → MACAEngine (S2 tensoriel)
                                        ↓
                                 Consensus tensoriel
                                        ↓
                                 DoRA adapter (15 Mo)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.logger import get_logger
from src.orchestration.brainstorming_salon import BrainstormingSalon
from src.orchestration.maca import (
    ConsensusResult,
    LatentTrajectory,
    MACAEngine,
    TTDistillBridge,
)
from src.orchestration.strategy import (
    BrainstormingMessage,
    BrainstormingMode,
)

logger = get_logger(__name__)


@dataclass
class MACASession:
    """Session hybride combinant salon textuel et délibération tensorielle."""

    session_id: str
    title: str
    description: str
    status: str = "active"
    resolution: str | None = None
    messages: list[BrainstormingMessage] = field(default_factory=list)
    latent_trajectories: list[LatentTrajectory] = field(default_factory=list)
    consensus_result: ConsensusResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MACASalonBridge:
    """Pont entre BrainstormingSalon (textuel) et MACAEngine (tensoriel).

    Ce pont permet d'utiliser le brainstorming textuel existant comme interface
    utilisateur, tout en exécutant la délibération réelle via MACA.

    Workflow:
        1. Les agents envoient des messages textuels via BrainstormingSalon
        2. Le pont extrait les intentions sémantiques des messages
        3. Les intentions sont converties en hidden states (via LLM embedding)
        4. MACAEngine exécute le consensus tensoriel
        5. Le barycentre est distillé en DoRA adapter
        6. Le résultat est enregistré dans la session
    """

    def __init__(
        self,
        salon: BrainstormingSalon | None = None,
        maca_engine: MACAEngine | None = None,
        latent_dim: int = 4096,
        seq_len: int = 32,
    ) -> None:
        """Initialiser le pont MACA Salon.

        Args:
            salon: Instance de BrainstormingSalon (créée par défaut)
            maca_engine: Instance de MACAEngine (créée par défaut)
            latent_dim: Dimension de l'espace latent
            seq_len: Longueur de séquence pour le rollout
        """
        self.salon = salon or BrainstormingSalon()
        self.maca_engine = maca_engine or MACAEngine(
            latent_dim=latent_dim, seq_len=seq_len
        )

        # Cache des sessions hybrides
        self._hybrid_sessions: dict[str, MACASession] = {}

        # Cache des embeddings pour conversion texte → latent
        self._text_embedding_cache: dict[str, np.ndarray] = {}

        logger.info(
            f"🌉 MACASalonBridge initialisé (latent_dim={latent_dim}, seq_len={seq_len})"
        )

    async def create_hybrid_session(
        self,
        title: str,
        description: str,
        participants: list[str],
    ) -> MACASession:
        """Créer une session hybride (texte + tensoriel).

        Args:
            title: Titre de la session
            description: Description de la session
            participants: Liste des participants

        Returns:
            MACASession hybride
        """
        # Créer la session textuelle via BrainstormingSalon
        text_session = await self.salon.create_session(title, description, participants)

        # Créer la session hybride
        hybrid_session = MACASession(
            session_id=text_session.session_id,
            title=title,
            description=description,
            status="active",
        )

        self._hybrid_sessions[hybrid_session.session_id] = hybrid_session

        logger.info(
            f"✅ Session hybride créée: {title} (id={hybrid_session.session_id})"
        )

        return hybrid_session

    async def add_text_message(
        self,
        session_id: str,
        agent_name: str,
        mode: BrainstormingMode,
        content: str,
    ) -> BrainstormingMessage:
        """Ajouter un message textuel à la session hybride.

        Args:
            session_id: ID de la session
            agent_name: Nom de l'agent
            mode: Mode du message
            content: Contenu textuel

        Returns:
            BrainstormingMessage créé
        """
        # Ajouter le message au salon textuel
        message = await self.salon.add_message(session_id, agent_name, mode, content)

        # Ajouter à la session hybride
        if session_id in self._hybrid_sessions:
            hybrid_session = self._hybrid_sessions[session_id]
            hybrid_session.messages.append(message)

            # Convertir le texte en hidden state (embedding)
            latent_state = await self._text_to_latent(content, agent_name)

            # Générer une trajectoire latente
            trajectory = self.maca_engine.rollout.generate_trajectory(
                initial_state=latent_state,
                agent_id=agent_name,
                confidence=0.9 if mode == BrainstormingMode.RESOLUTION else 0.7,
            )

            hybrid_session.latent_trajectories.append(trajectory)

            logger.debug(
                f"📝 Message textuel converti en trajectoire latente: {agent_name}"
            )

        return message

    async def _text_to_latent(self, text: str, agent_name: str) -> np.ndarray:
        """Convertir un texte en hidden state latent.

        Cette méthode utilise un embedding LLM pour transformer le texte
        en vecteur latent de dimension LATENT_DIM.

        Args:
            text: Texte à convertir
            agent_name: Nom de l'agent (pour le cache)

        Returns:
            Hidden state latent (shape: (latent_dim,))
        """
        # Vérifier le cache
        if text in self._text_embedding_cache:
            return self._text_embedding_cache[text]

        # Simuler un embedding (dans une implémentation réelle, utiliser un LLM)
        # Hashing déterministe du texte pour obtenir un vecteur reproductible
        np.random.seed(hash(text) % (2**32))
        latent_state = np.random.randn(self.maca_engine.latent_dim) * 0.5

        # Ajouter une signature basée sur l'agent
        agent_hash = hash(agent_name) % (2**32)
        latent_state += (
            np.random.randn(self.maca_engine.latent_dim)
            * 0.1
            * (agent_hash % 100)
            / 100
        )

        # Mettre en cache
        self._text_embedding_cache[text] = latent_state

        return latent_state

    async def run_tensorial_consensus(
        self,
        session_id: str,
        consensus_threshold: float = 0.9,
    ) -> ConsensusResult | None:
        """Exécuter le consensus tensoriel sur les trajectoires de la session.

        Args:
            session_id: ID de la session
            consensus_threshold: Seuil de consensus pour validation

        Returns:
            ConsensusResult si succès, None sinon
        """
        if session_id not in self._hybrid_sessions:
            logger.error(f"❌ Session {session_id} non trouvée")
            return None

        hybrid_session = self._hybrid_sessions[session_id]

        if not hybrid_session.latent_trajectories:
            logger.warning(f"⚠️ Aucune trajectoire latente pour la session {session_id}")
            return None

        # Préparer les intentions pour MACA
        agent_intentions = []
        agent_intentions = [
            {
                "agent_id": trajectory.agent_id,
                "initial_state": trajectory.hidden_states[0],
                "confidence": trajectory.confidence,
            }
            for trajectory in hybrid_session.latent_trajectories
        ]

        # Exécuter le consensus MACA
        consensus = await self.maca_engine.run_consensus(agent_intentions)

        # Mettre à jour la session hybride
        hybrid_session.consensus_result = consensus
        hybrid_session.status = (
            "resolved" if consensus.metadata["is_valid"] else "pending"
        )

        # Générer l'adaptateur DoRA
        bridge = TTDistillBridge(adapter_size_mb=15, rank=32, hidden_size=2048)
        adapter_tensors = bridge.distill_barycenter(consensus.barycentre)

        # Sauvegarder l'adaptateur
        output_path = f"generated/adapters/session_{session_id}_consensus.bin"
        bridge.save_adapter(adapter_tensors, output_path)

        hybrid_session.metadata["adapter_path"] = output_path
        hybrid_session.metadata["consensus_threshold"] = consensus_threshold

        logger.info(
            f"🎯 Consensus tensoriel atteint: score={consensus.consensus_score:.4f}, "
            f"valid={consensus.metadata['is_valid']}, adapter={output_path}"
        )

        return consensus

    def get_hybrid_session(self, session_id: str) -> MACASession | None:
        """Récupérer une session hybride.

        Args:
            session_id: ID de la session

        Returns:
            MACASession si trouvé, None sinon
        """
        return self._hybrid_sessions.get(session_id)

    def list_hybrid_sessions(self) -> list[MACASession]:
        """Lister toutes les sessions hybrides."""
        return list(self._hybrid_sessions.values())

    def clear_cache(self) -> None:
        """Vider les caches."""
        self._text_embedding_cache.clear()
        self.maca_engine.clear_cache()
        logger.debug("🗑️ Caches vidés")


async def demo_maca_salon_bridge() -> None:
    """Démonstration du pont MACA Salon."""

    # Initialiser le pont
    bridge = MACASalonBridge(latent_dim=64, seq_len=16)

    # Créer une session hybride
    session = await bridge.create_hybrid_session(
        title="Architecture TT-Distill",
        description="Délibération sur l'architecture TT-Distill",
        participants=["architect", "engineer", "reviewer"],
    )

    # Ajouter des messages textuels
    await bridge.add_text_message(
        session.session_id,
        "architect",
        BrainstormingMode.IDEA,
        "Le Resolver doit transformer les intentions en planning d'exécution",
    )

    await bridge.add_text_message(
        session.session_id,
        "engineer",
        BrainstormingMode.IDEA,
        "Le Produit doit exécuter à 125 Hz avec inputs_embeds",
    )

    await bridge.add_text_message(
        session.session_id,
        "reviewer",
        BrainstormingMode.CRITIQUE,
        "Attention à la latence: objectif 8 ms constants",
    )

    await bridge.add_text_message(
        session.session_id,
        "architect",
        BrainstormingMode.RESOLUTION,
        "Consensus: utiliser MACA pour la délibération tensorielle",
    )

    # Exécuter le consensus tensoriel
    consensus = await bridge.run_tensorial_consensus(session.session_id)

    if consensus:
        pass


if __name__ == "__main__":
    asyncio.run(demo_maca_salon_bridge())
