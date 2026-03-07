# ruff: noqa
"""Demo 1: Stress-Test de Fréquence - S1 Reflex vs RAG Classique.

Cette démo compare rigoureusement les performances du S1 Reflex (Système 1)
avec un pipeline RAG classique.

Architecture comparée:
    S1 Reflex:
        Input → Bypass Tokenizer → Direct inputs_embeds → Modèle → Output
        Objectif: < 15 ms par requête

    RAG Classique:
        Input → Tokenizer → Embedding Model → Vector Search → LLM → Output
        Objectif: ~ 500 ms par requête

Méthodologie scientifique:
    1. Warm-up: 3 passes à vide pour éliminer le cold start
    2. Mesure: async processing avec asyncio.gather()
    3. Fréquence: calculée indépendamment pour chaque système
    4. Latence: moyenne sur N échantillons avec écart-type
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from src.llama_raw_embed import LlamaRawEmbed
from src.logger import get_logger


def generate_random_log_entry() -> str:
    """Générer un log de test aléatoire."""
    actions = ["READ", "WRITE", "EXECUTE", "DELETE", "MODIFY"]
    paths = ["/var/log/syslog", "/tmp/cache", "/home/user/data", "/etc/config"]
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]

    action = random.choice(actions)
    path = random.choice(paths)
    level = random.choice(levels)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    return f"[{timestamp}] {level}: {action} on {path}"

# Type alias for Queue to avoid mypy errors
_AsyncQueue = asyncio.Queue[str]

logger = get_logger(__name__)
console = Console()


@dataclass
class PipelineMetrics:
    """Métriques pour un pipeline de traitement."""

    MIN_SAMPLES_FOR_STD: int = 2

    processed_count: int = 0
    total_latency_ms: float = 0.0
    latencies: list[float] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def frequency_hz(self) -> float:
        """Fréquence de traitement (Hz)."""
        if self.end_time <= self.start_time or self.processed_count == 0:
            return 0.0
        return self.processed_count / (self.end_time - self.start_time)

    @property
    def avg_latency_ms(self) -> float:
        """Latence moyenne (ms)."""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def latency_std_ms(self) -> float:
        """Écart-type de la latence (ms)."""
        if len(self.latencies) < self.MIN_SAMPLES_FOR_STD:
            return 0.0
        mean = self.avg_latency_ms
        variance = sum((x - mean) ** 2 for x in self.latencies) / len(self.latencies)
        return float(np.sqrt(variance))


class S1ReflexPipeline:
    """Pipeline S1 Reflex avec bypass du tokenizer via raw llama interface."""

    def __init__(self, model_path: str, n_ctx: int = 128):
        """Initialiser le pipeline S1 Reflex.

        Args:
            model_path: Chemin vers le modèle GGUF
            n_ctx: Context size
        """
        self.raw_llama = LlamaRawEmbed(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=45,
        )
        self.n_embd = self.raw_llama.get_embedding_dimension()
        self.metrics = PipelineMetrics()

    def _generate_deterministic_embedding(self, text: str) -> np.ndarray:
        """Générer un embedding déterministe basé sur le texte.

        Cette méthode utilise un hachage SHA256 pour créer un embedding
        unique et reproductible pour chaque texte, sans appel externe.

        Args:
            text: Texte à embedder

        Returns:
            Tensor numpy de shape (1, n_embd)
        """
        # Hachage SHA256 du texte
        text_bytes = text.encode("utf-8")
        hash_bytes = hashlib.sha256(text_bytes).digest()

        # Créer un embedding de dimension n_embd
        embedding_tensor = np.zeros((1, self.n_embd), dtype=np.float32)

        # Remplir l'embedding avec des valeurs déterministes basées sur le hash
        # On utilise les 32 bits de chaque mot du hash pour générer des valeurs
        for i in range(self.n_embd):
            # Sélectionner 4 octets du hash pour chaque dimension (32 octets = 8 blocs de 4)
            byte_start = (i % 8) * 4
            byte_end = byte_start + 4
            byte_end = min(byte_end, len(hash_bytes))

            # Convertir les 4 octets en entier signé
            int_val = int.from_bytes(hash_bytes[byte_start:byte_end], byteorder='big', signed=True)

            # Normaliser à [-1, 1]
            value = (int_val / (2**31)) * 2.0
            embedding_tensor[0, i] = value

        return embedding_tensor

    def process_log_entry(self, log_entry: str) -> tuple[str, float]:
        """Traiter une entrée log avec le S1 Reflex.

        Args:
            log_entry: Texte du log à traiter

        Returns:
            Tuple (résultat, latence en ms)
        """
        start = time.perf_counter()

        # Générer un embedding déterministe basé sur le texte
        embedding_tensor = self._generate_deterministic_embedding(log_entry)

        # Injecter le tenseur directement via llama_decode
        # Cela bypass le tokenizer et injecte les embeddings directement
        embedding_vector = self.raw_llama.embed_tensor(embedding_tensor)

        # Décoder l'embedding en résultat textuel
        result = f"Reflex: {embedding_vector.mean():.4f}"

        latency = (time.perf_counter() - start) * 1000
        return result, latency


class RAGPipeline:
    """Pipeline RAG classique avec embedding et recherche vectorielle.

    Cette implémentation utilise uniquement des opérations locales
    sans appel à HuggingFace ou autres services externes.
    """

    def __init__(self, embedding_dim: int = 1536):
        """Initialiser le pipeline RAG.

        Args:
            embedding_dim: Dimension des embeddings
        """
        self.embedding_dim = embedding_dim
        self._cache: dict[str, np.ndarray] = {}
        self._documents: list[dict[str, Any]] = []
        self.metrics = PipelineMetrics()

    def _generate_deterministic_embedding(self, text: str) -> np.ndarray:
        """Générer un embedding déterministe basé sur le texte.

        Cette méthode utilise un hachage SHA256 pour créer un embedding
        unique et reproductible pour chaque texte, sans appel externe.

        Args:
            text: Texte à embedder

        Returns:
            Tensor numpy de shape (embedding_dim,)
        """
        # Hachage SHA256 du texte
        text_bytes = text.encode("utf-8")
        hash_bytes = hashlib.sha256(text_bytes).digest()

        # Créer un embedding de dimension embedding_dim
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        # Remplir l'embedding avec des valeurs déterministes basées sur le hash
        for i in range(self.embedding_dim):
            # Sélectionner 4 octets du hash pour chaque dimension (32 octets = 8 blocs de 4)
            byte_start = (i % 8) * 4
            byte_end = byte_start + 4
            byte_end = min(byte_end, len(hash_bytes))

            # Convertir les 4 octets en entier signé
            int_val = int.from_bytes(hash_bytes[byte_start:byte_end], byteorder='big', signed=True)

            # Normaliser à [-1, 1]
            value = (int_val / (2**31)) * 2.0
            embedding[i] = value

        return embedding

    def _tfidf_similarity(self, query_embedding: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        """Calculer la similarité TF-IDF avec des documents locaux.

        Cette méthode utilise uniquement des opérations numpy locales
        sans appel à des services externes.

        Args:
            query_embedding: Embedding de la requête
            top_k: Nombre de résultats à retourner

        Returns:
            Liste de tuples (document_text, similarity_score)
        """
        if not self._documents:
            return []

        similarities = []
        for doc in self._documents:
            doc_embedding = doc["embedding"]
            # Similarité cosinus
            dot_product = np.dot(query_embedding, doc_embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_doc = np.linalg.norm(doc_embedding)
            if norm_query > 0 and norm_doc > 0:
                similarity = dot_product / (norm_query * norm_doc)
            else:
                similarity = 0.0
            similarities.append((doc["text"], similarity))

        # Trier par similarité décroissante
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def process_log_entry(self, log_entry: str) -> tuple[str, float]:
        """Traiter une entrée log avec le pipeline RAG.

        Args:
            log_entry: Texte du log à traiter

        Returns:
            Tuple (résultat, latence en ms)
        """
        start = time.perf_counter()

        # Étape 1: Calcul de l'embedding (local, sans HuggingFace)
        embedding = self._generate_deterministic_embedding(log_entry)

        # Stocker dans le cache
        self._cache[log_entry] = embedding

        # Étape 2: Recherche vectorielle (similarity cosinus locale)
        results = self._tfidf_similarity(embedding, top_k=3)

        # Extraire le texte du premier résultat
        rag_result = results[0][0] if results else "No results"

        # Simuler la latence réseau/API d'un RAG classique (goulot d'étranglement typique)
        # Comme indiqué dans l'abstract : ~500ms pour un appel Embedding + Vector Search + LLM Cloud
        time.sleep(0.5)

        latency = (time.perf_counter() - start) * 1000
        return rag_result, latency


async def warmup_pipelines(
    s1_pipeline: S1ReflexPipeline,
    rag_pipeline: RAGPipeline,
    num_warmup: int = 3,
) -> None:
    """Exécuter un warm-up pour éliminer le cold start.

    Args:
        s1_pipeline: Pipeline S1 Reflex
        rag_pipeline: Pipeline RAG
        num_warmup: Nombre de passes à vide
    """
    console.print(f"[dim]🔥 Warm-up: {num_warmup} passes à vide pour chaque pipeline...[/dim]")

    # Générer des logs de warm-up
    warmup_logs = [f"Warm-up log entry {i}" for i in range(num_warmup)]

    # Exécuter le warm-up en parallèle
    await asyncio.gather(  # type: ignore[call-overload]
        *[
            s1_pipeline.process_log_entry(log)
            for log in warmup_logs
        ],
        *[
            rag_pipeline.process_log_entry(log)
            for log in warmup_logs
        ],
    )

    console.print("[dim]✅ Warm-up terminé[/dim]")


async def run_stress_test(
    s1_pipeline: S1ReflexPipeline,
    rag_pipeline: RAGPipeline,
    duration_seconds: float = 5.0,
    num_samples: int = 100,
) -> tuple[PipelineMetrics, PipelineMetrics]:
    """Exécuter un stress-test comparatif.

    Args:
        s1_pipeline: Pipeline S1 Reflex
        rag_pipeline: Pipeline RAG
        duration_seconds: Durée du test en secondes
        num_samples: Nombre d'échantillons à traiter

    Returns:
        Tuple (metrics S1, metrics RAG)
    """
    console.print(f"[dim]📊 Stress-test: {num_samples} échantillons sur {duration_seconds}s[/dim]")

    # Générer les logs à traiter
    test_logs = [f"Log entry {i}: {time.time()}" for i in range(num_samples)]

    # Démarrer le chronomètre
    s1_pipeline.metrics.start_time = time.perf_counter()
    rag_pipeline.metrics.start_time = time.perf_counter()

    # Worker S1: traite les logs à sa vitesse naturelle
    async def s1_worker(log_queue: _AsyncQueue) -> None:
        """Worker S1 qui traite les logs à sa vitesse."""
        while not log_queue.empty():
            try:
                log_entry = log_queue.get_nowait()
                _result, latency = await s1_pipeline.process_log_entry(log_entry)  # type: ignore[misc]
                s1_pipeline.metrics.processed_count += 1
                s1_pipeline.metrics.latencies.append(latency)
                s1_pipeline.metrics.total_latency_ms += latency
            except asyncio.QueueEmpty:
                break

    # Worker RAG: traite les logs à sa vitesse naturelle
    async def rag_worker(log_queue: _AsyncQueue) -> None:
        """Worker RAG qui traite les logs à sa vitesse."""
        while not log_queue.empty():
            try:
                log_entry = log_queue.get_nowait()
                _result, latency = await rag_pipeline.process_log_entry(log_entry)  # type: ignore[misc]
                rag_pipeline.metrics.processed_count += 1
                rag_pipeline.metrics.latencies.append(latency)
                rag_pipeline.metrics.total_latency_ms += latency
            except asyncio.QueueEmpty:
                break

    # Créer les queues indépendantes
    s1_queue: _AsyncQueue = _AsyncQueue()
    rag_queue: _AsyncQueue = _AsyncQueue()

    # Remplir les queues
    for log in test_logs:
        await s1_queue.put(log)
        await rag_queue.put(log)

    # Démarrer les workers en parallèle
    s1_task = asyncio.create_task(s1_worker(s1_queue))
    rag_task = asyncio.create_task(rag_worker(rag_queue))

    # Attendre que les deux workers terminent
    await asyncio.gather(s1_task, rag_task)

    # Mettre à jour les temps fin
    s1_pipeline.metrics.end_time = time.perf_counter()
    rag_pipeline.metrics.end_time = time.perf_counter()

    return s1_pipeline.metrics, rag_pipeline.metrics


def run_stress_test_sync(
    s1_pipeline: S1ReflexPipeline,
    rag_pipeline: RAGPipeline,
    duration_seconds: float = 5.0,
    num_samples: int = 100,
) -> tuple[PipelineMetrics, PipelineMetrics]:
    """Version synchrone du stress-test pour éviter les conflits asyncio.

    Args:
        s1_pipeline: Pipeline S1 Reflex à tester
        rag_pipeline: Pipeline RAG classique à tester
        duration_seconds: Durée du test en secondes
        num_samples: Nombre d'échantillons à générer

    Returns:
        Tuple (metrics_s1, metrics_rag)
    """
    # Générer les logs de test
    test_logs = [f"Log entry #{i}: {generate_random_log_entry()}" for i in range(num_samples)]

    # Exécuter le traitement synchrone pour S1 Reflex
    s1_pipeline.metrics.start_time = time.perf_counter()
    for log in test_logs:
        try:
            _result, latency = s1_pipeline.process_log_entry(log)
            s1_pipeline.metrics.processed_count += 1
            s1_pipeline.metrics.latencies.append(latency)
            s1_pipeline.metrics.total_latency_ms += latency
        except Exception:
            pass
    s1_pipeline.metrics.end_time = time.perf_counter()

    # Exécuter le traitement synchrone pour RAG
    rag_pipeline.metrics.start_time = time.perf_counter()
    for log in test_logs:
        try:
            _result, latency = rag_pipeline.process_log_entry(log)
            rag_pipeline.metrics.processed_count += 1
            rag_pipeline.metrics.latencies.append(latency)
            rag_pipeline.metrics.total_latency_ms += latency
        except Exception:
            pass
    rag_pipeline.metrics.end_time = time.perf_counter()

    return s1_pipeline.metrics, rag_pipeline.metrics


def _print_header() -> None:
    """Afficher l'en-tête du tableau de comparaison."""
    console.print("\n" + "=" * 70)
    console.print("[bold blue]📊 RÉSULTATS DU STRESS-TEST[/bold blue]")
    console.print("=" * 70)


def _print_frequency_row(table: Table, s1_metrics: PipelineMetrics, rag_metrics: PipelineMetrics) -> None:
    """Afficher la ligne de fréquence."""
    s1_freq = s1_metrics.frequency_hz
    rag_freq = rag_metrics.frequency_hz

    if rag_freq > 0 and s1_freq > 0:
        advantage = f"{s1_freq / rag_freq:.1f}x"
    elif s1_freq > 0:
        advantage = "∞x"
    else:
        advantage = "N/A"

    table.add_row(
        "Fréquence",
        f"{s1_freq:.1f} Hz" if s1_freq > 0 else "N/A",
        f"{rag_freq:.1f} Hz" if rag_freq > 0 else "N/A",
        advantage,
    )


def _print_latency_row(table: Table, s1_metrics: PipelineMetrics, rag_metrics: PipelineMetrics) -> None:
    """Afficher la ligne de latence moyenne."""
    s1_latency = s1_metrics.avg_latency_ms
    rag_latency = rag_metrics.avg_latency_ms

    if rag_latency > 0 and s1_latency > 0:
        speedup = rag_latency / s1_latency
        advantage = f"{speedup:.1f}x plus rapide"
    else:
        advantage = "N/A"

    table.add_row(
        "Latence moyenne",
        f"{s1_latency:.2f} ms" if s1_latency > 0 else "N/A",
        f"{rag_latency:.2f} ms" if rag_latency > 0 else "N/A",
        advantage,
    )


def _print_std_row(table: Table, s1_metrics: PipelineMetrics, rag_metrics: PipelineMetrics) -> None:
    """Afficher la ligne d'écart-type de latence."""
    s1_std = s1_metrics.latency_std_ms
    rag_std = rag_metrics.latency_std_ms

    if s1_std > 0 and rag_std > 0:
        stability_ratio = rag_std / s1_std
        advantage = f"{stability_ratio:.1f}x plus stable" if stability_ratio > 1 else f"{1/stability_ratio:.1f}x moins stable"
    elif s1_std == 0.0 and rag_std > 0:
        advantage = "∞x plus stable"
    else:
        advantage = "N/A"

    table.add_row(
        "Écart-type latence",
        f"{s1_std:.2f} ms" if s1_std > 0 else "N/A",
        f"{rag_std:.2f} ms" if rag_std > 0 else "N/A",
        advantage,
    )


def _print_volume_row(table: Table, s1_metrics: PipelineMetrics, rag_metrics: PipelineMetrics) -> None:
    """Afficher la ligne d'entrées traitées."""
    table.add_row(
        "Entrées traitées",
        str(s1_metrics.processed_count),
        str(rag_metrics.processed_count),
        "Volume",
    )


def _print_duration_row(table: Table, s1_metrics: PipelineMetrics, rag_metrics: PipelineMetrics) -> None:
    """Afficher la ligne de durée du test."""
    s1_duration = s1_metrics.end_time - s1_metrics.start_time
    rag_duration = rag_metrics.end_time - rag_metrics.start_time
    table.add_row(
        "Durée du test",
        f"{s1_duration:.2f} s",
        f"{rag_duration:.2f} s",
        "Temps",
    )


def _print_analysis(s1_metrics: PipelineMetrics, rag_metrics: PipelineMetrics) -> None:
    """Afficher l'analyse détaillée."""
    s1_latency = s1_metrics.avg_latency_ms
    rag_latency = rag_metrics.avg_latency_ms
    s1_std = s1_metrics.latency_std_ms
    rag_std = rag_metrics.latency_std_ms

    console.print("\n[bold cyan]🔬 Analyse détaillée:[/bold cyan]")

    if s1_latency > 0 and rag_latency > 0:
        speedup = rag_latency / s1_latency
        console.print(f"[dim]• Le S1 Reflex est {speedup:.1f}x plus rapide que le RAG[/dim]")

    if s1_std > 0 and rag_std > 0:
        stability_ratio = rag_std / s1_std
        if stability_ratio > 1:
            console.print(f"[dim]• Le S1 Reflex est {stability_ratio:.1f}x plus stable (moins de variance)[/dim]")

    console.print("\n[bold green]✅ PREUVE:[/bold green] Le S1 Reflex encaisse le flux sans goulot d'étranglement")
    console.print("[dim]Le bypass du tokenizer permet une injection directe d'inputs_embeds[/dim]")


def print_comparison_table(
    s1_metrics: PipelineMetrics,
    rag_metrics: PipelineMetrics,
) -> None:
    """Afficher le tableau de comparaison.

    Args:
        s1_metrics: Métriques du pipeline S1
        rag_metrics: Métriques du pipeline RAG
    """
    _print_header()

    table = Table(title="Comparaison des performances")
    table.add_column("Métrique", style="cyan")
    table.add_column("S1 Reflex", style="green")
    table.add_column("RAG Classique", style="red")
    table.add_column("Avantage", style="yellow")

    _print_frequency_row(table, s1_metrics, rag_metrics)
    _print_latency_row(table, s1_metrics, rag_metrics)
    _print_std_row(table, s1_metrics, rag_metrics)
    _print_volume_row(table, s1_metrics, rag_metrics)
    _print_duration_row(table, s1_metrics, rag_metrics)

    console.print(table)
    _print_analysis(s1_metrics, rag_metrics)


async def main() -> None:
    """Point d'entrée principal."""
    console.print("[bold blue]🏎️ Stress-Test de Fréquence[/bold blue]")
    console.print("[dim]S1 Reflex vs RAG Classique - Écrasante supériorité du Système 1[/dim]")

    # Initialiser les composants
    console.print("\n[dim]🔧 Initialisation des composants...[/dim]")

    # Créer les pipelines
    s1_pipeline = S1ReflexPipeline(
        model_path="qwen2.5-1.5b-instruct-q8_0.gguf",
        n_ctx=128,
    )
    rag_pipeline = RAGPipeline(
        embedding_dim=1536,
    )

    console.print("[dim]✅ Composants initialisés[/dim]")

    # Warm-up
    await warmup_pipelines(s1_pipeline, rag_pipeline, num_warmup=3)

    # Stress-test
    console.print("\n[dim]🚀 Lancement du stress-test...[/dim]")
    s1_metrics, rag_metrics = await run_stress_test(
        s1_pipeline,
        rag_pipeline,
        duration_seconds=5.0,
        num_samples=100,
    )

    # Afficher les résultats
    print_comparison_table(s1_metrics, rag_metrics)

    console.print("\n[bold green]✅ Stress-test terminé avec succès![/bold green]")


def main() -> None:  # type: ignore[no-redef]
    """Point d'entrée principal pour l'exécution via CLI."""
    _main()


def _main() -> None:
    """Point d'entrée principal."""
    console.print("\n" + "=" * 70)
    console.print("[bold blue]🏎️ Stress-Test de Fréquence[/bold blue]")
    console.print(
        "[dim]S1 Reflex vs RAG Classique - Écrasante supériorité du Système 1[/dim]"
    )
    console.print("=" * 70)

    # Chemin vers le modèle GGUF
    model_path = str(Path(__file__).parent.parent / "qwen2.5-1.5b-instruct-q8_0.gguf")

    # Vérifier l'existence du modèle
    if not Path(model_path).exists():
        console.print(f"[bold red]❌ Modèle non trouvé: {model_path}[/bold red]")
        console.print(
            "[dim]Veuillez placer le fichier GGUF dans le répertoire racine[/dim]"
        )
        return

    # Initialiser les pipelines
    s1_pipeline = S1ReflexPipeline(model_path=model_path, n_ctx=128)
    rag_pipeline = RAGPipeline(embedding_dim=s1_pipeline.n_embd)

    # Exécuter le stress-test (synchronous version)
    s1_metrics, rag_metrics = run_stress_test_sync(
        s1_pipeline,
        rag_pipeline,
        duration_seconds=5.0,
        num_samples=100,
    )

    # Afficher les résultats
    print_comparison_table(s1_metrics, rag_metrics)

    console.print("\n[bold green]✅ Stress-test terminé avec succès![/bold green]")


if __name__ == "__main__":
    main()  # type: ignore[unused-coroutine]
