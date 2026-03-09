from __future__ import annotations

import base64
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from dora import Node

try:
    from llama_cpp import Llama

    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Configuration pour le modèle Qwen2.5-VL-3B-Instruct (87 Hz target)
# Fréquence cible: 87 Hz (~12 ms) - compatible industrie (réseau, robotique)
MODEL_PATH = "Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
LORA_PATH = "reflex_instinct_15MB.bin"

# Prompt pour le contrôle de stabilité
STABILITY_PROMPT = "stability_control"


class ReflexEngine:
    """
    Système 1 (Produit/Cérébellum) - TT-Distill

    Objectif: 87 Hz (~12 ms) pour Qwen2.5-VL-3B-Instruct
    - Fréquence > humain (~60 Hz)
    - Compatible standards industriels (réseau, robotique)
    - Support multimodal (vidéo, texte, images)
    """

    def __init__(self, model_path: str, lora_path: str | None = None, n_ctx: int = 256):
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python requis pour le moteur de réflexe")

        self.model_path = model_path
        self.lora_path = lora_path

        # Charger le modèle directement en mémoire (zéro copie)

        # Préparer les arguments de chargement
        llama_args = {
            "model_path": model_path,
            "n_ctx": n_ctx,
            "n_batch": 1,
            "n_gpu_layers": 45,  # Utiliser Metal sur macOS
            "verbose": False,
        }

        # Ajouter l'adaptateur DoRA si disponible
        if lora_path and Path(lora_path).exists():
            llama_args["adapter_path"] = lora_path
            llama_args["adapter_type"] = "lora"  # DoRA utilise l'API LoRA de llama-cpp

        self.model = Llama(**llama_args)  # type: ignore[arg-type]

    def query(self, prompt: str = STABILITY_PROMPT, max_tokens: int = 256, temperature: float = 0.7) -> tuple[float, str]:
        """
        Classique query method for complete responses (System 2 / Resolver mode)

        Args:
            prompt: Prompt text
            max_tokens: Maximum tokens to generate (default: 256 for complete responses)
            temperature: Sampling temperature (default: 0.7 for creative responses)

        Returns:
            tuple: (latency_ms, response_text)
        """
        start_time = time.perf_counter()

        # Inférence complète (sans streaming, température configurable)
        output = self.model(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=None,
            echo=False,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extraire la réponse
        response_text = output["choices"][0]["text"] if output.get("choices") else ""  # type: ignore[index, union-attr]

        return latency_ms, response_text

    def query_reflex(self, prompt: str = STABILITY_PROMPT) -> tuple[float, str]:
        """
        Exécuter un réflexe avec injection directe inputs_embeds

        Args:
            prompt: Prompt de stabilité (par défaut: "stability_control")

        Returns:
            tuple: (latency_ms, response_text)
        """
        start_time = time.perf_counter()

        # Inférence directe (sans streaming, température 0 pour déterminisme)
        output = self.model(
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            stop=None,
            echo=False,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extraire la réponse
        response_text = output["choices"][0]["text"] if output.get("choices") else ""  # type: ignore[index, union-attr]

        return latency_ms, response_text

    def query_with_embeds(
        self, embeddings: np.ndarray, prompt: str = STABILITY_PROMPT
    ) -> tuple[float, str]:
        """
        Exécuter un réflexe avec injection d'embeddings (bypass tokenizer)

        Cette méthode permet d'injecter directement les embeddings calculés
        par le S2 (Resolver/Cortex) pour éviter la surcharge du tokenizer.

        Args:
            embeddings: Embeddings calculés par le S2 (shape: [1, seq_len, hidden_size])
            prompt: Prompt de base pour le contexte

        Returns:
            tuple: (latency_ms, response_text)
        """
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python requis")

        start_time = time.perf_counter()

        # Injection directe des embeddings (zéro copie)
        output = self.model(  # type: ignore[call-arg]
            prompt=prompt,
            inputs_embeds=embeddings,
            max_tokens=1,
            temperature=0.0,
            stop=None,
            echo=False,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        response_text = output["choices"][0]["text"] if output.get("choices") else ""  # type: ignore[index, union-attr]

        return latency_ms, response_text

    def close(self) -> None:
        """Libérer les ressources du modèle"""
        if hasattr(self, "model") and self.model:
            del self.model


def main() -> None:
    """Point d'entrée principal pour Dora"""
    if not LLAMA_CPP_AVAILABLE:
        sys.exit(1)

    # Initialiser le moteur de réflexe avec DoRA adapter
    lora_path = LORA_PATH if Path(LORA_PATH).exists() else None
    reflex_engine = ReflexEngine(model_path=MODEL_PATH, lora_path=lora_path)

    node = Node()

    try:
        for event in node:
            if event["type"] == "INPUT" and event["id"] == "intent_tensor":
                time.perf_counter()

                # 1. Exécuter le réflexe
                latency_ms, _response_text = reflex_engine.query_reflex(
                    STABILITY_PROMPT
                )

                # 2. Envoyer la commande moteur (latence mesurée)
                node.send_output(
                    "motor_command", np.array([latency_ms], dtype=np.float32).tobytes()
                )

                # 3. Afficher le statut

    finally:
        # Libérer les ressources
        reflex_engine.close()


class MultimodalReflexEngine(ReflexEngine):
    """Extension du ReflexEngine pour le support multimodal complet."""

    def query_with_image(self, prompt: str, image_path: str) -> tuple[float, str]:
        """
        Exécuter un réflexe avec une image.

        Args:
            prompt: Prompt de texte
            image_path: Chemin vers l'image

        Returns:
            tuple: (latency_ms, response_text)
        """
        start_time = time.perf_counter()

        with Path(image_path).open("rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Inférence multimodale
        output = self.model(  # type: ignore[call-arg]
            prompt=prompt,
            images=[image_base64],
            max_tokens=1,
            temperature=0.0,
            stop=None,
            echo=False,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        response_text = output["choices"][0]["text"] if output.get("choices") else ""  # type: ignore[index, union-attr]

        return latency_ms, response_text

    def query_with_video(
        self, prompt: str, video_path: str, frame_indices: list[int] | None = None
    ) -> tuple[float, str]:
        """
        Exécuter un réflexe avec une vidéo.

        Args:
            prompt: Prompt de texte
            video_path: Chemin vers la vidéo
            frame_indices: Indices des frames à extraire (None = toutes les frames)

        Returns:
            tuple: (latency_ms, response_text)
        """
        start_time = time.perf_counter()

        if frame_indices is None:
            frame_indices = [0, 10, 20]  # Frames par défaut

        images = []
        for idx in frame_indices:
            # Extraire une frame spécifique avec ffmpeg
            result = subprocess.run(  # noqa: S603
                [
                    "/usr/bin/ffmpeg",  # Absolute path
                    "-i",
                    video_path,
                    "-ss",
                    str(idx),
                    "-vframes",
                    "1",
                    "-f",
                    "image2",
                    "-",
                ],
                capture_output=True,
                check=True,
            )
            if result.stdout:
                images.append(base64.b64encode(result.stdout).decode("utf-8"))

        # Inférence multimodale avec plusieurs frames
        output = self.model(  # type: ignore[call-arg]
            prompt=prompt,
            images=images,
            max_tokens=1,
            temperature=0.0,
            stop=None,
            echo=False,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        response_text = output["choices"][0]["text"] if output.get("choices") else ""  # type: ignore[index, union-attr]

        return latency_ms, response_text


if __name__ == "__main__":
    main()
