"""Post-Silicon Hyperspecialization for Hardware-Aware Autonomy.

This module implements the final layer of the TT-Distill architecture:
hardware-aware autonomy where the System 1 (Cervelet) can adjust hardware
parameters in real-time without waking the System 2 (Resolver).

Architecture:
    S1 Reflex → Hardware Hooks → NPU/TPU Parameter Adjustment
                          ↓
            Real-time Optimization (8 ms loop)

Key Features:
- Hardware parameter exposure via API/Hooks
- Safe adjustment of NPU/TPU parameters
- Temperature and threshold clamping control
- Cache and scheduling optimization

References:
- ArchAgent (AlphaEvolve): Post-silicon specialization
- Hardware-Aware Neural Architecture Search (2026)
"""

from __future__ import annotations

import asyncio
import platform
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


class HardwareComponent(Enum):
    """Composants hardware exposés."""

    NPU = "npu"
    TPU = "tpu"
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    CACHE = "cache"


# Magic value constants
SUBPROCESS_TIMEOUT_SECONDS = 5
NPU_HIGH_TEMP_THRESHOLD = 65
CACHE_HIT_RATE_LOW_THRESHOLD = 0.8
NPU_HIGH_UTILIZATION_THRESHOLD = 0.85
MIN_NVIDIA_SMI_PARTS = 4


class AdjustmentType(Enum):
    """Types d'ajustement hardware."""

    TEMPERATURE = "temperature"  # Température cognitive du modèle
    THRESHOLD_CLAMP = "threshold_clamp"  # Seuil de clamping algébrique
    CACHE_SIZE = "cache_size"  # Taille du cache
    SCHEDULING_POLICY = "scheduling_policy"  # Politique d'ordonnancement
    POWER_MODE = "power_mode"  # Mode énergétique
    FREQUENCY = "frequency"  # Fréquence de clock


@dataclass
class HardwareParameter:
    """Paramètre hardware ajustable."""

    component: HardwareComponent
    parameter_type: AdjustmentType
    name: str
    current_value: float
    min_value: float
    max_value: float
    default_value: float
    unit: str = ""
    description: str = ""
    is_safe: bool = True  # True si l'ajustement est sans risque

    def to_dict(self) -> dict[str, Any]:
        """Convertir en dictionnaire."""
        return {
            "component": self.component.value,
            "parameter_type": self.parameter_type.value,
            "name": self.name,
            "current_value": self.current_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "default_value": self.default_value,
            "unit": self.unit,
            "description": self.description,
            "is_safe": self.is_safe,
        }


@dataclass
class HardwareAdjustment:
    """Ajustement hardware proposé."""

    parameter: HardwareParameter
    target_value: float
    reason: str
    estimated_impact: str  # "latency", "power", "accuracy", etc.
    confidence: float = 1.0  # Confiance de l'ajustement (0-1)


@dataclass
class HardwareState:
    """État actuel du hardware."""

    npu_temperature: float = 0.0
    npu_utilization: float = 0.0
    tpu_temperature: float = 0.0
    tpu_utilization: float = 0.0
    gpu_temperature: float = 0.0
    gpu_utilization: float = 0.0
    cpu_temperature: float = 0.0
    cpu_utilization: float = 0.0
    memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    power_consumption: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convertir en dictionnaire."""
        return {
            "npu_temperature": self.npu_temperature,
            "npu_utilization": self.npu_utilization,
            "tpu_temperature": self.tpu_temperature,
            "tpu_utilization": self.tpu_utilization,
            "gpu_temperature": self.gpu_temperature,
            "gpu_utilization": self.gpu_utilization,
            "cpu_temperature": self.cpu_temperature,
            "cpu_utilization": self.cpu_utilization,
            "memory_usage": self.memory_usage,
            "cache_hit_rate": self.cache_hit_rate,
            "power_consumption": self.power_consumption,
        }


class HardwareHook:
    """Hook pour l'ajustement des paramètres hardware."""

    def __init__(
        self,
        component: HardwareComponent,
        parameter_type: AdjustmentType,
        apply_fn: Callable[[float], bool],
        validate_fn: Callable[[float], bool] | None = None,
    ) -> None:
        """Initialiser le hook hardware.

        Args:
            component: Composant hardware cible
            parameter_type: Type d'ajustement
            apply_fn: Fonction pour appliquer l'ajustement
            validate_fn: Fonction optionnelle pour valider l'ajustement
        """
        self.component = component
        self.parameter_type = parameter_type
        self.apply_fn = apply_fn
        self.validate_fn = validate_fn or (lambda x: True)

    def apply(self, value: float) -> bool:
        """Appliquer l'ajustement.

        Args:
            value: Nouvelle valeur

        Returns:
            True si l'ajustement a réussi
        """
        if not self.validate_fn(value):
            logger.warning(
                f"❌ Validation échouée pour {self.parameter_type.value}={value}"
            )
            return False

        return self.apply_fn(value)


class PostSiliconController:
    """Contrôleur Post-Silicon pour l'autonomie hardware.

    Ce contrôleur permet au Système 1 (Cervelet) d'ajuster les paramètres
    hardware en temps réel sans réveiller le Système 2 (Resolver).

    Workflow:
        1. Monitoring: Collecter l'état actuel du hardware
        2. Analysis: Analyser les besoins d'optimisation
        3. Adjustment: Proposer et appliquer les ajustements
        4. Validation: Vérifier l'impact des ajustements
    """

    def __init__(self) -> None:
        """Initialiser le contrôleur Post-Silicon."""
        self._parameters: dict[str, HardwareParameter] = {}
        self._hooks: dict[str, HardwareHook] = {}
        self._hardware_state = HardwareState()
        self._adjustment_history: list[HardwareAdjustment] = []
        self._running = False

        # Initialiser les paramètres par défaut
        self._init_default_parameters()

        logger.info("🔧 Post-Silicon Controller initialisé")

    def _init_default_parameters(self) -> None:
        """Initialiser les paramètres hardware par défaut."""
        # NPU Temperature
        self._parameters["npu_temperature"] = HardwareParameter(
            component=HardwareComponent.NPU,
            parameter_type=AdjustmentType.TEMPERATURE,
            name="npu_cognitive_temperature",
            current_value=0.7,
            min_value=0.0,
            max_value=1.0,
            default_value=0.7,
            unit="",
            description="Température cognitive du NPU (0=froid/déterministe, 1=chaud/créatif)",
            is_safe=True,
        )

        # NPU Threshold Clamp
        self._parameters["npu_threshold_clamp"] = HardwareParameter(
            component=HardwareComponent.NPU,
            parameter_type=AdjustmentType.THRESHOLD_CLAMP,
            name="npu_threshold_clamp",
            current_value=0.95,
            min_value=0.5,
            max_value=1.0,
            default_value=0.95,
            unit="",
            description="Seuil de clamping algébrique pour la stabilité",
            is_safe=True,
        )

        # Cache Size
        self._parameters["cache_size"] = HardwareParameter(
            component=HardwareComponent.CACHE,
            parameter_type=AdjustmentType.CACHE_SIZE,
            name="cache_size_mb",
            current_value=64.0,
            min_value=16.0,
            max_value=256.0,
            default_value=64.0,
            unit="MB",
            description="Taille du cache L1 pour le RAG sub-millisecond",
            is_safe=True,
        )

        # Scheduling Policy
        self._parameters["scheduling_policy"] = HardwareParameter(
            component=HardwareComponent.NPU,
            parameter_type=AdjustmentType.SCHEDULING_POLICY,
            name="scheduling_policy",
            current_value=1.0,  # 1=real-time, 0=balanced
            min_value=0.0,
            max_value=1.0,
            default_value=1.0,
            unit="",
            description="Politique d'ordonnancement (1=real-time, 0=balanced)",
            is_safe=True,
        )

        # Power Mode
        self._parameters["power_mode"] = HardwareParameter(
            component=HardwareComponent.NPU,
            parameter_type=AdjustmentType.POWER_MODE,
            name="power_mode",
            current_value=0.5,  # 0=eco, 1=performance
            min_value=0.0,
            max_value=1.0,
            default_value=0.5,
            unit="",
            description="Mode énergétique (0=eco, 1=performance)",
            is_safe=True,
        )

        # Frequency
        self._parameters["frequency"] = HardwareParameter(
            component=HardwareComponent.NPU,
            parameter_type=AdjustmentType.FREQUENCY,
            name="frequency_ghz",
            current_value=2.0,
            min_value=1.0,
            max_value=4.0,
            default_value=2.0,
            unit="GHz",
            description="Fréquence de clock du NPU",
            is_safe=False,  # Ajustement risqué
        )

    def register_hook(
        self,
        component: HardwareComponent,
        parameter_type: AdjustmentType,
        name: str,
        apply_fn: Callable[[float], bool],
        validate_fn: Callable[[float], bool] | None = None,
    ) -> None:
        """Enregistrer un hook hardware.

        Args:
            component: Composant hardware cible
            parameter_type: Type d'ajustement
            name: Nom du hook
            apply_fn: Fonction pour appliquer l'ajustement
            validate_fn: Fonction optionnelle pour valider l'ajustement
        """
        hook = HardwareHook(component, parameter_type, apply_fn, validate_fn)
        self._hooks[name] = hook

        # Mettre à jour le paramètre associé
        param_key = f"{component.value}_{parameter_type.value}"
        if param_key in self._parameters:
            logger.info(f"🔗 Hook enregistré: {name}")

    def get_parameter(self, name: str) -> HardwareParameter | None:
        """Récupérer un paramètre par nom.

        Args:
            name: Nom du paramètre (peut être le nom interne comme "npu_temperature"
                  ou le nom externe comme "npu_cognitive_temperature")

        Returns:
            HardwareParameter ou None si non trouvé
        """
        # Essayer d'abord par nom interne
        param = self._parameters.get(name)
        if param:
            return param

        # Sinon, chercher par nom externe (param.name)
        for p in self._parameters.values():
            if p.name == name:
                return p

        return None

    def get_all_parameters(self) -> list[HardwareParameter]:
        """Récupérer tous les paramètres."""
        return list(self._parameters.values())

    def get_hardware_state(self) -> HardwareState:
        """Récupérer l'état actuel du hardware."""
        return self._hardware_state

    async def update_hardware_state(self) -> HardwareState:
        """Mettre à jour l'état du hardware.

        Sur macOS, utilise `ioreg` pour collecter les métriques GPU/NPU.
        Sur Linux, utilise `nvidia-smi` pour les GPU NVIDIA.
        En mode simulation, génère des valeurs aléatoires.
        """
        system = platform.system()

        if system == "Darwin":  # macOS
            await self._update_hardware_macos()
        elif system == "Linux":  # Linux
            await self._update_hardware_linux()
        else:
            # Mode simulation pour autres systèmes
            self._simulate_hardware_state()

        return self._hardware_state

    async def _update_hardware_macos(self) -> None:
        """Collecter les métriques hardware sur macOS via ioreg."""
        try:
            # Récupérer les informations GPU via ioreg
            proc = await asyncio.create_subprocess_exec(
                "ioreg",
                "-r",
                "-c",
                "AppleGPU",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS
            )

            if proc.returncode == 0:
                output = stdout.decode()
                # Parser les valeurs de température et d'utilisation
                # Format typique: "Temperature = {23}" ou "GPUUtilization = {45}"

                temp_match = re.search(r"Temperature\s*=\s*\{(\d+)\}", output)
                if temp_match:
                    self._hardware_state.gpu_temperature = float(temp_match.group(1))

                util_match = re.search(r"GPUUtilization\s*=\s*\{(\d+)\}", output)
                if util_match:
                    self._hardware_state.gpu_utilization = (
                        float(util_match.group(1)) / 100.0
                    )

                # CPU temperature (si disponible)
                cpu_proc = await asyncio.create_subprocess_exec(
                    "ioreg",
                    "-r",
                    "-c",
                    "CPU",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                cpu_stdout, _ = await asyncio.wait_for(
                    cpu_proc.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS
                )

                if cpu_proc.returncode == 0:
                    cpu_temp_match = re.search(
                        r"Temperature\s*=\s*\{(\d+)\}", cpu_stdout.decode()
                    )
                    if cpu_temp_match:
                        self._hardware_state.cpu_temperature = float(
                            cpu_temp_match.group(1)
                        )

            # Mémoire système
            mem_proc = await asyncio.create_subprocess_exec(
                "sysctl",
                "hw.memsize",
                "vm.loadavg",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(
                mem_proc.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS
            )

            if mem_proc.returncode == 0:
                self._hardware_state.memory_usage = np.random.uniform(
                    0.4, 0.8
                )  # Fallback

        except (TimeoutError, FileNotFoundError, Exception) as e:
            logger.warning(f"Échec de collecte hardware macOS: {e}")
            self._simulate_hardware_state()

    async def _update_hardware_linux(self) -> None:
        """Collecter les métriques hardware sur Linux via nvidia-smi."""
        try:
            # Récupérer les informations GPU NVIDIA
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS
            )

            if proc.returncode == 0:
                lines = stdout.decode().strip().split("\n")
                for line in lines:
                    parts = line.split(",")
                    if len(parts) >= MIN_NVIDIA_SMI_PARTS:
                        self._hardware_state.gpu_temperature = float(parts[0].strip())
                        self._hardware_state.gpu_utilization = (
                            float(parts[1].strip()) / 100.0
                        )

                        mem_used = float(parts[2].strip())
                        mem_total = float(parts[3].strip())
                        self._hardware_state.memory_usage = (
                            mem_used / mem_total if mem_total > 0 else 0.0
                        )

            # CPU temperature (via lm-sensors)
            cpu_proc = await asyncio.create_subprocess_exec(
                "sensors",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            cpu_stdout, _ = await asyncio.wait_for(
                cpu_proc.communicate(), timeout=SUBPROCESS_TIMEOUT_SECONDS
            )

            if cpu_proc.returncode == 0:
                temp_match = re.search(r"Core 0:\s+([\d.]+)\s+", cpu_stdout.decode())
                if temp_match:
                    self._hardware_state.cpu_temperature = float(temp_match.group(1))

        except (TimeoutError, FileNotFoundError, Exception) as e:
            logger.warning(f"Échec de collecte hardware Linux: {e}")
            self._simulate_hardware_state()

    def _simulate_hardware_state(self) -> None:
        """Générer des valeurs de hardware aléatoires (mode simulation)."""
        self._hardware_state.npu_temperature = np.random.uniform(40, 70)
        self._hardware_state.npu_utilization = np.random.uniform(0.3, 0.9)
        self._hardware_state.tpu_temperature = np.random.uniform(35, 65)
        self._hardware_state.tpu_utilization = np.random.uniform(0.2, 0.8)
        self._hardware_state.gpu_temperature = np.random.uniform(45, 75)
        self._hardware_state.gpu_utilization = np.random.uniform(0.4, 0.95)
        self._hardware_state.cpu_temperature = np.random.uniform(30, 60)
        self._hardware_state.cpu_utilization = np.random.uniform(0.1, 0.7)
        self._hardware_state.memory_usage = np.random.uniform(0.4, 0.8)
        self._hardware_state.cache_hit_rate = np.random.uniform(0.7, 0.99)
        self._hardware_state.power_consumption = np.random.uniform(10, 50)

    def propose_adjustment(
        self,
        reason: str,
        estimated_impact: str,
        confidence: float = 1.0,
    ) -> HardwareAdjustment | None:
        """Proposer un ajustement hardware basé sur l'état actuel.

        Args:
            reason: Raison de l'ajustement
            estimated_impact: Impact estimé ("latency", "power", "accuracy")
            confidence: Confiance de l'ajustement (0-1)

        Returns:
            HardwareAdjustment ou None si aucun ajustement nécessaire
        """
        state = self._hardware_state

        # Logique d'ajustement basée sur l'état
        if state.npu_temperature > NPU_HIGH_TEMP_THRESHOLD:
            # Température élevée → réduire la température cognitive
            param = self._parameters.get("npu_temperature")
            if param:
                return HardwareAdjustment(
                    parameter=param,
                    target_value=max(param.min_value, param.current_value - 0.1),
                    reason=reason,
                    estimated_impact=estimated_impact,
                    confidence=confidence,
                )

        if state.cache_hit_rate < CACHE_HIT_RATE_LOW_THRESHOLD:
            # Cache hit rate faible → augmenter la taille du cache
            param = self._parameters.get("cache_size")
            if param:
                return HardwareAdjustment(
                    parameter=param,
                    target_value=min(param.max_value, param.current_value + 16.0),
                    reason=reason,
                    estimated_impact=estimated_impact,
                    confidence=confidence,
                )

        if state.npu_utilization > NPU_HIGH_UTILIZATION_THRESHOLD:
            # Utilisation élevée → basculer en mode real-time
            param = self.get_parameter("scheduling_policy")
            if param:
                return HardwareAdjustment(
                    parameter=param,
                    target_value=1.0,
                    reason=reason,
                    estimated_impact=estimated_impact,
                    confidence=confidence,
                )

        return None

    async def apply_adjustment(self, adjustment: HardwareAdjustment) -> bool:
        """Appliquer un ajustement hardware.

        Args:
            adjustment: Ajustement à appliquer

        Returns:
            True si l'ajustement a réussi
        """
        if adjustment.parameter is None:
            logger.warning("❌ Ajustement avec paramètre None")  # type: ignore[unreachable]
            return False

        param_key = f"{adjustment.parameter.component.value}_{adjustment.parameter.parameter_type.value}"
        hook = self._hooks.get(param_key)

        if hook:
            success = hook.apply(adjustment.target_value)
        else:
            # Mode simulation: mettre à jour le paramètre directement
            # Chercher le paramètre par son nom interne ou externe
            param = self.get_parameter(adjustment.parameter.name)
            if param:
                param.current_value = adjustment.target_value
                success = True
            else:
                success = False

        if success:
            self._adjustment_history.append(adjustment)
            logger.info(
                f"✅ Ajustement appliqué: {adjustment.parameter.name} = "
                f"{adjustment.target_value} ({adjustment.estimated_impact})"
            )
        else:
            logger.warning(f"❌ Échec d'ajustement: {adjustment.parameter.name}")

        return success

    async def run_optimization_loop(self, interval_ms: float = 8.0) -> None:
        """Exécuter la boucle d'optimisation Post-Silicon.

        Args:
            interval_ms: Intervalle de la boucle en ms (défaut: 8 ms pour 125 Hz)
        """
        self._running = True
        interval_s = interval_ms / 1000.0

        logger.info(
            f"🚀 Boucle d'optimisation Post-Silicon démarrée ({interval_ms} ms)"
        )

        try:
            while self._running:
                start_time = asyncio.get_event_loop().time()

                # 1. Mettre à jour l'état du hardware
                await self.update_hardware_state()

                # 2. Proposer un ajustement si nécessaire
                adjustment = self.propose_adjustment(
                    reason="Optimisation automatique",
                    estimated_impact="latency",
                    confidence=0.9,
                )

                if adjustment:
                    await self.apply_adjustment(adjustment)

                # 3. Respecter l'intervalle
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, interval_s - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        except asyncio.CancelledError:
            logger.info("🛑 Boucle d'optimisation arrêtée")
        finally:
            self._running = False

    def stop(self) -> None:
        """Arrêter le contrôleur."""
        self._running = False
        logger.info("🛑 Post-Silicon Controller arrêté")

    def get_adjustment_history(self) -> list[HardwareAdjustment]:
        """Récupérer l'historique des ajustements."""
        return self._adjustment_history.copy()

    def reset_history(self) -> None:
        """Réinitialiser l'historique des ajustements."""
        self._adjustment_history.clear()


class PostSiliconAPI:
    """API pour l'interaction avec le contrôleur Post-Silicon.

    Cette API expose les fonctionnalités du contrôleur via des méthodes
    simples pour que le Système 1 puisse ajuster le hardware en temps réel.
    """

    def __init__(self, controller: PostSiliconController) -> None:
        """Initialiser l'API.

        Args:
            controller: Instance du contrôleur Post-Silicon
        """
        self.controller = controller
        self._optimization_task: asyncio.Task[None] | None = None

    async def get_parameters(self) -> list[dict[str, Any]]:
        """Récupérer tous les paramètres hardware."""
        return [p.to_dict() for p in self.controller.get_all_parameters()]

    async def get_parameter(self, name: str) -> dict[str, Any] | None:
        """Récupérer un paramètre par nom."""
        param = self.controller.get_parameter(name)
        return param.to_dict() if param else None

    async def set_parameter(self, name: str, value: float) -> bool:
        """Définir la valeur d'un paramètre.

        Args:
            name: Nom du paramètre
            value: Nouvelle valeur

        Returns:
            True si la valeur a été définie avec succès
        """
        param = self.controller.get_parameter(name)
        if not param:
            return False

        if value < param.min_value or value > param.max_value:
            logger.warning(
                f"Valeur hors limites: {value} (min={param.min_value}, max={param.max_value})"
            )
            return False

        param.current_value = value
        return True

    async def get_state(self) -> dict[str, Any]:
        """Récupérer l'état actuel du hardware."""
        return self.controller.get_hardware_state().to_dict()

    async def propose_and_apply(self, reason: str, impact: str) -> bool:
        """Proposer et appliquer un ajustement.

        Args:
            reason: Raison de l'ajustement
            impact: Impact estimé

        Returns:
            True si l'ajustement a été appliqué
        """
        adjustment = self.controller.propose_adjustment(reason, impact)
        if adjustment:
            return await self.controller.apply_adjustment(adjustment)
        return False

    async def start_optimization(self, interval_ms: float = 8.0) -> None:
        """Démarrer la boucle d'optimisation.

        Args:
            interval_ms: Intervalle de la boucle en ms
        """
        self.controller._running = True
        self._optimization_task = asyncio.create_task(
            self.controller.run_optimization_loop(interval_ms)
        )

    async def stop_optimization(self) -> None:
        """Arrêter la boucle d'optimisation."""
        self.controller.stop()


async def demo_post_silicon() -> None:
    """Démonstration du contrôleur Post-Silicon."""

    # Initialiser le contrôleur
    controller = PostSiliconController()

    # Afficher les paramètres
    for _param in controller.get_all_parameters():
        pass

    # Simuler l'optimisation

    # Mettre à jour l'état et proposer un ajustement
    await controller.update_hardware_state()
    controller.get_hardware_state()

    # Proposer et appliquer un ajustement
    adjustment = controller.propose_adjustment(
        reason="Optimisation de latence",
        estimated_impact="latency",
    )

    if adjustment:
        await controller.apply_adjustment(adjustment)
    else:
        pass

    # Démarrer la boucle d'optimisation
    await controller.run_optimization_loop(interval_ms=8.0)

    # Arrêter après 2 secondes
    await asyncio.sleep(2.0)
    controller.stop()


if __name__ == "__main__":
    asyncio.run(demo_post_silicon())
