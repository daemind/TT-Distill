"""TT-Distill Demo Suite.

Démonstrations interactives pour prouver chaque pilier de l'architecture TT-Distill.

Demos:
    - demo_stress_test: Stress-Test de Fréquence (S1 Reflex vs RAG Classique)
    - demo_maca_dora: Autopsie Algébrique (MACA & DoRA Visualizer)
    - demo_reality_filter: Reality Filter (Compliance AGENT.md)
    - demo_post_silicon: Symbiose Post-Silicon (Hardware Profiling)
"""

from .demo_maca_dora import main as demo_maca_dora
from .demo_post_silicon import main as demo_post_silicon
from .demo_reality_filter import main as demo_reality_filter
from .demo_stress_test import main as demo_stress_test

__all__ = [
    "demo_maca_dora",
    "demo_post_silicon",
    "demo_reality_filter",
    "demo_stress_test",
]
