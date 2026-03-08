"""Orchestration module for mathematical abstract continuous agents."""

from .arc_math_solver import prove_correctness, synthesize_program
from .arc_solvers import solve_task
from .maca import LatentTrajectory, MACAEngine, SinkhornBarycenter, TTDistillBridge
from .metal_swap import MetalDoRASwapper
from .moa_gating import MoAGater

__all__ = [
    "LatentTrajectory",
    "MACAEngine",
    "MetalDoRASwapper",
    "MoAGater",
    "SinkhornBarycenter",
    "TTDistillBridge",
    "prove_correctness",
    "solve_task",
    "synthesize_program",
]
