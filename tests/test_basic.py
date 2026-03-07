"""Test suite for TT-Distill minimal codebase."""

def test_imports_and_availability() -> None:
    """Ensure that the core mathematical solver is importable."""
    from src.orchestration.arc_math_solver import (
        prove_correctness,
        synthesize_program,
    )

    assert synthesize_program is not None
    assert prove_correctness is not None
