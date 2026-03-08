import numpy as np

from src.orchestration.arc_math_solver import (
    AffineSpace,
    BooleanLattice,
    DihedralGroup,
    TopologicalGraph,
    VectorSpace,
)


def test_dihedral_group() -> None:
    grid = np.array([[1, 2], [3, 4]])

    # Rotation 90 (clockwise)
    rot90 = DihedralGroup.rotate_90(grid)
    expected_rot90 = np.array([[3, 1], [4, 2]])
    np.testing.assert_array_equal(rot90, expected_rot90)

    # Flip H
    flip_h = DihedralGroup.flip_h(grid)
    expected_flip_h = np.array([[2, 1], [4, 3]])
    np.testing.assert_array_equal(flip_h, expected_flip_h)


def test_vector_space_scaling() -> None:
    grid = np.array([[1, 0], [0, 1]])
    scaled = VectorSpace.scale(grid, 2, 2)
    assert scaled.shape == (4, 4)
    assert scaled[0, 0] == 1
    assert scaled[0, 1] == 1
    assert scaled[2, 2] == 1


def test_affine_space_translation() -> None:
    grid = np.array([[1, 0, 0], [0, 0, 0]])
    translated = AffineSpace.translate_modulo(grid, 1, 1)
    # [1, 0, 0]    [0, 0, 0]
    # [0, 0, 0] -> [0, 1, 0]
    assert translated[1, 1] == 1
    assert translated[0, 0] == 0


def test_boolean_lattice() -> None:
    grid_a = np.array([[1, 0], [0, 1]])
    grid_b = np.array([[0, 2], [0, 2]])

    # Use overlay instead of union
    overlay = BooleanLattice.overlay(grid_a, grid_b, bg=0)
    expected_overlay = np.array([[1, 2], [0, 2]])
    np.testing.assert_array_equal(overlay, expected_overlay)


def test_topological_flood_fill() -> None:
    grid = np.zeros((5, 5), dtype=int)
    # Flood fill at (2, 2) from color 0 to color 7
    filled = TopologicalGraph.flood_fill(grid, 2, 2, 0, 7)
    assert filled[2, 2] == 7
    assert filled[0, 0] == 7
    assert filled.all()
