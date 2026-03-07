# ruff: noqa

import logging
from collections.abc import Callable

import numpy as np

logger = logging.getLogger(__name__)

Grid = np.ndarray  # 2D int array


# ═══════════════════════════════════════════════════════════════════════
#  ALGEBRAIC STRUCTURES & MATHEMATICAL LAWS
# ═══════════════════════════════════════════════════════════════════════

class AlgebraicSpace:
    """Formal definition of the resolution spaces."""
    DIHEDRAL_GROUP = "D4_Group"       # Isometric spatial transformations
    COLOR_FIELD_F10 = "F10_Field"     # Bijective mappings in color space
    BOOLEAN_LATTICE = "Bool_Lattice"  # Set operations, masks, overlays
    TOPOLOGICAL_GRAPH = "Topo_Graph"  # Connectivity, flood fill, closures
    VECTOR_SPACE = "Vector_Space"     # Gravity, translations, scaling
    AFFINE_SPACE = "Affine_Space"     # Translations with periodic boundaries
    CELLULAR_AUTOMATA = "Cell_Auto"   # Dynamical systems (Fixed points)
    HOMOLOGY = "Homology"             # Topological invariants (Betti numbers, holes)
    MORPHOLOGICAL_SPACE = "Morph_Space" # Minkowski sums/differences, dilation, erosion
    QUOTIENT_GRAPH = "Quotient_Graph"   # Components, sorting by cardinality
    PROJECTIVE_SPACE = "Proj_Space"     # Local reflections, raycasting, partial symmetry
    HARMONIC_SPACE = "Harm_Space"       # Fractals, Self-similarity, 2D periodic signals
    GENERATIVE_GRAMMAR = "Gen_Grammar"  # 1D Raycasting, periodic sequences
    SYMMETRY_QUOTIENT = "Sym_Quotient"  # Reconstruct masked regions via group symmetry
    TRANSLATION_PERIOD = "Trans_Period"  # Repair periodic patterns via orbit projection
    SHAPE_MORPHISM = "Shape_Morph"      # Extract/map objects via topological invariants


# --- 1. Groupe Diédral D4 (Isométries Spatiales) ---

class DihedralGroup:
    @staticmethod
    def identity(grid: Grid) -> Grid: return grid.copy()

    @staticmethod
    def rotate_90(grid: Grid) -> Grid: return np.rot90(grid, k=-1)

    @staticmethod
    def rotate_180(grid: Grid) -> Grid: return np.rot90(grid, k=2)

    @staticmethod
    def rotate_270(grid: Grid) -> Grid: return np.rot90(grid, k=-3)

    @staticmethod
    def flip_h(grid: Grid) -> Grid: return np.fliplr(grid)

    @staticmethod
    def flip_v(grid: Grid) -> Grid: return np.flipud(grid)

    @staticmethod
    def transpose_main(grid: Grid) -> Grid: return grid.T

    @staticmethod
    def transpose_anti(grid: Grid) -> Grid: return np.fliplr(grid).T

    @classmethod
    def all_elements(cls):  # type: ignore[no-untyped-def]
        return [cls.identity, cls.rotate_90, cls.rotate_180, cls.rotate_270,
                cls.flip_h, cls.flip_v, cls.transpose_main, cls.transpose_anti]

# --- 2. Algèbre Vectorielle (Scaling) ---

class VectorSpace:
    @staticmethod
    def scale(grid: Grid, sy: int, sx: int) -> Grid:
        if sy < 1 or sx < 1: raise ValueError("Scale factors must be >= 1")
        return np.repeat(np.repeat(grid, sy, axis=0), sx, axis=1)

    @staticmethod
    def downscale(grid: Grid, sy: int, sx: int) -> Grid:
        if grid.shape[0] % sy != 0 or grid.shape[1] % sx != 0:
            raise ValueError("Grid not divisible")
        return grid[::sy, ::sx]

    @staticmethod
    def tile(grid: Grid, ty: int, tx: int) -> Grid:
        if ty < 1 or tx < 1: raise ValueError("Tile factors must be >= 1")
        return np.tile(grid, (ty, tx))

# --- 2.1 Espace Affine (Translations Périodiques & Déformations) ---

class AffineSpace:
    @staticmethod
    def translate_modulo(grid: Grid, dy: int, dx: int) -> Grid:
        """Translation on a torus Z/hZ x Z/wZ."""
        return np.roll(grid, shift=(dy, dx), axis=(0, 1))

    @staticmethod
    def reflect_main_diagonal(grid: Grid) -> Grid:
        """y = x axis reflection (Transpose)."""
        return grid.T

    @staticmethod
    def reflect_anti_diagonal(grid: Grid) -> Grid:
        """y = -x axis reflection."""
        return np.fliplr(grid.T).copy()


# --- 3. Corps Fini F_10 (Color Mapping) ---

class ColorField:
    @staticmethod
    def apply_bijection(grid: Grid, mapping: dict[int, int]) -> Grid:
        result = grid.copy()
        for src, dst in mapping.items():
            result[grid == src] = dst
        return result


# --- 4. Topologie & Graphes ---

class TopologicalGraph:
    @staticmethod
    def flood_fill(grid: Grid, start_x: int, start_y: int, src: int, dst: int) -> Grid:
        """Transitive closure of connectivity graph."""
        if grid[start_x, start_y] != src or src == dst: return grid.copy()
        result = grid.copy()
        H, W = result.shape
        queue = [(start_x, start_y)]
        result[start_x, start_y] = dst
        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == src:
                    result[nr, nc] = dst
                    queue.append((nr, nc))
        return result

# --- 4.1 Homologie (Invariants Topologiques) ---

class Homology:
    @staticmethod
    def fill_enclosures(grid: Grid, bg_color: int = 0, fill_color: int = 2) -> Grid:
        """
        Fills topological holes (Betti-1 = 1).
        Finds all background pixels NOT reachable from the image borders.
        """
        result = grid.copy()
        H, W = grid.shape
        visited = set()
        queue = []

        # Start BFS from all border pixels
        for r in range(H):
            if result[r, 0] == bg_color: queue.append((r, 0)); visited.add((r, 0))
            if result[r, W-1] == bg_color: queue.append((r, W-1)); visited.add((r, W-1))
        for c in range(W):
            if result[0, c] == bg_color: queue.append((0, c)); visited.add((0, c))
            if result[H-1, c] == bg_color: queue.append((H-1, c)); visited.add((H-1, c))

        while queue:
            r, c = queue.pop(0)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if (nr, nc) not in visited and result[nr, nc] == bg_color:
                        visited.add((nr, nc))
                        queue.append((nr, nc))

        # Any bg_color not visited is an enclosure
        for r in range(H):
            for c in range(W):
                if result[r, c] == bg_color and (r, c) not in visited:
                    result[r, c] = fill_color
        return result


# --- 5. Treillis Booléen (Assemblages / Masking) ---

class BooleanLattice:
    @staticmethod
    def crop_nonzero(grid: Grid, bg_color: int = 0) -> Grid:
        coords = np.argwhere(grid != bg_color)
        if coords.size == 0: return grid.copy()
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        return grid[r_min:r_max+1, c_min:c_max+1]

    @staticmethod
    def overlay(grid: Grid, layer: Grid, bg: int = 0) -> Grid:
        """Union logic in boolean space."""
        if grid.shape != layer.shape: return grid.copy()
        return np.where(layer != bg, layer, grid)

    @classmethod
    def symmetric_overlays(cls, grid: Grid) -> list[Grid]:
        return [
            cls.overlay(grid, DihedralGroup.flip_h(grid)),
            cls.overlay(grid, DihedralGroup.flip_v(grid)),
            cls.overlay(grid, DihedralGroup.transpose_main(grid)) if grid.shape[0]==grid.shape[1] else grid,
            cls.overlay(grid, DihedralGroup.transpose_anti(grid)) if grid.shape[0]==grid.shape[1] else grid,
        ]


# --- 6. Systèmes Dynamiques (Gravité & Point Fixe) ---

class ParticleSystem:
    @staticmethod
    def gravity_down(grid: Grid, bg_color: int = 0) -> Grid:
        result = np.full_like(grid, bg_color)
        for c in range(grid.shape[1]):
            col = grid[:, c]
            nz = col[col != bg_color]
            if nz.size > 0: result[-nz.size:, c] = nz
        return result

    @staticmethod
    def gravity_up(grid: Grid, bg_color: int = 0) -> Grid:
        result = np.full_like(grid, bg_color)
        for c in range(grid.shape[1]):
            col = grid[:, c]
            nz = col[col != bg_color]
            if nz.size > 0: result[:nz.size, c] = nz
        return result


class CellularAutomata:
    @staticmethod
    def grow_Moore(grid: Grid, bg_color: int = 0) -> Grid:
        """One step of Moore neighborhood expansion for all objects."""
        H, W = grid.shape
        result = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] != bg_color:
                    for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == bg_color:
                            result[nr, nc] = grid[r, c]
        return result

    @staticmethod
    def run_until_fixed_point(grid: Grid, rule: Callable[[Grid], Grid], max_steps: int = 50) -> Grid:
        """Lim_n->inf T^n(grid) = grid_stable"""
        current = grid.copy()
        for _ in range(max_steps):
            nxt = rule(current)
            if np.array_equal(current, nxt):
                return current
            current = nxt
        return current


# --- 7. Algèbre Morphologique (Minkowski Addition) ---

class MorphologicalSpace:
    @staticmethod
    def dilation(grid: Grid, bg_color: int = 0) -> Grid:
        """A ⊕ B where B is a 3x3 cross (von Neumann). Expands non-bg pixels."""
        result = grid.copy()
        H, W = grid.shape
        for r in range(H):
            for c in range(W):
                if grid[r, c] != bg_color:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == bg_color:
                            result[nr, nc] = grid[r, c]
        return result

    @staticmethod
    def extract_outline(grid: Grid, bg_color: int = 0, outline_color: int = 2) -> Grid:
        """(A ⊕ B) \\ A: Minkowski difference outline (OUTER boundary)."""
        dilated = MorphologicalSpace.dilation(grid, bg_color)
        result = grid.copy()
        result[(dilated != bg_color) & (grid == bg_color)] = outline_color
        return result

    @staticmethod
    def extract_inner_outline(grid: Grid, bg_color: int = 0) -> Grid:
        """Keep only non-bg pixels that are adjacent to bg (INNER boundary)."""
        H, W = grid.shape
        result = np.full_like(grid, bg_color)
        for r in range(H):
            for c in range(W):
                if grid[r, c] != bg_color:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if nr < 0 or nr >= H or nc < 0 or nc >= W or grid[nr, nc] == bg_color:
                            result[r, c] = grid[r, c]
                            break
        return result

    @staticmethod
    def border_fill(grid: Grid, bg_color: int = 0, border_color: int = 1) -> Grid:
        """A ⊕ B_Moore \\ A: Add a 1-pixel border using Moore (8-connected) dilation."""
        H, W = grid.shape
        result = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] != bg_color:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0: continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < H and 0 <= nc < W and result[nr, nc] == bg_color:
                                result[nr, nc] = border_color
        return result


# --- 8. Théorie des Graphes (Quotients et Composantes) ---

class QuotientGraph:
    @staticmethod
    def get_components(grid: Grid, bg_color: int = 0) -> list[tuple[set[tuple[int, int]], int]]:
        """Compute the sets [C] in G/~ where ~ is pixel connectivity."""
        H, W = grid.shape
        visited = set()
        components = []
        for r in range(H):
            for c in range(W):
                color = grid[r, c]
                if color != bg_color and (r, c) not in visited:
                    comp_pixels = set()
                    queue = [(r, c)]
                    visited.add((r, c))
                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        comp_pixels.add((curr_r, curr_c))
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < H and 0 <= nc < W and grid[nr, nc] == color and (nr, nc) not in visited:
                                visited.add((nr, nc))
                                queue.append((nr, nc))
                    components.append((comp_pixels, color))
        return components

    @staticmethod
    def map_by_cardinality(grid: Grid, bg_color: int = 0, target_color: int = 3, mapping_func: Callable = max) -> Grid:  # type: ignore[type-arg]
        """Map components [C] to target_color based on a cardinality function (e.g., max)."""
        components = QuotientGraph.get_components(grid, bg_color)
        if not components: return grid.copy()

        target_size = mapping_func(len(comp[0]) for comp in components)
        result = grid.copy()
        for comp_pixels, _color in components:
            if len(comp_pixels) == target_size:
                for r, c in comp_pixels:
                    result[r, c] = target_color
        return result

    @staticmethod
    def recolor_by_size_mapping(grid: Grid, size_color_map: dict[int, int], bg_color: int = 0) -> Grid:
        """Recolor each component based on a learned size→color mapping."""
        components = QuotientGraph.get_components(grid, bg_color)
        result = np.full_like(grid, bg_color)
        for comp_pixels, color in components:
            size = len(comp_pixels)
            new_color = size_color_map.get(size, color)
            for r, c in comp_pixels:
                result[r, c] = new_color
        return result


# --- 9. Géométrie Projective (Symétries locales) ---

class ProjectiveSpace:
    @staticmethod
    def mirror_overlay_half(grid: Grid, axis: str = 'v') -> Grid:
        """Reflect a subset of S across line L and Union it."""
        H, W = grid.shape
        result = grid.copy()
        if axis == 'v': # Vertical reflection (left to right usually, or max info side)
            half_W = W // 2
            left = grid[:, :half_W]
            right = grid[:, half_W + (W%2):]
            if np.count_nonzero(left) > np.count_nonzero(right):
                result[:, W-half_W:] = np.where(result[:, W-half_W:] == 0, np.fliplr(left), result[:, W-half_W:])
            else:
                result[:, :half_W] = np.where(result[:, :half_W] == 0, np.fliplr(right), result[:, :half_W])
        elif axis == 'h': # Horizontal reflection
            half_H = H // 2
            top = grid[:half_H, :]
            bottom = grid[half_H + (H%2):, :]
            if np.count_nonzero(top) > np.count_nonzero(bottom):
                result[H-half_H:, :] = np.where(result[H-half_H:, :] == 0, np.flipud(top), result[H-half_H:, :])
            else:
                result[:half_H, :] = np.where(result[:half_H, :] == 0, np.flipud(bottom), result[:half_H, :])
        return result


# --- 10. Analyse Harmonique (Fractales & Périodicités 2D) ---

class HarmonicSpace:
    @staticmethod
    def kronecker_fractal(grid: Grid, bg_color: int = 0) -> Grid:
        """M ⊗ M : Remplacer chaque pixel coloré par la matrice entière."""
        H, W = grid.shape
        result = np.full((H * H, W * W), bg_color, dtype=grid.dtype)
        for r in range(H):
            for c in range(W):
                if grid[r, c] != bg_color:
                    # Place a scaled version of the grid here
                    result[r*H:(r+1)*H, c*W:(c+1)*W] = grid
        return result

    @staticmethod
    def checkerboard(grid: Grid, color1: int = 1, color2: int = 2) -> Grid:
        """F(x,y) = (x+y) mod 2 sur les zones non nulles."""
        H, W = grid.shape
        result = grid.copy()
        for r in range(H):
            for c in range(W):
                if result[r, c] != 0:
                    result[r, c] = color1 if (r + c) % 2 == 0 else color2
        return result


# --- 11. Grammaires de Markov 1D (Raycasting formel) ---

class GenerativeGrammar:
    @staticmethod
    def raycast_orthogonal(grid: Grid, src_color: int, target_color: int, line_color: int) -> Grid:
        """Trace une ligne formelle de src_color jusqu'à target_color sur le même axe."""
        result = grid.copy()
        _H, _W = grid.shape
        src_coords = np.argwhere(grid == src_color)
        tgt_coords = np.argwhere(grid == target_color)
        if src_coords.size == 0 or tgt_coords.size == 0:
            return result

        for sr, sc in src_coords:
            for tr, tc in tgt_coords:
                if sr == tr: # Ligne horizontale
                    col_start, col_end = min(sc, tc), max(sc, tc)
                    result[sr, col_start+1:col_end] = line_color
                elif sc == tc: # Ligne verticale
                    row_start, row_end = min(sr, tr), max(sr, tr)
                    result[row_start+1:row_end, sc] = line_color
        return result


# --- 14. Quotient par Symétrie (Inpainting via action de groupe) ---

class SymmetryQuotient:
    """Reconstruct masked regions using the symmetry group of the grid.

    Given a grid with a rectangular region filled with sentinel color s,
    find the symmetry σ ∈ D4 such that σ maps the masked region to an
    unmasked region, and reconstruct: output[mask] = grid[σ(mask)].
    """

    @staticmethod
    def find_mask_rect(grid: Grid, sentinel: int = 8) -> tuple[int, int, int, int] | None:
        """Find the bounding box of the rectangular sentinel region."""
        mask = (grid == sentinel)
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        r_min, r_max = np.where(rows)[0][[0, -1]]
        c_min, c_max = np.where(cols)[0][[0, -1]]
        return int(r_min), int(r_max), int(c_min), int(c_max)

    @staticmethod
    def reconstruct_from_symmetry(grid: Grid, sentinel: int = 8) -> Grid | None:
        """Reconstruct masked region using the grid's own symmetry.

        Tries D4 symmetry elements to find one that maps
        unmasked pixels onto the masked region consistently.
        """
        rect = SymmetryQuotient.find_mask_rect(grid, sentinel)
        if rect is None:
            return None

        r_min, r_max, c_min, c_max = rect
        h, w = grid.shape
        result = grid.copy()

        # D4 symmetry candidates + rot_90/rot_270 (for square grids)
        symmetries = [
            ("flip_v", lambda r, c: (h - 1 - r, c)),
            ("flip_h", lambda r, c: (r, w - 1 - c)),
            ("rot_180", lambda r, c: (h - 1 - r, w - 1 - c)),
        ]
        if h == w:
            symmetries.extend([
                ("rot_90", lambda r, c: (c, h - 1 - r)),
                ("rot_270", lambda r, c: (w - 1 - c, r)),
                ("diag_main", lambda r, c: (c, r)),
                ("diag_anti", lambda r, c: (w - 1 - c, h - 1 - r)),
            ])

        for _name, sigma in symmetries:
            valid = True
            patch = result.copy()
            for r in range(r_min, r_max + 1):
                for c in range(c_min, c_max + 1):
                    if grid[r, c] == sentinel:
                        sr, sc = sigma(r, c)  # type: ignore[no-untyped-call]
                        if 0 <= sr < h and 0 <= sc < w and grid[sr, sc] != sentinel:
                            patch[r, c] = grid[sr, sc]
                        else:
                            valid = False
                            break
                if not valid:
                    break
            if valid:
                return patch

        # Fallback: translational reconstruction — find a matching region
        mh = r_max - r_min + 1
        mw = c_max - c_min + 1
        best_patch = None
        best_score = -1

        for sr in range(h - mh + 1):
            for sc in range(w - mw + 1):
                # Skip if this region overlaps the mask
                if sr <= r_max and sr + mh - 1 >= r_min and sc <= c_max and sc + mw - 1 >= c_min:
                    continue
                region = grid[sr:sr+mh, sc:sc+mw]
                if np.any(region == sentinel):
                    continue
                # Score by border context similarity
                score = 0
                for dr in [-1, mh]:
                    rr_mask = r_min + dr
                    rr_src = sr + dr
                    if 0 <= rr_mask < h and 0 <= rr_src < h:
                        score += int(np.sum(grid[rr_mask, c_min:c_max+1] == grid[rr_src, sc:sc+mw]))
                for dc in [-1, mw]:
                    cc_mask = c_min + dc
                    cc_src = sc + dc
                    if 0 <= cc_mask < w and 0 <= cc_src < w:
                        score += int(np.sum(grid[r_min:r_max+1, cc_mask] == grid[sr:sr+mh, cc_src]))
                if score > best_score:
                    best_score = score
                    best_patch = region.copy()

        if best_patch is not None and best_score > 0:
            result[r_min:r_max+1, c_min:c_max+1] = best_patch
            return result

        return None

    @staticmethod
    def extract_masked_content(grid: Grid, sentinel: int = 8) -> Grid | None:
        """Extract just the content that should fill the masked region.

        When output size != input size, the output IS the reconstruction patch.
        """
        reconstructed = SymmetryQuotient.reconstruct_from_symmetry(grid, sentinel)
        if reconstructed is None:
            return None
        rect = SymmetryQuotient.find_mask_rect(grid, sentinel)
        if rect is None:
            return None
        r_min, r_max, c_min, c_max = rect
        return reconstructed[r_min:r_max+1, c_min:c_max+1]


# --- 15. Réseau de Translation (Réparation Périodique) ---

class TranslationPeriod:
    """Repair patterns by projecting onto the orbit of a 2D translation lattice.

    The grid contains a repeating pattern with period (py, px).
    Defective pixels are those that break the periodicity.
    Repair: pixel[r, c] = canonical_tile[r % py, c % px].
    """

    @staticmethod
    def detect_period(grid: Grid) -> tuple[int, int] | None:
        """Find the smallest 2D period (py, px) of the grid."""
        h, w = grid.shape

        # Find smallest py such that grid[r] == grid[r + py] for most rows
        best_py, best_px = h, w

        for py in range(1, h):
            match_count = 0
            total = 0
            for r in range(h - py):
                for c in range(w):
                    total += 1
                    if grid[r, c] == grid[r + py, c]:
                        match_count += 1
            if total > 0 and match_count / total > 0.85:
                best_py = py
                break

        for px in range(1, w):
            match_count = 0
            total = 0
            for r in range(h):
                for c in range(w - px):
                    total += 1
                    if grid[r, c] == grid[r, c + px]:
                        match_count += 1
            if total > 0 and match_count / total > 0.85:
                best_px = px
                break

        if best_py == h and best_px == w:
            return None
        return best_py, best_px

    @staticmethod
    def build_canonical_tile(grid: Grid, py: int, px: int, exclude_sentinel: int = 8) -> Grid:
        """Build canonical tile by majority vote, excluding sentinel pixels."""
        tile = np.zeros((py, px), dtype=grid.dtype)
        h, w = grid.shape

        for tr in range(py):
            for tc in range(px):
                # Collect all instances of this tile position (exclude sentinel)
                votes: dict[int, int] = {}
                for r in range(tr, h, py):
                    for c in range(tc, w, px):
                        v = int(grid[r, c])
                        if v == exclude_sentinel:
                            continue  # Skip sentinel pixels
                        votes[v] = votes.get(v, 0) + 1
                # Majority vote (fallback to grid value if no valid votes)
                if votes:
                    tile[tr, tc] = max(votes, key=votes.get)  # type: ignore[arg-type]
                else:
                    tile[tr, tc] = grid[tr, tc]
        return tile

    @staticmethod
    def repair_periodic(grid: Grid) -> Grid | None:
        """Repair the grid by projecting onto the translation lattice orbit."""
        period = TranslationPeriod.detect_period(grid)
        if period is None:
            return None
        py, px = period
        if py < 2 and px < 2:
            return None

        tile = TranslationPeriod.build_canonical_tile(grid, py, px)
        h, w = grid.shape
        result = grid.copy()
        for r in range(h):
            for c in range(w):
                result[r, c] = tile[r % py, c % px]
        return result


# --- 16. Morphisme de Formes (Extraction via Invariants Topologiques) ---

class ShapeMorphism:
    """Map between quotient graphs preserving topological invariants.

    Extract objects (connected components), compute their invariants
    (bounding box, Betti numbers, color), and map them to outputs.
    """

    @staticmethod
    def extract_objects(grid: Grid, bg_color: int = 0) -> list[dict]:  # type: ignore[type-arg]
        """Extract all objects with their topological invariants."""
        from scipy.ndimage import label

        objects = []
        mask = (grid != bg_color)
        labeled, n = label(mask)

        for i in range(1, n + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) == 0:
                continue
            r_min, c_min = coords.min(axis=0)
            r_max, c_max = coords.max(axis=0)
            bbox = grid[r_min:r_max+1, c_min:c_max+1].copy()
            colors = set(grid[labeled == i].tolist())
            objects.append({
                "bbox": bbox,
                "r_min": int(r_min), "c_min": int(c_min),
                "r_max": int(r_max), "c_max": int(c_max),
                "size": len(coords),
                "colors": colors,
                "shape": (int(r_max - r_min + 1), int(c_max - c_min + 1)),
            })
        return objects

    @staticmethod
    def reconstruct_from_mask(grid: Grid, sentinel: int = 8) -> Grid | None:
        """Reconstruct masked region by finding its symmetric counterpart.

        For grids with approximate symmetry: find where the mask is,
        then find the corresponding unmasked region by checking all
        possible symmetric positions.
        """
        rect = SymmetryQuotient.find_mask_rect(grid, sentinel)
        if rect is None:
            return None

        r_min, r_max, c_min, c_max = rect
        mh = r_max - r_min + 1
        mw = c_max - c_min + 1
        h, w = grid.shape

        best_patch = None
        best_overlap = 0

        # Search for a region of same size with no sentinel pixels
        for sr in range(h - mh + 1):
            for sc in range(w - mw + 1):
                region = grid[sr:sr+mh, sc:sc+mw]
                if np.any(region == sentinel):
                    continue
                # Score: number of matching border pixels
                overlap = 0
                # Check if border context matches
                for dr in [-1, mh]:
                    r_check = r_min + dr
                    r_src = sr + dr
                    if 0 <= r_check < h and 0 <= r_src < h:
                        row_mask = grid[r_check, c_min:c_max+1]
                        row_src = grid[r_src, sc:sc+mw]
                        overlap += np.sum(row_mask == row_src)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_patch = region.copy()

        if best_patch is not None:
            result = grid.copy()
            result[r_min:r_max+1, c_min:c_max+1] = best_patch
            return result
        return None


#  INVARIANT DETECTION & FORMAL VERIFICATION
#  These functions detect invariant properties between Input and Output
#  to parameterize the correct mathematical laws.
# ═══════════════════════════════════════════════════════════════════════

def prove_correctness(transform_func: Callable[[Grid], Grid], inp: Grid, out: Grid) -> bool:
    """Formal verification: check if transform(inp) == out exactly."""
    try:
        pred = transform_func(inp)
        if pred is None or pred.shape != out.shape:
            return False
        return bool(np.array_equal(pred, out))
    except Exception:
        return False

def detect_dimension_invariant(inp: Grid, out: Grid) -> tuple[float, float]:
    """Calculate scale factor (sy, sx) from dimensions."""
    if inp.shape[0] == 0 or inp.shape[1] == 0:
        return 0.0, 0.0
    sy = out.shape[0] / inp.shape[0]
    sx = out.shape[1] / inp.shape[1]
    return sy, sx

def detect_color_mapping(inp: Grid, out: Grid) -> dict[int, int] | None:
    """Find the exact color mapping if one exists (must be consistent)."""
    if inp.shape != out.shape:
        return None
    mapping = {}  # type: ignore[var-annotated]
    for i_val, o_val in zip(inp.flatten(), out.flatten(), strict=False):
        if i_val in mapping and mapping[i_val] != o_val:
            return None # Contradiction in mapping
        mapping[i_val] = o_val
    return mapping


# ═══════════════════════════════════════════════════════════════════════
#  PROGRAM SYNTHESIS
#  Compose pure functions to build the exact deterministic program.
# ═══════════════════════════════════════════════════════════════════════

def synthesize_program(train_pairs: list[dict]) -> Callable[[Grid], Grid] | None:  # type: ignore[type-arg]
    """
    Synthesize a single mathematical function that correctly transforms
    ALL training inputs to outputs.
    """
    ref_inp = np.array(train_pairs[0]["input"])
    ref_out = np.array(train_pairs[0]["output"])

    sy, sx = detect_dimension_invariant(ref_inp, ref_out)
    same_shape = (sy == 1.0 and sx == 1.0)
    int_scale_up = (sy >= 1.0 and sx >= 1.0 and sy.is_integer() and sx.is_integer())
    int_scale_down = (0 < sy <= 1.0 and 0 < sx <= 1.0 and (1/sy).is_integer() and (1/sx).is_integer())

    # Evaluate Spaces

    # 1. Algebraic Group D4 (Isometric Transforms)
    if same_shape:
        for transform in DihedralGroup.all_elements():  # type: ignore[no-untyped-call]
            if all(prove_correctness(transform, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                return transform  # type: ignore[no-any-return]

    # 2. Color Field F10 Space (Bijective/Surjective color mapping)
    if same_shape:
        global_mapping = None
        valid_color_map = True
        for p in train_pairs:
            inp = np.array(p["input"])
            out = np.array(p["output"])
            mapping = detect_color_mapping(inp, out)
            if mapping is None:
                valid_color_map = False
                break
            if global_mapping is None:
                global_mapping = mapping
            else:
                for k, v in mapping.items():
                    if k in global_mapping and global_mapping[k] != v:
                        valid_color_map = False
                        break
                global_mapping.update(mapping)

        if valid_color_map and global_mapping is not None:
            return lambda g, m=global_mapping: ColorField.apply_bijection(g, m)  # type: ignore[misc]

    # 2.1 Boolean Lattice (Union of overlays)
    if same_shape:
        def overlay_h(g): return BooleanLattice.overlay(g, DihedralGroup.flip_h(g))  # type: ignore[no-untyped-def]
        def overlay_v(g): return BooleanLattice.overlay(g, DihedralGroup.flip_v(g))  # type: ignore[no-untyped-def]
        if all(prove_correctness(overlay_h, np.array(p["input"]), np.array(p["output"])) for p in train_pairs): return overlay_h
        if all(prove_correctness(overlay_v, np.array(p["input"]), np.array(p["output"])) for p in train_pairs): return overlay_v


    # 3. Product Space: D4 Group ⊗ Color Field F10
    if same_shape:
        for transform in DihedralGroup.all_elements():  # type: ignore[no-untyped-call]
            valid_composition = True
            global_mapping = None

            for p in train_pairs:
                inp = transform(np.array(p["input"]))
                out = np.array(p["output"])
                mapping = detect_color_mapping(inp, out)

                if mapping is None:
                    valid_composition = False
                    break

                if global_mapping is None:
                    global_mapping = mapping
                else:
                    for k, v in mapping.items():
                        if k in global_mapping and global_mapping[k] != v:
                            valid_composition = False
                            break
                    global_mapping.update(mapping)

            if valid_composition and global_mapping is not None:
                return lambda g, t=transform, m=global_mapping: ColorField.apply_bijection(t(g), m)  # type: ignore[misc]

    # 4. Vector Space Scaling & Tiling
    if int_scale_up and sy >= 1.0 and sx >= 1.0 and (sy > 1.0 or sx > 1.0):
        tsy, tsx = int(sy), int(sx)
        def scale_forward(g): return VectorSpace.scale(g, tsy, tsx)  # type: ignore[no-untyped-def]
        if all(prove_correctness(scale_forward, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return scale_forward

        def tile_forward(g): return VectorSpace.tile(g, tsy, tsx)  # type: ignore[no-untyped-def]
        if all(prove_correctness(tile_forward, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return tile_forward

    if int_scale_down and sy < 1.0 and sx < 1.0:
        def scale_backward(g): return VectorSpace.downscale(g, int(1/sy), int(1/sx))  # type: ignore[no-untyped-def]
        if all(prove_correctness(scale_backward, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return scale_backward

    # Exhaustive tile search for non-exact-ratio cases
    if not same_shape and ref_inp.shape[0] > 0 and ref_inp.shape[1] > 0:
        for ty in range(1, 6):
            for tx in range(1, 6):
                if ty == 1 and tx == 1: continue
                def tile_t(g, ty=ty, tx=tx): return VectorSpace.tile(g, ty, tx)  # type: ignore[no-untyped-def]
                if all(prove_correctness(tile_t, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                    return tile_t


    # 5. Geometrical & Topo Spaces

    # Pure Crop (Boolean Lattice Intersection)
    if not same_shape and sy <= 1.0 and sx <= 1.0:
        if all(prove_correctness(BooleanLattice.crop_nonzero, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return BooleanLattice.crop_nonzero

        # Product Space: Lattice ⊗ Target Color Field
        valid_composition = True
        global_mapping = None
        for p in train_pairs:
            inp = BooleanLattice.crop_nonzero(np.array(p["input"]))
            out = np.array(p["output"])
            mapping = detect_color_mapping(inp, out)
            if mapping is None:
                valid_composition = False
                break
            if global_mapping is None:
                global_mapping = mapping
            else:
                for k, v in mapping.items():
                    if k in global_mapping and global_mapping[k] != v:
                        valid_composition = False
                        break
                global_mapping.update(mapping)
        if valid_composition and global_mapping is not None:
             return lambda g, m=global_mapping: ColorField.apply_bijection(BooleanLattice.crop_nonzero(g), m)  # type: ignore[misc]

    # Particle Systems / Gravity Action
    if same_shape:
        if all(prove_correctness(ParticleSystem.gravity_down, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return ParticleSystem.gravity_down
        if all(prove_correctness(ParticleSystem.gravity_up, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return ParticleSystem.gravity_up

    # Pure Flood Fill (Topological Graph closure)
    if same_shape:
        def synthesize_flood_fill(train_pairs):  # type: ignore[no-untyped-def]
            for p in train_pairs:
                inp = np.array(p["input"])
                out = np.array(p["output"])

                diff = np.where(inp != out)
                if diff[0].size == 0:
                    continue

                src_color = inp[diff][0]
                dst_color = out[diff][0]

                start_x, start_y = diff[0][0], diff[1][0]
                pred = TopologicalGraph.flood_fill(inp, start_x, start_y, src_color, dst_color)
                if not np.array_equal(pred, out):
                    return
            return

        ff_prog = synthesize_flood_fill(train_pairs)  # type: ignore[no-untyped-call]
        if ff_prog: return ff_prog  # type: ignore[no-any-return]

        # Homology (Filling holes)
        colors_in_out = set(ref_out.flatten())
        bg_colors_in_inp = set(ref_inp.flatten())
        bg = 0
        for c in colors_in_out:
            if c == bg: continue
            def fill_h(g, c=c): return Homology.fill_enclosures(g, bg_color=bg, fill_color=c)  # type: ignore[no-untyped-def]
            if all(prove_correctness(fill_h, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                return fill_h

        # Morphological Algebra (Extract Outlines, Border Fills, Minkowski boundaries)
        if 0 in bg_colors_in_inp:
            # Inner outline (keep boundary pixels touching bg)
            if all(prove_correctness(MorphologicalSpace.extract_inner_outline, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                return MorphologicalSpace.extract_inner_outline
            # Outer outline with specific color
            for c in colors_in_out:
                if c == bg: continue
                def morph_outline(g, c=c): return MorphologicalSpace.extract_outline(g, bg_color=bg, outline_color=c)  # type: ignore[no-untyped-def]
                if all(prove_correctness(morph_outline, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                    return morph_outline
            # Border fill (Moore 8-connected dilation)
            for bc in colors_in_out:
                if bc == bg: continue
                def bfill(g, bc=bc): return MorphologicalSpace.border_fill(g, bg_color=bg, border_color=bc)  # type: ignore[no-untyped-def]
                if all(prove_correctness(bfill, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                    return bfill

        # Projective Space (Partial Mirror Overlays)
        def proj_mirror_v(g): return ProjectiveSpace.mirror_overlay_half(g, axis='v')  # type: ignore[no-untyped-def]
        if all(prove_correctness(proj_mirror_v, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return proj_mirror_v

        def proj_mirror_h(g): return ProjectiveSpace.mirror_overlay_half(g, axis='h')  # type: ignore[no-untyped-def]
        if all(prove_correctness(proj_mirror_h, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return proj_mirror_h

        # Quotient Graph (Component Cardinality filtering / Recolor by Size)
        for target_c in colors_in_out:
            if target_c == 0: continue
            # Check recoloring the largest component
            def recolor_max(g, c=target_c): return QuotientGraph.map_by_cardinality(g, target_color=c, mapping_func=max)  # type: ignore[no-untyped-def]
            if all(prove_correctness(recolor_max, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                return recolor_max
            # Check recoloring the smallest component
            def recolor_min(g, c=target_c): return QuotientGraph.map_by_cardinality(g, target_color=c, mapping_func=min)  # type: ignore[no-untyped-def]
            if all(prove_correctness(recolor_min, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                return recolor_min

        # Quotient Graph: Learned size→color mapping
        # Extract size→color from first training pair, verify on all
        ref_components = QuotientGraph.get_components(ref_inp)
        if len(ref_components) >= 2:
            size_color_map = {}  # type: ignore[var-annotated]
            valid_size_map = True
            for comp_pixels, _comp_color in ref_components:
                size = len(comp_pixels)
                # What color does this component have in the output?
                sample_r, sample_c = next(iter(comp_pixels))
                out_color = int(ref_out[sample_r, sample_c])
                if size in size_color_map and size_color_map[size] != out_color:
                    valid_size_map = False
                    break
                size_color_map[size] = out_color
            if valid_size_map and size_color_map:
                def recolor_by_map(g, m=size_color_map): return QuotientGraph.recolor_by_size_mapping(g, m)  # type: ignore[no-untyped-def]
                if all(prove_correctness(recolor_by_map, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                    return recolor_by_map

        # Affine modulo translations & Diagonals
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0: continue
                def translate(g, dy=dy, dx=dx): return AffineSpace.translate_modulo(g, dy, dx)  # type: ignore[no-untyped-def]
                if all(prove_correctness(translate, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                    return translate
        if all(prove_correctness(AffineSpace.reflect_main_diagonal, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return AffineSpace.reflect_main_diagonal
        if all(prove_correctness(AffineSpace.reflect_anti_diagonal, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return AffineSpace.reflect_anti_diagonal

        # Harmonic Space (Fractals, Checkerboards)
        if all(prove_correctness(HarmonicSpace.kronecker_fractal, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return HarmonicSpace.kronecker_fractal
        if len(colors_in_out) >= 2:
            c1, c2 = list(colors_in_out)[:2]
            def check_b(g, c1=c1, c2=c2): return HarmonicSpace.checkerboard(g, c1, c2)  # type: ignore[no-untyped-def]
            if all(prove_correctness(check_b, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                return check_b

        # Generative Grammars (Raycasting)
        if len(bg_colors_in_inp) >= 2 and len(colors_in_out) >= 1:
             src = next(iter(bg_colors_in_inp))
             tgt = list(bg_colors_in_inp)[1]
             line = next(iter(colors_in_out))
             def raycast_prog(g, s=src, t=tgt, l=line): return GenerativeGrammar.raycast_orthogonal(g, s, t, l)  # type: ignore[no-untyped-def]
             if all(prove_correctness(raycast_prog, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
                 return raycast_prog

        # Cellular Automata fixed points
        def ca_fixed_grow(g): return CellularAutomata.run_until_fixed_point(g, CellularAutomata.grow_Moore)  # type: ignore[no-untyped-def]
        if all(prove_correctness(ca_fixed_grow, np.array(p["input"]), np.array(p["output"])) for p in train_pairs):
            return ca_fixed_grow

    return None


# ═══════════════════════════════════════════════════════════════════════
#  SOLVER ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════

def solve_task_math(task_data: dict) -> dict:  # type: ignore[type-arg]
    """
    Solve an ARC task using pure deterministic mathematical laws.
    """
    train_pairs = task_data.get("train", [])
    test_pairs = task_data.get("test", [])

    if not train_pairs or not test_pairs:
        return {"solved": False, "strategy": "none", "predictions": [], "correct": []}

    # 1. Synthesize the deterministic mathematical program
    program = synthesize_program(train_pairs)

    # 2. Evaluate Program on test datasets
    if program is not None:
        predictions = []
        correct_flags = []

        for p in test_pairs:
            test_inp = np.array(p["input"])
            test_out = np.array(p.get("output", []))

            try:
                pred = program(test_inp)
                predictions.append(pred)
                if test_out.size > 0:
                    correct_flags.append(bool(np.array_equal(pred, test_out)))
                else:
                    correct_flags.append(False)
            except Exception:
                predictions.append(test_inp.copy())
                correct_flags.append(False)

        return {
            "solved": True,
            "strategy": "math_synthesis",
            "predictions": predictions,
            "correct": correct_flags,
        }

    return {"solved": False, "strategy": "none", "predictions": [], "correct": []}
