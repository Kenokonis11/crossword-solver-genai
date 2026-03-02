"""
PDF Grid Geometry Parser — Vector Drawing Analysis & Cell Mapping

This module handles the second stage of PDF crossword ingestion: analyzing
the vector drawings (lines, rectangles, fills) in a PDF to extract the
crossword grid's spatial structure.

It determines:
    - Grid dimensions (rows × columns)
    - Black/blocked square positions
    - Clue number placements (which cells have numbers)

This geometry data, combined with the clue text from PDFTextExtractor,
provides everything needed to build a fully validated CrosswordGraph.

Pipeline Position:
    PDF File (bytes/path)
        → PDFGridParser.extract_grid_geometry()   [vector drawings → grid structure]
        + PDFTextExtractor.extract_clues()         [text → clue dicts]
        → BaseCrosswordParser.build_graph()        [structure + clues → CrosswordGraph]

Design Note:
    This parser targets PDFs with vector-drawn crossword grids (lines and
    rectangles). Rasterized/image-only grids will require a separate
    image-based parser using computer vision techniques.
"""

from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF


class PDFGridParser:
    """
    Analyzes vector drawings and text coordinates in a crossword PDF
    to extract grid dimensions, black square positions, and clue number
    placements.
    """

    @staticmethod
    def _find_grid_bbox(drawings: List[Dict[str, Any]], page_rect: fitz.Rect) -> Optional[fitz.Rect]:
        """
        Identifies the crossword grid's bounding box from the page's vector drawings.

        Handles two PDF styles:
        1. Grids drawn with a single outer border rectangle (traditional).
        2. Grids composed of individual cell-sized rectangles (e.g., LA Times web print).

        Strategy:
        - First, try to find a large outer border (>1% of page area).
        - If that fails, detect a cluster of similarly-sized small rectangles
          (the cells themselves) and compute the bounding box from their union.

        Args:
            drawings: List of drawing dictionaries from page.get_drawings().
            page_rect: The page's total bounding rectangle.

        Returns:
            A fitz.Rect representing the grid's bounding box, or None if
            no suitable grid region was found.
        """
        if not drawings:
            return None

        page_area = page_rect.width * page_rect.height

        # --- Strategy 1: Look for a large outer border ---
        large_rects: List[fitz.Rect] = []
        for drawing in drawings:
            rect = fitz.Rect(drawing["rect"])
            area = rect.width * rect.height
            if area > page_area * 0.01 and area < page_area * 0.95:
                large_rects.append(rect)

        if large_rects:
            best_rect = max(large_rects, key=lambda r: r.width * r.height)
            grid_region = fitz.Rect(best_rect)
            for rect in large_rects:
                intersection = rect & grid_region
                if intersection.is_empty:
                    continue
                overlap_ratio = (intersection.width * intersection.height) / (rect.width * rect.height)
                if overlap_ratio > 0.3:
                    grid_region |= rect
            return grid_region

        # --- Strategy 2: Detect a cluster of cell-sized rectangles ---
        # Find the most common rectangle size (the canonical cell)
        from collections import Counter, defaultdict

        # Only consider filled rectangles (cells have fills; stray lines don't)
        filled_drawings = [d for d in drawings if d.get("fill") is not None]

        size_counts: Counter = Counter()
        for drawing in filled_drawings:
            rect = fitz.Rect(drawing["rect"])
            w_rounded = round(rect.width, 0)
            h_rounded = round(rect.height, 0)
            # Only consider small rectangles that could be cells (5-60pt)
            if 5 < w_rounded < 60 and 5 < h_rounded < 60:
                size_counts[(w_rounded, h_rounded)] += 1

        if not size_counts:
            return None

        # The most common small rectangle size is almost certainly the grid cell
        (canonical_w, canonical_h), count = size_counts.most_common(1)[0]

        # A crossword grid needs at least ~30 cells
        if count < 30:
            return None

        # Collect all rectangles matching the canonical cell size (within tolerance)
        cell_tolerance = 2.0
        candidate_rects: List[fitz.Rect] = []
        for drawing in filled_drawings:
            rect = fitz.Rect(drawing["rect"])
            if (abs(round(rect.width, 0) - canonical_w) <= cell_tolerance and
                    abs(round(rect.height, 0) - canonical_h) <= cell_tolerance):
                candidate_rects.append(rect)

        if not candidate_rects:
            return None

        # --- Grid-alignment filter ---
        # In a real crossword grid, each x0 position repeats at many y0 positions
        # (one rect per row) and vice versa. Stray decorative rects of the same size
        # only appear at 1-2 positions. Filter to x0/y0 that appear at ≥3 positions.
        x0_to_y0s: defaultdict = defaultdict(set)
        y0_to_x0s: defaultdict = defaultdict(set)
        for rect in candidate_rects:
            rx0 = round(rect.x0, 0)
            ry0 = round(rect.y0, 0)
            x0_to_y0s[rx0].add(ry0)
            y0_to_x0s[ry0].add(rx0)

        # Keep only x0/y0 positions that appear in at least 3 distinct rows/columns
        valid_x0s = {x for x, ys in x0_to_y0s.items() if len(ys) >= 3}
        valid_y0s = {y for y, xs in y0_to_x0s.items() if len(xs) >= 3}

        # Filter candidate rects to only grid-aligned ones
        cell_rects = [
            r for r in candidate_rects
            if round(r.x0, 0) in valid_x0s and round(r.y0, 0) in valid_y0s
        ]

        if not cell_rects:
            return None

        # Compute the bounding box as the union of all grid-aligned cell rectangles
        grid_region = fitz.Rect(cell_rects[0])
        for rect in cell_rects[1:]:
            grid_region |= rect

        return grid_region

    @staticmethod
    def _infer_dimensions(grid_bbox: fitz.Rect, drawings: List[Dict[str, Any]]) -> Tuple[int, int, float, float]:
        """
        Determines the number of rows and columns in the grid.

        Uses multiple strategies:
        1. Cell-cluster method: Find the most common small rectangle size
           and divide the grid bbox by it.
        2. Line-counting method: Count distinct horizontal/vertical grid lines.
        3. Fallback: Estimate from grid size using a standard cell size.

        Args:
            grid_bbox: The bounding box of the crossword grid.
            drawings: All drawings from the page.

        Returns:
            Tuple of (num_cols, num_rows, cell_width, cell_height).
        """
        from collections import Counter

        size_counts: Counter = Counter()
        for drawing in drawings:
            rect = fitz.Rect(drawing["rect"])
            if grid_bbox.intersects(rect):
                w_rounded = round(rect.width, 0)
                h_rounded = round(rect.height, 0)
                if 5 < w_rounded < 60 and 5 < h_rounded < 60:
                    size_counts[(w_rounded, h_rounded)] += 1

        if size_counts:
            (canonical_w, canonical_h), count = size_counts.most_common(1)[0]
            if count >= 30:
                num_cols = max(round(grid_bbox.width / canonical_w), 1)
                num_rows = max(round(grid_bbox.height / canonical_h), 1)
                cell_width = grid_bbox.width / num_cols
                cell_height = grid_bbox.height / num_rows
                return num_cols, num_rows, cell_width, cell_height

        # --- Fallback: Count grid lines ---
        snap_tolerance = 2.0
        x_positions: List[float] = []
        y_positions: List[float] = []

        for drawing in drawings:
            rect = fitz.Rect(drawing["rect"])
            if not grid_bbox.contains(rect) and not (grid_bbox & rect):
                continue
            if rect.height < snap_tolerance and rect.width > grid_bbox.width * 0.5:
                y_positions.append(rect.y0)
            if rect.width < snap_tolerance and rect.height > grid_bbox.height * 0.5:
                x_positions.append(rect.x0)

        x_unique = _snap_positions(x_positions, snap_tolerance)
        y_unique = _snap_positions(y_positions, snap_tolerance)

        num_cols = max(len(x_unique) - 1, 1)
        num_rows = max(len(y_unique) - 1, 1)

        if num_cols <= 1 or num_rows <= 1:
            estimated_cell_size = 24.0
            num_cols = max(round(grid_bbox.width / estimated_cell_size), 1)
            num_rows = max(round(grid_bbox.height / estimated_cell_size), 1)

        cell_width = grid_bbox.width / num_cols
        cell_height = grid_bbox.height / num_rows

        return num_cols, num_rows, cell_width, cell_height

    @staticmethod
    def _find_black_squares(
        grid_bbox: fitz.Rect,
        drawings: List[Dict[str, Any]],
        num_cols: int,
        num_rows: int,
        cell_width: float,
        cell_height: float,
    ) -> List[Tuple[int, int]]:
        """
        Identifies filled (black) squares using a cell-first approach.

        Iterates over every cell position, computes its mathematically
        perfect center point, and checks if any dark-filled drawing
        covers that point. This eliminates coordinate division errors.
        """
        # Pre-filter: collect only dark-filled, cell-sized drawings
        min_area = cell_width * cell_height * 0.3
        max_area = cell_width * cell_height * 1.5
        dark_rects: List[fitz.Rect] = []

        for drawing in drawings:
            fill = drawing.get("fill")
            if fill is None:
                continue
            if isinstance(fill, (list, tuple)) and len(fill) >= 3:
                if fill[0] + fill[1] + fill[2] > 1.5:
                    continue
            rect = fitz.Rect(drawing["rect"])
            area = rect.width * rect.height
            if area < min_area or area > max_area:
                continue
            if not grid_bbox.intersects(rect):
                continue
            dark_rects.append(rect)

        # Cell-first: probe each cell's center point
        black_squares: List[Tuple[int, int]] = []
        for row in range(num_rows):
            for col in range(num_cols):
                cx = grid_bbox.x0 + (col + 0.5) * cell_width
                cy = grid_bbox.y0 + (row + 0.5) * cell_height
                center = fitz.Point(cx, cy)
                for rect in dark_rects:
                    if rect.contains(center):
                        black_squares.append((col, row))
                        break

        return black_squares

    @staticmethod
    def _find_numbered_cells(
        page: fitz.Page,
        grid_bbox: fitz.Rect,
        num_cols: int,
        num_rows: int,
        cell_width: float,
        cell_height: float,
    ) -> Dict[int, Tuple[int, int]]:
        """
        Locates clue numbers using a cell-first approach.

        Pre-filters numeric text, then iterates every cell and checks
        if a number falls inside its top-left quadrant.
        """
        # Step 1: Pre-filter valid numeric text items in the grid
        number_items: List[Tuple[int, float, float]] = []
        words = page.get_text("words")

        for word_info in words:
            x0, y0, x1, y1 = word_info[0], word_info[1], word_info[2], word_info[3]
            raw_text = word_info[4].strip()
            digits_only = ''.join(c for c in raw_text if c.isdigit())
            if not digits_only:
                continue
            number = int(digits_only)
            if number > 200:
                continue
            if len(digits_only) < len(raw_text) * 0.4:
                continue
            if x0 < grid_bbox.x0 or y0 < grid_bbox.y0:
                continue
            if x1 > grid_bbox.x1 or y1 > grid_bbox.y1:
                continue
            number_items.append((number, x0, y0))

        # Step 2: Cell-first — for each cell, find any number in its top-left
        numbered_cells: Dict[int, Tuple[int, int]] = {}
        for row in range(num_rows):
            for col in range(num_cols):
                cell_left = grid_bbox.x0 + col * cell_width
                cell_top = grid_bbox.y0 + row * cell_height
                cell_right = cell_left + cell_width * 0.6
                cell_bottom = cell_top + cell_height * 0.6
                for number, nx, ny in number_items:
                    if cell_left <= nx < cell_right and cell_top <= ny < cell_bottom:
                        numbered_cells[number] = (col, row)
                        break

        return numbered_cells

    def extract_grid_geometry(self, pdf_path: str) -> Dict[str, Any]:
        """
        Master method: extracts the complete grid geometry from a crossword PDF.

        Pipeline:
        1. Opens the PDF and extracts all vector drawings.
        2. Identifies the grid's bounding box.
        3. Infers the grid dimensions (rows × columns).
        4. Locates all black/blocked squares (cell-first probe).
        5. Maps clue numbers to their cell coordinates (cell-first probe).
        """
        doc = fitz.open(pdf_path)
        page = doc[0]

        drawings = page.get_drawings()

        grid_bbox = self._find_grid_bbox(drawings, page.rect)
        if grid_bbox is None:
            doc.close()
            raise ValueError(
                f"No crossword grid detected in '{pdf_path}'. "
                "The PDF may use rasterized images instead of vector drawings."
            )

        num_cols, num_rows, cell_width, cell_height = self._infer_dimensions(grid_bbox, drawings)

        black_cells = self._find_black_squares(
            grid_bbox, drawings, num_cols, num_rows, cell_width, cell_height
        )

        numbered_cells = self._find_numbered_cells(
            page, grid_bbox, num_cols, num_rows, cell_width, cell_height
        )

        doc.close()

        return {
            "width": num_cols,
            "height": num_rows,
            "cell_width": cell_width,
            "cell_height": cell_height,
            "grid_bbox": tuple(grid_bbox),
            "black_cells": black_cells,
            "numbered_cells": numbered_cells,
        }


def _snap_positions(positions: List[float], tolerance: float) -> List[float]:
    """
    Deduplicates a list of coordinate positions by snapping nearby values together.

    Positions within `tolerance` of each other are merged into a single
    representative value (their average).

    Args:
        positions: Raw coordinate values (potentially with duplicates/noise).
        tolerance: Maximum distance between positions to consider them the same.

    Returns:
        Sorted list of unique, snapped positions.
    """
    if not positions:
        return []

    sorted_pos = sorted(positions)
    snapped: List[float] = []
    current_group: List[float] = [sorted_pos[0]]

    for pos in sorted_pos[1:]:
        if pos - current_group[-1] <= tolerance:
            # Close enough — add to the current group
            current_group.append(pos)
        else:
            # New distinct position — flush the current group
            snapped.append(sum(current_group) / len(current_group))
            current_group = [pos]

    # Flush the last group
    snapped.append(sum(current_group) / len(current_group))

    return snapped
