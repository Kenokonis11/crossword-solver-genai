"""
PDF Crossword Parser — The Graph Stitcher

This module is the master orchestrator for PDF crossword ingestion. It
combines the text extraction pipeline (PDFTextExtractor) with the spatial
grid analysis (PDFGridParser) to produce a fully validated CrosswordGraph.

The critical computation this module performs is the "spatial stitching":
    1. For each extracted clue, look up its starting cell from the grid geometry.
    2. Walk across (or down) from that cell until hitting a black square or grid edge.
    3. The walk length = word length. The walked cells = cell coordinate list.
    4. Cross-reference all Across and Down cell lists to find intersections.

This is what gives the LLM spatial awareness it cannot derive from raw text:
    "1-Across is 5 letters long, and its 3rd letter intersects with 4-Down's 1st letter."

Pipeline Position:
    PDF File (path)
        → PDFTextExtractor  [raw text → clue dicts]
        → PDFGridParser      [vector drawings → grid geometry]
        → PDFCrosswordParser [clue dicts + geometry → CrosswordGraph]
"""

import logging
from typing import Any, Dict, List, Set, Tuple

from parsers.base_parser import BaseCrosswordParser
from parsers.pdf_text_extractor import PDFTextExtractor
from parsers.pdf_grid_parser import PDFGridParser
from crossword_schema import Clue, Coordinate, CrosswordGraph, Intersection

logger = logging.getLogger(__name__)


class PDFCrosswordParser(BaseCrosswordParser):
    """
    Concrete parser for vector-drawn crossword PDFs.

    Inherits from BaseCrosswordParser and implements the full 4-stage pipeline:
        1. extract_raw_text()        → Ordered text lines from the PDF
        2. parse_clues()             → Structured clue dictionaries
        3. parse_grid_geometry()     → Grid dimensions, black squares, numbered cells
        4. build_graph()             → Validated CrosswordGraph with lengths & intersections

    Usage:
        parser = PDFCrosswordParser(source="path/to/crossword.pdf")
        graph = parser.build_graph()
    """

    def __init__(self, source: Any) -> None:
        """
        Initialize the PDF parser with a filesystem path to a crossword PDF.

        Args:
            source: Filesystem path (str) to the PDF file.
        """
        super().__init__(source)
        self._text_extractor = PDFTextExtractor()
        self._grid_parser = PDFGridParser()

        # Cache parsed results so stages don't re-execute
        self._raw_lines: List[str] = []
        self._clues: List[Dict[str, Any]] = []
        self._grid_geo: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Stage 1: Text Extraction
    # ------------------------------------------------------------------
    def extract_raw_text(self) -> str:
        """
        Extracts column-aware ordered text lines from the PDF.

        Returns:
            A single string of all extracted lines joined by newlines.
            The raw lines are also cached internally for parse_clues().
        """
        self._raw_lines = self._text_extractor._get_ordered_lines(self.source)
        return "\n".join(self._raw_lines)

    # ------------------------------------------------------------------
    # Stage 2: Clue Parsing
    # ------------------------------------------------------------------
    def parse_clues(self) -> List[Dict[str, Any]]:
        """
        Parses the extracted text into structured clue dictionaries.

        Must be called after extract_raw_text() or will extract automatically.

        Returns:
            List of clue dicts with keys: "number", "direction", "text".
        """
        if not self._raw_lines:
            self.extract_raw_text()

        self._clues = self._text_extractor.extract_clues(self._raw_lines)
        return self._clues

    # ------------------------------------------------------------------
    # Stage 3: Grid Geometry
    # ------------------------------------------------------------------
    def parse_grid_geometry(self) -> Dict[str, Any]:
        """
        Extracts grid dimensions, black squares, and numbered cells.

        Returns:
            Dictionary with keys: "width", "height", "cell_width", "cell_height",
            "grid_bbox", "black_cells", "numbered_cells".
        """
        self._grid_geo = self._grid_parser.extract_grid_geometry(self.source)
        return self._grid_geo

    # ------------------------------------------------------------------
    # Spatial Stitching: Word Lengths & Cell Coordinates
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_clue_spatial_data(
        clues: List[Dict[str, Any]],
        grid_geo: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Calculates word lengths and cell coordinate lists for each clue by
        walking from its starting cell until hitting a black square or grid edge.

        For Across clues: walk right (incrementing col).
        For Down clues:   walk down (incrementing row).

        Args:
            clues: List of clue dicts (must have "number", "direction", "text").
            grid_geo: Grid geometry dict from PDFGridParser.

        Returns:
            The same clue list, enriched with "length" (int) and "cells" (List[Coordinate]).
            Clues whose number doesn't appear in the grid geometry are skipped with a warning.
        """
        width: int = grid_geo["width"]
        height: int = grid_geo["height"]
        numbered_cells: Dict[int, Tuple[int, int]] = grid_geo["numbered_cells"]

        # Build a fast-lookup set of black square coordinates
        black_set: Set[Tuple[int, int]] = set(
            (col, row) for col, row in grid_geo["black_cells"]
        )

        enriched_clues: List[Dict[str, Any]] = []

        for clue in clues:
            clue_number: int = clue["number"]
            direction: str = clue["direction"]

            # --- Validate: does this clue number exist in the grid? ---
            if clue_number not in numbered_cells:
                logger.warning(
                    "Clue %d-%s has no matching numbered cell in the grid geometry. "
                    "Skipping this clue.",
                    clue_number,
                    direction,
                )
                continue

            start_col, start_row = numbered_cells[clue_number]
            cells: List[Coordinate] = []

            if direction == "Across":
                # Walk right from the starting cell
                col = start_col
                while col < width and (col, start_row) not in black_set:
                    cells.append(Coordinate(col=col, row=start_row))
                    col += 1

            elif direction == "Down":
                # Walk down from the starting cell
                row = start_row
                while row < height and (start_col, row) not in black_set:
                    cells.append(Coordinate(col=start_col, row=row))
                    row += 1

            else:
                logger.warning(
                    "Clue %d has unknown direction '%s'. Skipping.",
                    clue_number,
                    direction,
                )
                continue

            # Enrich the clue with spatial data
            enriched = dict(clue)  # Shallow copy to avoid mutating the original
            enriched["length"] = len(cells)
            enriched["cells"] = cells

            if len(cells) == 0:
                logger.warning(
                    "Clue %d-%s starting at (%d, %d) computed 0 cells. "
                    "The starting cell may be a black square.",
                    clue_number,
                    direction,
                    start_col,
                    start_row,
                )
                continue

            enriched_clues.append(enriched)

        return enriched_clues

    # ------------------------------------------------------------------
    # Intersection Mapping
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_intersections(populated_clues: List[Dict[str, Any]]) -> List[Intersection]:
        """
        Finds all points where an Across clue and a Down clue share a cell.

        For every Across clue, checks if any of its cells appear in any Down
        clue's cell list. When a shared cell is found, records the character
        indices in both words.

        Args:
            populated_clues: Clue dicts that have been enriched with "cells" lists.

        Returns:
            List of Intersection objects mapping shared cells between Across/Down clues.
        """
        # Separate clues by direction
        across_clues = [c for c in populated_clues if c["direction"] == "Across"]
        down_clues = [c for c in populated_clues if c["direction"] == "Down"]

        # Build a fast lookup: (col, row) → (clue_number, index_in_word) for all Down clues
        down_cell_map: Dict[Tuple[int, int], Tuple[int, int]] = {}
        for down_clue in down_clues:
            for idx, cell in enumerate(down_clue["cells"]):
                down_cell_map[(cell.col, cell.row)] = (down_clue["number"], idx)

        intersections: List[Intersection] = []

        for across_clue in across_clues:
            for across_idx, cell in enumerate(across_clue["cells"]):
                coord_key = (cell.col, cell.row)

                if coord_key in down_cell_map:
                    down_number, down_idx = down_cell_map[coord_key]

                    intersections.append(
                        Intersection(
                            across_clue_number=across_clue["number"],
                            down_clue_number=down_number,
                            across_index=across_idx,
                            down_index=down_idx,
                        )
                    )

        return intersections

    # ------------------------------------------------------------------
    # Stage 4: The Master Orchestrator
    # ------------------------------------------------------------------
    def build_graph(self) -> CrosswordGraph:
        """
        Orchestrates the full PDF-to-CrosswordGraph pipeline.

        Executes:
            1. Text extraction → ordered lines
            2. Clue parsing    → clue dicts with text
            3. Grid geometry   → dimensions, black squares, numbered cells
            4. Spatial stitch  → word lengths and cell coordinates
            5. Intersection    → crossing point mappings
            6. Validation      → Pydantic-validated CrosswordGraph

        Returns:
            A fully validated CrosswordGraph ready for the LLM and CSP engine.

        Raises:
            ValueError: If grid detection fails or Pydantic validation fails.
        """
        logger.info("Starting PDF crossword parsing for: %s", self.source)

        # Stage 1 & 2: Extract and parse clue text
        raw_clues = self.parse_clues()
        logger.info("Extracted %d clues from text.", len(raw_clues))

        # Stage 3: Extract grid geometry
        grid_geo = self.parse_grid_geometry()
        logger.info(
            "Grid geometry: %dx%d, %d black cells, %d numbered cells.",
            grid_geo["width"],
            grid_geo["height"],
            len(grid_geo["black_cells"]),
            len(grid_geo["numbered_cells"]),
        )

        # Stage 4a: Compute spatial data (word lengths + cell coordinates)
        enriched_clues = self._compute_clue_spatial_data(raw_clues, grid_geo)
        logger.info(
            "Spatially enriched %d of %d clues.", len(enriched_clues), len(raw_clues)
        )

        # Stage 4b: Compute intersection mappings
        intersections = self._compute_intersections(enriched_clues)
        logger.info("Found %d intersection points.", len(intersections))

        # Stage 5: Build the validated Pydantic model
        pydantic_clues = [
            Clue(
                number=c["number"],
                direction=c["direction"],
                text=c["text"],
                length=c["length"],
                cells=c["cells"],
            )
            for c in enriched_clues
        ]

        graph = CrosswordGraph(
            title="PDF Crossword Puzzle",
            width=grid_geo["width"],
            height=grid_geo["height"],
            clues=pydantic_clues,
            intersections=intersections,
        )

        logger.info(
            "CrosswordGraph built successfully: %d clues, %d intersections.",
            len(graph.clues),
            len(graph.intersections),
        )

        return graph
