"""
Base Crossword Parser — Abstract Interface for All Input Sources

This module defines the abstract base class that all crossword parsers must implement.
Whether the input is a PDF, a photograph, a .puz file, or a web scrape, every parser
must be capable of producing a canonical CrosswordGraph.

The parser pipeline follows a strict 4-stage process:

    1. extract_raw_text()     → Pull raw text/OCR from the source
    2. parse_clues()          → Structure the raw text into clue dictionaries
    3. parse_grid_geometry()  → Extract the grid dimensions, black squares, and numbering
    4. build_graph()          → Synthesize stages 1-3 into a validated CrosswordGraph

This architecture ensures that the LLM reasoning engine (llm_agent.py) never touches
raw, unstructured input. It always receives a mathematically precise constraint graph
with validated word lengths, cell coordinates, and intersection mappings.

Design Rationale:
    - The LLM cannot count letters, detect grid geometry, or infer intersections from
      raw text. These are computational tasks that must be handled programmatically.
    - By mandating a universal output schema (CrosswordGraph), we decouple the parsing
      logic from the reasoning logic, allowing us to add new input formats without
      modifying the AI tutor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from crossword_schema import CrosswordGraph


class BaseCrosswordParser(ABC):
    """
    Abstract base class for all crossword input parsers.

    Every parser — regardless of input format — must implement these four
    methods to produce a validated CrosswordGraph. The methods are designed
    to be called sequentially, with each stage building on the previous one.

    Subclasses:
        - PuzFileParser: Parses .puz binary files (e.g., NYT, LA Times downloads)
        - PDFParser: Extracts crosswords from PDF documents via OCR + layout analysis
        - ImageParser: Processes photographs of crossword grids via computer vision
        - WebScrapeParser: Scrapes crossword data from online puzzle sources

    Usage:
        parser = PuzFileParser(source="path/to/puzzle.puz")
        graph = parser.build_graph()
        # graph is now a validated CrosswordGraph ready for the LLM
    """

    def __init__(self, source: Any) -> None:
        """
        Initialize the parser with a source input.

        Args:
            source: The raw input source. Type varies by parser implementation:
                    - str path for file-based parsers (PDF, .puz)
                    - bytes for in-memory file data
                    - str URL for web scrape parsers
                    - PIL.Image for image-based parsers
        """
        self.source: Any = source

    @abstractmethod
    def extract_raw_text(self) -> str:
        """
        Stage 1: Extract raw, unstructured text from the input source.

        For file-based parsers, this reads and decodes the file contents.
        For image parsers, this runs OCR to produce text.
        For web scrape parsers, this fetches and strips HTML.

        Returns:
            The raw text content extracted from the source, before any
            structural parsing has been applied.
        """
        ...

    @abstractmethod
    def parse_clues(self) -> List[Dict[str, Any]]:
        """
        Stage 2: Parse the raw text into structured clue dictionaries.

        Each dictionary must contain at minimum:
            - "number": int — the clue number
            - "direction": str — "Across" or "Down"
            - "text": str — the clue text
            - "length": int — the answer length (from grid geometry)

        Returns:
            A list of clue dictionaries, ordered by direction then number.
        """
        ...

    @abstractmethod
    def parse_grid_geometry(self) -> Dict[str, Any]:
        """
        Stage 3: Extract the spatial grid structure from the source.

        Must determine:
            - Grid dimensions (width x height)
            - Black/blocked square positions
            - Clue numbering (which cells have numbers)
            - Cell coordinate mappings for each clue

        Returns:
            A dictionary containing grid metadata:
                - "width": int
                - "height": int
                - "black_cells": List of (col, row) tuples
                - "numbered_cells": Dict mapping cell positions to clue numbers
                - "clue_cells": Dict mapping (number, direction) to ordered cell lists
        """
        ...

    @abstractmethod
    def build_graph(self) -> CrosswordGraph:
        """
        Stage 4: Synthesize all parsed data into a validated CrosswordGraph.

        This is the final output method. It calls the previous three stages
        (or uses cached results), computes all intersection mappings, and
        returns a fully validated CrosswordGraph instance.

        The returned graph is the canonical representation that the LLM
        reasoning engine and CSP solver will use. It must pass all Pydantic
        validators (cell counts match lengths, coordinates within bounds, etc.).

        Returns:
            A validated CrosswordGraph instance ready for the LLM and CSP engine.

        Raises:
            ValueError: If the parsed data fails validation (mismatched lengths,
                       out-of-bounds coordinates, etc.)
        """
        ...
