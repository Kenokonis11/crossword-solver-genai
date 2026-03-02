"""
Crossword Schema — Universal Constraint Graph Models

This module defines the canonical data models for representing a crossword puzzle
as a structured constraint graph. It serves as the universal translation layer
between raw visual inputs (PDFs, images, web scrapes) and the LLM reasoning engine.

The schema encodes spatial geometry, clue metadata, word lengths, cell coordinates,
and intersection mappings — the exact constraints that an LLM cannot infer from
unstructured text alone. By converting any input source into a CrosswordGraph,
we give the AI access to mathematically precise constraint information that enables
accurate candidate selection, ambiguity detection, and cross-reference resolution.

Architecture:
    Raw Input (PDF/Image/Scrape)
        → BaseCrosswordParser.build_graph()
            → CrosswordGraph (this schema)
                → LLM Reasoning Engine (llm_agent.py)
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class Coordinate(BaseModel):
    """
    Represents a single cell position on the crossword grid.

    Uses (col, row) convention where (0, 0) is the top-left corner.
    Col increases left-to-right, row increases top-to-bottom.
    """

    col: int = Field(..., ge=0, description="Zero-indexed column position (x-axis, left to right)")
    row: int = Field(..., ge=0, description="Zero-indexed row position (y-axis, top to bottom)")

    def __hash__(self) -> int:
        return hash((self.col, self.row))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Coordinate):
            return NotImplemented
        return self.col == other.col and self.row == other.row


class GridCell(BaseModel):
    """
    Represents a single cell in the crossword grid.

    A cell is either a black (blocked) square or a letter square.
    If a clue starts at this cell, `number` contains the clue number
    printed in the top-left corner of the cell.
    """

    position: Coordinate = Field(..., description="The (col, row) position of this cell on the grid")
    is_black: bool = Field(default=False, description="True if this is a blocked/black square")
    number: Optional[int] = Field(
        default=None,
        ge=1,
        description="The clue number printed in this cell, if any. Only cells where a clue starts have a number.",
    )


class Clue(BaseModel):
    """
    Represents a single crossword clue with its full spatial constraint data.

    This is the core model that bridges the gap between raw clue text and
    the spatial grid. The `length` and `cells` fields are what make constraint
    synthesis possible — they encode the exact letter count and the physical
    grid positions that each letter occupies, enabling intersection detection.
    """

    number: int = Field(..., ge=1, description="The clue number as printed on the grid")
    direction: Literal["Across", "Down"] = Field(
        ..., description="Whether this clue reads left-to-right (Across) or top-to-bottom (Down)"
    )
    text: str = Field(..., min_length=1, description="The raw clue text as written by the puzzle constructor")
    length: int = Field(
        ...,
        ge=1,
        description="The exact number of letters in the answer. This is the primary mathematical constraint.",
    )
    cells: List[Coordinate] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of grid coordinates this answer occupies, from first letter to last. "
            "len(cells) MUST equal length. This enables intersection computation."
        ),
    )

    @model_validator(mode="after")
    def validate_cells_match_length(self) -> Clue:
        """Enforce that the number of cells exactly matches the declared word length."""
        if len(self.cells) != self.length:
            raise ValueError(
                f"Clue {self.number}-{self.direction}: cells count ({len(self.cells)}) "
                f"does not match declared length ({self.length})"
            )
        return self


class Intersection(BaseModel):
    """
    Represents the point where an Across clue and a Down clue share a cell.

    Intersections are the backbone of crossword constraint propagation.
    When the solver locks in a letter for one clue, the intersection mapping
    tells us exactly which position in the crossing clue is now constrained.

    Example:
        If 1-Across and 2-Down share a cell, and the shared cell is the 3rd
        letter of 1-Across (across_index=2) and the 1st letter of 2-Down
        (down_index=0), then solving 1-Across constrains the first letter
        of 2-Down.
    """

    across_clue_number: int = Field(..., ge=1, description="The clue number of the Across word")
    down_clue_number: int = Field(..., ge=1, description="The clue number of the Down word")
    across_index: int = Field(
        ...,
        ge=0,
        description="Zero-indexed character position within the Across word where the intersection occurs",
    )
    down_index: int = Field(
        ...,
        ge=0,
        description="Zero-indexed character position within the Down word where the intersection occurs",
    )


class CrosswordGraph(BaseModel):
    """
    The master constraint graph representing a complete crossword puzzle.

    This is the universal output format that all parsers must produce.
    It contains everything the LLM and CSP engine need to reason about
    the puzzle: the grid dimensions, every clue with its spatial mapping,
    and every intersection point for constraint propagation.

    The CrosswordGraph is immutable once constructed — it represents the
    puzzle's ground truth geometry that does not change during solving.
    """

    title: str = Field(default="Untitled Puzzle", description="The puzzle title, if available")
    width: int = Field(..., ge=1, description="Number of columns in the grid")
    height: int = Field(..., ge=1, description="Number of rows in the grid")
    clues: List[Clue] = Field(
        ...,
        min_length=1,
        description="All clues in the puzzle, both Across and Down, with full spatial data",
    )
    intersections: List[Intersection] = Field(
        default_factory=list,
        description="All intersection points where Across and Down clues share a cell",
    )

    @model_validator(mode="after")
    def validate_clue_coordinates_within_grid(self) -> CrosswordGraph:
        """Ensure all clue cell coordinates fall within the declared grid dimensions."""
        for clue in self.clues:
            for cell in clue.cells:
                if cell.col >= self.width or cell.row >= self.height:
                    raise ValueError(
                        f"Clue {clue.number}-{clue.direction}: cell ({cell.col}, {cell.row}) "
                        f"is outside grid bounds ({self.width}x{self.height})"
                    )
        return self
