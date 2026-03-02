"""
PDF Text Extractor — Column-Aware Text Ripping & Regex Clue Stitching

This module handles the first stage of PDF crossword ingestion: extracting
raw text from a PDF file, preserving reading order across potentially
scrambled and multi-column text blocks, and stitching fragmented multi-line
clues back into single, clean strings.

The output is a list of structured clue dictionaries ready to be consumed
by a concrete BaseCrosswordParser subclass that will enrich them with
spatial data (lengths, cell coordinates) and produce a CrosswordGraph.

Pipeline Position:
    PDF File (bytes/path)
        → PDFTextExtractor._get_ordered_lines()  [raw text blocks → column-sorted lines]
        → PDFTextExtractor.extract_clues()        [sorted lines → clue dicts]
        → BaseCrosswordParser.build_graph()        [clue dicts → CrosswordGraph]

Column Handling:
    Many crossword PDFs lay out Across and Down clues in side-by-side columns.
    Naive y0-primary sorting would interleave lines from both columns. This
    extractor clusters text blocks into columns by x0 proximity (50px tolerance),
    then reads each column top-to-bottom before moving to the next column.
"""

import re
from typing import List, Dict, Any, Optional

import fitz  # PyMuPDF

# Blocks whose x0 values are within this many points of each other
# are considered part of the same column. 50pt ≈ 0.7 inches.
COLUMN_TOLERANCE_PX: float = 50.0


class PDFTextExtractor:
    """
    Extracts and structures crossword clue text from PDF documents.

    Uses PyMuPDF's block-level extraction with column-aware clustering to
    preserve spatial reading order, then applies a regex-driven state machine
    to separate Across/Down sections and stitch fragmented multi-line clues
    into complete strings.
    """

    @staticmethod
    def _cluster_into_columns(text_blocks: List[tuple], tolerance: float = COLUMN_TOLERANCE_PX) -> List[List[tuple]]:
        """
        Groups text blocks into columns based on the proximity of their x0 coordinates.

        Blocks are sorted by x0, then scanned sequentially. If a block's x0 is within
        `tolerance` pixels of the current column's representative x0, it joins that column.
        Otherwise, a new column is started.

        Args:
            text_blocks: Raw text block tuples from PyMuPDF (x0, y0, x1, y1, text, ...).
            tolerance: Maximum x0 difference (in points) for blocks to be in the same column.

        Returns:
            A list of column groups, each being a list of text block tuples.
            Columns are ordered left-to-right; blocks within each column are unsorted.
        """
        if not text_blocks:
            return []

        # Sort blocks by x0 so we can cluster sequentially
        sorted_by_x = sorted(text_blocks, key=lambda b: b[0])

        columns: List[List[tuple]] = []
        current_column: List[tuple] = [sorted_by_x[0]]
        # Track the average x0 of the current column for comparison
        current_column_x0: float = sorted_by_x[0][0]

        for block in sorted_by_x[1:]:
            block_x0 = block[0]

            if abs(block_x0 - current_column_x0) <= tolerance:
                # Block belongs to the current column
                current_column.append(block)
                # Update the running average x0 for better clustering
                current_column_x0 = sum(b[0] for b in current_column) / len(current_column)
            else:
                # Start a new column
                columns.append(current_column)
                current_column = [block]
                current_column_x0 = block_x0

        # Don't forget the last column
        columns.append(current_column)

        return columns

    def _get_ordered_lines(self, pdf_path: str) -> List[str]:
        """
        Extracts all text from a PDF, sorted in column-aware reading order.

        Instead of naive y0/x0 sorting (which interleaves side-by-side columns),
        this method:
        1. Extracts text blocks with bounding box coordinates.
        2. Clusters blocks into columns by x0 proximity.
        3. Sorts columns left-to-right.
        4. Sorts blocks within each column top-to-bottom (by y0).
        5. Flattens the result into a single list of lines.

        This ensures that all Across clues (typically in the left column) are
        read before all Down clues (right column).

        Args:
            pdf_path: Filesystem path to the PDF file.

        Returns:
            A flat list of stripped, non-empty text lines in column-aware reading order.
        """
        doc = fitz.open(pdf_path)
        all_lines: List[str] = []

        for page in doc:
            # Extract text blocks: each block is a tuple of
            # (x0, y0, x1, y1, "text", block_no, block_type)
            # block_type 0 = text, 1 = image
            blocks = page.get_text("blocks")

            # Filter to text blocks only (block_type == 0)
            text_blocks = [b for b in blocks if b[6] == 0]

            # Cluster blocks into columns by x0 proximity
            columns = self._cluster_into_columns(text_blocks)

            # Process each column left-to-right (columns are already in x0 order)
            for column in columns:
                # Sort blocks within this column top-to-bottom by y0
                column.sort(key=lambda b: b[1])

                # Split each block's text into individual lines
                for block in column:
                    raw_text = block[4]  # The text content of this block
                    for line in raw_text.split("\n"):
                        stripped = line.strip()
                        if stripped:  # Skip empty lines
                            all_lines.append(stripped)

        doc.close()
        return all_lines

    def extract_clues(self, text_lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parses ordered text lines into structured clue dictionaries using
        a regex-driven state machine.

        Handles three types of lines:
        1. Section headers ("ACROSS" / "DOWN") — update the current direction.
        2. Clue starts (e.g., "13 Emma of 'Madame") — begin a new clue entry.
        3. Continuation lines (e.g., 'Web"') — append to the active clue's text.

        Lines appearing before the first ACROSS/DOWN header are treated as
        garbage (titles, copyright, author names) and are silently discarded.

        Args:
            text_lines: Pre-sorted, non-empty text lines from _get_ordered_lines().

        Returns:
            A list of clue dictionaries, each containing:
                - "number" (int): The clue number.
                - "direction" (str): "Across" or "Down".
                - "text" (str): The full, stitched clue text.
        """
        # Regex to detect the start of a new clue line:
        # One or more digits at the start, followed by whitespace, then the clue text.
        # Examples: "1 Capital of France", "13 Emma of \"Madame", "100 Type of fish"
        clue_start_pattern = re.compile(r"^(\d+)\s+(.*)")

        # Regex to detect section headers (case-insensitive, with optional whitespace)
        header_pattern = re.compile(r"^\s*(ACROSS|DOWN)\s*$", re.IGNORECASE)

        clues: List[Dict[str, Any]] = []
        current_direction: Optional[str] = None
        current_clue: Optional[Dict[str, Any]] = None

        for line in text_lines:
            # --- Check for section header ---
            header_match = header_pattern.match(line)
            if header_match:
                # Flush the active clue before switching direction
                if current_clue is not None:
                    current_clue["text"] = current_clue["text"].strip()
                    clues.append(current_clue)
                    current_clue = None

                # Normalize to title case: "ACROSS" → "Across", "DOWN" → "Down"
                current_direction = header_match.group(1).capitalize()
                continue

            # --- Garbage collection: skip lines before first header ---
            if current_direction is None:
                continue

            # --- Invisible character filter ---
            # Tribune PDFs embed zero-width joiners (\u200d) and non-breaking
            # spaces in grid number sequences. Strip them before processing.
            line = line.replace('\u200d', '').replace('\u00a0', ' ').strip()

            # Skip lines that are purely digits (grid number artifacts)
            stripped_digits = re.sub(r'\D', '', line)
            if stripped_digits and stripped_digits == line.replace(' ', ''):
                continue

            # --- Check for clue start ---
            clue_match = clue_start_pattern.match(line)
            if clue_match:
                # Flush the previous clue before starting a new one
                if current_clue is not None:
                    current_clue["text"] = current_clue["text"].strip()
                    clues.append(current_clue)

                # Start a new clue entry
                current_clue = {
                    "number": int(clue_match.group(1)),
                    "direction": current_direction,
                    "text": clue_match.group(2).strip(),
                }
                continue

            # --- Continuation line (fragmented clue text) ---
            if current_clue is not None:
                stripped = line.strip()
                # Guard: Skip pure-digit fragments (grid cell numbers 1-127)
                if stripped.isdigit():
                    continue
                # Guard: Skip URL fragments (page footer artifacts)
                stripped_lower = stripped.lower()
                if 'http' in stripped_lower or 'www' in stripped_lower or 'amuselabs' in stripped_lower:
                    continue
                # Stitch the fragment onto the active clue with a space separator
                current_clue["text"] += " " + stripped

        # Flush the final clue (the loop ends without appending the last one)
        if current_clue is not None:
            current_clue["text"] = current_clue["text"].strip()
            clues.append(current_clue)

        # --- Post-processing: strip trailing grid numbers from all clues ---
        # Tribune PDFs can append "1 2 3 4 ... 127" to the end of clue text
        trailing_numbers_re = re.compile(r'([\d\u200d\s]{10,})$')
        for clue in clues:
            clue["text"] = trailing_numbers_re.sub('', clue["text"]).strip()

        return clues
