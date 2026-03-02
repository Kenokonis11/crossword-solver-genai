"""
Autonomous Crossword Solver — Deterministic Pre-Solve Pipeline

This module separates the "Solver" from the "Tutor" in the AI Crossword
system architecture. Instead of asking the LLM to solve clues live during
conversation (which leads to hallucinations and incorrect answers), this
solver runs BEFORE the chat session begins, locking in Ground Truth answers
that the Tutor Agent can then hint at with 100% confidence.

Architecture:
    PDF Upload
        → PDFCrosswordParser → CrosswordGraph
        → AutonomousSolver   → CrosswordGraph + solved answers
        → CrosswordTutorAgent (receives Ground Truth, only hints, never guesses)

Solve Phases:
    Phase 1 — Absolute Certainty Pass:
        Solve unambiguous clues (no puns, no fill-in-blanks) where exactly
        one candidate matches the clue meaning AND the exact letter count.

    Phase 2 — Constraint Propagation (future):
        Use locked-in letters to narrow down ambiguous clues via crossing
        letter constraints (e.g., if 1-Across = SCRIPT, then 1-Down starts
        with 'S').

    Phase 3 — Intersecting Pass (future):
        Re-attempt skipped clues using the accumulated crossing letters
        to resolve ambiguity (e.g., __ Goose vodka + pattern G_E_ → GREY).
"""
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import google.generativeai as genai
from ddgs import DDGS

from crossword_schema import Clue, Coordinate, CrosswordGraph

logger = logging.getLogger(__name__)

# --- Zero-Persona Solver Model ---
# A lightweight, dedicated model instance for candidate generation.
# Completely separate from the conversational Tutor Agent — no persona,
# no chat history, no tools. Just structured JSON extraction.
_SOLVER_MODEL_NAME = "gemini-2.5-flash"

_SOLVER_PROMPT_TEMPLATE = (
    "You are a crossword solver API. You are given a crossword clue, "
    "its required letter count, and web search results for context.\n\n"
    "Clue: \"{clue_text}\"\n"
    "Required letters: {length}\n"
    "Known letter pattern: {pattern}\n\n"
    "Search results:\n{search_results}\n\n"
    "Return a JSON array of possible answers as single words (no spaces, "
    "no punctuation). Each answer MUST be exactly {length} letters long. "
    "Include ALL plausible candidates. Output ONLY valid JSON, nothing else."
)


class AutonomousSolver:
    """
    Deterministic crossword solver that pre-computes Ground Truth answers
    before the Tutor Agent begins chatting.

    Maintains two parallel state representations:
    1. grid_state: Maps (col, row) → character for every solved cell.
       This is the spatial "board" that enables constraint propagation.
    2. solved_clues: Maps (number, direction) → answer string.
       This is the semantic lookup the Tutor Agent uses.

    Usage:
        solver = AutonomousSolver(graph)
        solved_graph = solver.execute_phase_1_pass()
        # solved_graph now contains answers; feed to CrosswordTutorAgent
    """

    def __init__(self, graph: CrosswordGraph) -> None:
        """
        Initialize the solver with a parsed CrosswordGraph.

        Args:
            graph: A fully validated CrosswordGraph containing clues,
                   cell coordinates, and intersection mappings.
        """
        self.graph = graph

        # --- State Management ---
        # Spatial board: (col, row) → single uppercase character
        # This is the authoritative source of truth for what's on the grid.
        self.grid_state: Dict[Tuple[int, int], str] = {}

        # Semantic lookup: (clue_number, direction) → full answer string
        # e.g., (1, "Across") → "SCRIPT"
        self.solved_clues: Dict[Tuple[int, str], str] = {}

        # Track which clues have been solved to avoid re-processing
        self._solved_keys: Set[Tuple[int, str]] = set()

        # Candidate cache: prevents redundant search+LLM calls across phases.
        # Key: "<number>-<direction>" → List[str] of raw candidates.
        self.candidate_cache: Dict[str, List[str]] = {}

        # --- Local Word List (Phase 3) ---
        # Pre-indexed by word length for O(1) bucket lookups.
        self._words_by_length: Dict[int, List[str]] = {}
        self._load_word_list()

        # Build a coordinate-to-clue index for fast constraint lookups
        self._coord_to_clues: Dict[Tuple[int, int], List[Tuple[Clue, int]]] = {}
        for clue in self.graph.clues:
            for idx, cell in enumerate(clue.cells):
                coord = (cell.col, cell.row)
                if coord not in self._coord_to_clues:
                    self._coord_to_clues[coord] = []
                self._coord_to_clues[coord].append((clue, idx))

    def _load_word_list(self) -> None:
        """Load the local English word list, indexed by word length."""
        words_path = Path(__file__).parent.parent / "words_alpha.txt"
        if not words_path.exists():
            logger.warning(
                "Word list not found at %s. Phase 3 will be skipped.",
                words_path,
            )
            return

        with open(words_path, "r", encoding="utf-8") as f:
            for line in f:
                word = line.strip().upper()
                if word and word.isalpha():
                    length = len(word)
                    if length not in self._words_by_length:
                        self._words_by_length[length] = []
                    self._words_by_length[length].append(word)

        total = sum(len(v) for v in self._words_by_length.values())
        logger.info(
            "Loaded %d words from local dictionary (%d length buckets).",
            total, len(self._words_by_length),
        )

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------
    def lock_in_answer(self, clue: Clue, answer: str) -> None:
        """
        Locks a solved answer into the grid state and semantic lookup.

        Updates every cell coordinate that the clue occupies with the
        corresponding character from the answer string. This makes the
        letters immediately available for constraint propagation to
        intersecting clues.

        Args:
            clue: The Clue object being solved.
            answer: The uppercase answer string (must match clue.length).

        Raises:
            ValueError: If the answer length doesn't match the clue length,
                        or if a locked cell conflicts with an existing letter.
        """
        if len(answer) != clue.length:
            raise ValueError(
                f"Answer '{answer}' ({len(answer)} letters) doesn't match "
                f"clue {clue.number}-{clue.direction} ({clue.length} letters)"
            )

        key = (clue.number, clue.direction)

        # Write each character to its grid coordinate
        for idx, cell in enumerate(clue.cells):
            coord = (cell.col, cell.row)
            char = answer[idx]

            # Conflict detection: if a cell already has a letter, it must match
            if coord in self.grid_state:
                existing = self.grid_state[coord]
                if existing != char:
                    raise ValueError(
                        f"Conflict at ({cell.col}, {cell.row}): "
                        f"existing '{existing}' vs new '{char}' "
                        f"from {clue.number}-{clue.direction} = '{answer}'"
                    )
            else:
                self.grid_state[coord] = char

        # Update semantic lookup
        self.solved_clues[key] = answer
        self._solved_keys.add(key)

        logger.info(
            "Locked in %d-%s: '%s' (%d letters)",
            clue.number, clue.direction, answer, clue.length,
        )

    def get_known_pattern(self, clue: Clue) -> str:
        """
        Returns the current known letter pattern for a clue based on
        already-solved intersecting cells.

        Example: If clue is 5 letters and the 1st and 3rd cells are
        already filled with 'S' and 'R', returns "S_R__".

        Args:
            clue: The Clue to generate a pattern for.

        Returns:
            A pattern string where known letters are uppercase and
            unknown positions are underscores.
        """
        pattern_chars = []
        for cell in clue.cells:
            coord = (cell.col, cell.row)
            if coord in self.grid_state:
                pattern_chars.append(self.grid_state[coord])
            else:
                pattern_chars.append("_")
        return "".join(pattern_chars)

    def is_solved(self, clue: Clue) -> bool:
        """Check if a clue has already been solved."""
        return (clue.number, clue.direction) in self._solved_keys

    # ------------------------------------------------------------------
    # Candidate Generation (DuckDuckGo Search + Gemini JSON Extraction)
    # ------------------------------------------------------------------
    def _generate_candidates(self, clue: Clue) -> List[str]:
        """
        Generates candidate answers by searching the web and feeding results
        to a zero-persona LLM for structured JSON extraction.

        Pipeline:
        1. Build an optimized search query from the clue text.
        2. Search DuckDuckGo for relevant snippets.
        3. Send clue + snippets to Gemini with response_mime_type='application/json'
           to extract a JSON array of candidate words.
        4. Parse and normalize the results (uppercase, letters only).

        This is completely isolated from the conversational Tutor Agent —
        no persona, no chat history, no tools.

        Args:
            clue: The Clue object containing the text to solve.

        Returns:
            A list of raw candidate answer strings (may include wrong-length
            answers — the caller handles length filtering). Returns empty
            list if search or LLM call fails.
        """
        # --- Cache check ---
        cache_key = f"{clue.number}-{clue.direction}"
        if cache_key in self.candidate_cache:
            return self.candidate_cache[cache_key]

        try:
            # --- Step 1: Build search query ---
            query = f"{clue.text} crossword clue {clue.length} letters"

            # --- Step 2: Search DuckDuckGo ---
            try:
                results = DDGS().text(query, max_results=5)
                bodies = [r.get("body", "") for r in results if r.get("body")]
                search_text = "\n".join(bodies) if bodies else "No search results."
                # Sanitize for Windows console encoding
                search_text = search_text.encode("ascii", errors="replace").decode("ascii")
            except Exception as e:
                logger.warning(
                    "Search failed for %d-%s: %s. Attempting LLM-only.",
                    clue.number, clue.direction, e,
                )
                search_text = "No search results available."

            # --- Step 3: Zero-persona LLM call ---
            pattern = self.get_known_pattern(clue)
            prompt = _SOLVER_PROMPT_TEMPLATE.format(
                clue_text=clue.text,
                length=clue.length,
                pattern=pattern,
                search_results=search_text,
            )

            model = genai.GenerativeModel(_SOLVER_MODEL_NAME)
            response = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2,  # Low temp for deterministic extraction
                ),
                request_options={"timeout": 30},  # Fail fast on transient errors
            )

            raw_text = response.text.strip()

            # --- Step 4: Parse JSON + normalize ---
            candidates_raw = json.loads(raw_text)

            if not isinstance(candidates_raw, list):
                logger.warning(
                    "LLM returned non-list JSON for %d-%s: %s",
                    clue.number, clue.direction, type(candidates_raw).__name__,
                )
                return []

            # Normalize: uppercase, letters only
            candidates: List[str] = []
            for item in candidates_raw:
                if isinstance(item, str):
                    normalized = "".join(c for c in item if c.isalpha()).upper()
                    if normalized:
                        candidates.append(normalized)

            logger.debug(
                "Generated %d candidates for %d-%s '%s': %s",
                len(candidates), clue.number, clue.direction,
                clue.text, candidates[:5],
            )
            # Store in cache before returning
            self.candidate_cache[cache_key] = candidates
            return candidates

        except json.JSONDecodeError as e:
            logger.warning(
                "Bad JSON from LLM for %d-%s: %s",
                clue.number, clue.direction, e,
            )
            self.candidate_cache[cache_key] = []
            return []
        except Exception as e:
            # Retry once on transient errors (timeout, 503, etc.)
            import time
            logger.warning(
                "Candidate generation failed for %d-%s (retrying in 3s): %s",
                clue.number, clue.direction, e,
            )
            time.sleep(3)
            try:
                response = genai.GenerativeModel(_SOLVER_MODEL_NAME).generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.2,
                    ),
                    request_options={"timeout": 30},
                )
                raw_text = response.text.strip()
                candidates_raw = json.loads(raw_text)
                if isinstance(candidates_raw, dict):
                    candidates_raw = candidates_raw.get("candidates", candidates_raw.get("answers", []))
                if not isinstance(candidates_raw, list):
                    candidates_raw = [candidates_raw]
                candidates = [
                    "".join(c for c in item if c.isalpha()).upper()
                    for item in candidates_raw if isinstance(item, str)
                ]
                candidates = [c for c in candidates if c]
                self.candidate_cache[cache_key] = candidates
                return candidates
            except Exception as retry_e:
                logger.warning(
                    "Retry also failed for %d-%s: %s",
                    clue.number, clue.direction, retry_e,
                )
                self.candidate_cache[cache_key] = []
                return []

    # ------------------------------------------------------------------
    # Phase 1: Absolute Certainty Pass
    # ------------------------------------------------------------------
    def execute_phase_1_pass(self) -> int:
        """
        Executes the Absolute Certainty Pass: solves only clues where
        exactly ONE candidate matches both the meaning and the letter count.

        Rules:
        1. SKIP pun clues (text contains '?') — these require wordplay
           reasoning that is deferred to Phase 2/3.
        2. SKIP fill-in-the-blank clues (text contains '__') — these are
           inherently ambiguous without crossing letters.
        3. SKIP already-solved clues.
        4. Generate candidates via _generate_candidates().
        5. Filter candidates strictly by clue.length.
        6. ZERO AMBIGUITY GATE: Lock in ONLY if exactly 1 candidate survives
           the length filter. If 0 or 2+ candidates match, skip.

        Returns:
            The number of clues successfully locked in during this pass.
        """
        locked_count = 0

        for clue in self.graph.clues:
            key = (clue.number, clue.direction)

            # Skip already-solved clues
            if key in self._solved_keys:
                continue

            # --- Ambiguity Filter: Skip puns and fill-in-the-blanks ---
            if "?" in clue.text:
                logger.debug(
                    "Skipping %d-%s (pun clue): '%s'",
                    clue.number, clue.direction, clue.text,
                )
                continue

            if "__" in clue.text or "___" in clue.text:
                logger.debug(
                    "Skipping %d-%s (fill-in-the-blank): '%s'",
                    clue.number, clue.direction, clue.text,
                )
                continue

            # --- Generate and filter candidates ---
            raw_candidates = self._generate_candidates(clue)

            if not raw_candidates:
                continue

            # Length filter: strip whitespace/punctuation and match exact length
            filtered: List[str] = []
            seen: Set[str] = set()
            for candidate in raw_candidates:
                # Normalize: uppercase, letters only
                normalized = "".join(c for c in candidate if c.isalpha()).upper()
                if len(normalized) == clue.length and normalized not in seen:
                    filtered.append(normalized)
                    seen.add(normalized)

            # --- Zero Ambiguity Gate ---
            if len(filtered) == 1:
                answer = filtered[0]

                # Validate against any already-known crossing letters
                pattern = self.get_known_pattern(clue)
                conflict = False
                for i, (pat_char, ans_char) in enumerate(zip(pattern, answer)):
                    if pat_char != "_" and pat_char != ans_char:
                        logger.warning(
                            "Candidate '%s' for %d-%s conflicts at index %d: "
                            "expected '%s', got '%s'. Skipping.",
                            answer, clue.number, clue.direction,
                            i, pat_char, ans_char,
                        )
                        conflict = True
                        break

                if not conflict:
                    try:
                        self.lock_in_answer(clue, answer)
                        locked_count += 1
                    except ValueError as e:
                        logger.warning(
                            "Failed to lock %d-%s = '%s': %s",
                            clue.number, clue.direction, answer, e,
                        )
            elif len(filtered) > 1:
                logger.debug(
                    "Ambiguous: %d-%s has %d candidates: %s. Skipping.",
                    clue.number, clue.direction, len(filtered),
                    filtered[:5],
                )
            # len(filtered) == 0: no candidates matched length, skip silently

        logger.info(
            "Phase 1 complete: locked in %d of %d total clues.",
            locked_count, len(self.graph.clues),
        )

        return locked_count

    # ------------------------------------------------------------------
    # Phase 2: Constraint Propagation (The Domino Effect)
    # ------------------------------------------------------------------
    def execute_phase_2_pass(self) -> int:
        """
        Iteratively solves remaining clues using crossing letters from
        already-locked answers.

        Each iteration:
        1. Loops through all unsolved clues.
        2. Builds a known_pattern from the grid_state (e.g., 'G_E_').
        3. Fetches candidates (from cache) and filters by BOTH length
           AND pattern match.
        4. Locks in if exactly 1 candidate matches (zero ambiguity).
        5. Repeats until no new clues are solved in an iteration (stable state).

        This handles pun clues (?) and fill-in-the-blanks (__) that
        Phase 1 skipped — crossing letters can resolve the ambiguity.

        Returns:
            Total number of clues solved across all iterations.
        """
        total_locked = 0
        iteration = 0

        while True:
            iteration += 1
            newly_solved = 0

            for clue in self.graph.clues:
                key = (clue.number, clue.direction)

                # Skip already-solved clues
                if key in self._solved_keys:
                    continue

                # Get the current known letter pattern from crossing words
                pattern = self.get_known_pattern(clue)

                # Skip clues with no known letters yet — nothing to constrain
                if pattern == "_" * clue.length:
                    continue

                # --- Fully-formed domino auto-lock ---
                # If crossing letters have completely spelled out this answer
                # (no underscores remain), lock it in immediately. This handles
                # multi-word answers (OHISEE, SNORTAT) not in any dictionary.
                if "_" not in pattern:
                    try:
                        self.lock_in_answer(clue, pattern)
                        newly_solved += 1
                        logger.info(
                            "Phase 2: Auto-locked %d-%s = '%s' (fully formed by crossings)",
                            clue.number, clue.direction, pattern,
                        )
                    except ValueError as e:
                        logger.warning(
                            "Phase 2: Auto-lock conflict %d-%s = '%s': %s",
                            clue.number, clue.direction, pattern, e,
                        )
                    continue

                # Fetch candidates (hits cache from Phase 1 or previous iterations)
                raw_candidates = self._generate_candidates(clue)
                if not raw_candidates:
                    continue

                # Filter by BOTH length AND pattern match
                filtered: List[str] = []
                seen: Set[str] = set()
                for candidate in raw_candidates:
                    normalized = "".join(c for c in candidate if c.isalpha()).upper()

                    # Length check
                    if len(normalized) != clue.length:
                        continue

                    # Deduplicate
                    if normalized in seen:
                        continue
                    seen.add(normalized)

                    # Pattern match: every known letter must match
                    match = True
                    for i, (pat_char, cand_char) in enumerate(zip(pattern, normalized)):
                        if pat_char != "_" and pat_char != cand_char:
                            match = False
                            break

                    if match:
                        filtered.append(normalized)

                # Zero Ambiguity Gate (Phase 2)
                if len(filtered) == 1:
                    answer = filtered[0]
                    try:
                        self.lock_in_answer(clue, answer)
                        newly_solved += 1
                    except ValueError as e:
                        logger.warning(
                            "Phase 2: Failed to lock %d-%s = '%s': %s",
                            clue.number, clue.direction, answer, e,
                        )
                elif len(filtered) > 1:
                    logger.debug(
                        "Phase 2: %d-%s still ambiguous with pattern '%s': %d candidates %s",
                        clue.number, clue.direction, pattern,
                        len(filtered), filtered[:5],
                    )

            total_locked += newly_solved

            logger.info(
                "Phase 2 iteration %d: solved %d new clues (total Phase 2: %d).",
                iteration, newly_solved, total_locked,
            )

            # Break when no progress is made (stable state)
            if newly_solved == 0:
                break

        logger.info(
            "Phase 2 complete: solved %d additional clues in %d iterations. "
            "Total solved: %d / %d.",
            total_locked, iteration,
            len(self.solved_clues), len(self.graph.clues),
        )

        # --- Unsolved Diagnostic Dump ---
        unsolved = [
            c for c in self.graph.clues
            if (c.number, c.direction) not in self._solved_keys
        ]
        if unsolved:
            logger.info("=== UNSOLVED CLUES: %d remaining ===", len(unsolved))
            for clue in unsolved:
                pattern = self.get_known_pattern(clue)
                cache_key = f"{clue.number}-{clue.direction}"
                cached = self.candidate_cache.get(cache_key, [])
                # Show which cached candidates match the pattern
                matching = [
                    c for c in cached
                    if len("".join(ch for ch in c if ch.isalpha())) == clue.length
                    and all(
                        p == "_" or p == a
                        for p, a in zip(pattern, "".join(ch for ch in c if ch.isalpha()).upper())
                    )
                ]
                logger.info(
                    "UNSOLVED: %d-%s: \"%s\" (Length: %d) - Pattern: %s | "
                    "Cached candidates: %d, Pattern-matched: %d %s",
                    clue.number, clue.direction, clue.text, clue.length,
                    pattern, len(cached), len(matching),
                    matching[:5] if matching else "",
                )

        return total_locked

    # ------------------------------------------------------------------
    # Phase 3: Local Regex Dictionary Pass (Zero API Calls)
    # ------------------------------------------------------------------
    def execute_phase_3_pass(self) -> int:
        """
        Solves remaining clues using a local English word list and regex
        pattern matching. No web searches, no LLM calls.

        For each unsolved clue with at least one known crossing letter:
        1. Builds a regex from the known pattern (e.g., 'S_R__' → ^S.R..$ ).
        2. Scans the length-indexed local word list for matches.
        3. Locks in if exactly 1 word matches (zero ambiguity gate).
        4. Iterates until no new progress (domino effect continues).

        Returns:
            Total number of clues solved across all iterations.
        """
        if not self._words_by_length:
            logger.info("Phase 3 skipped: no local word list loaded.")
            return 0

        total_locked = 0
        iteration = 0

        while True:
            iteration += 1
            newly_solved = 0

            for clue in self.graph.clues:
                key = (clue.number, clue.direction)
                if key in self._solved_keys:
                    continue

                pattern = self.get_known_pattern(clue)

                # Skip clues with no known letters
                if pattern == "_" * clue.length:
                    continue

                # --- Fully-formed domino auto-lock ---
                if "_" not in pattern:
                    try:
                        self.lock_in_answer(clue, pattern)
                        newly_solved += 1
                        logger.info(
                            "Phase 3: Auto-locked %d-%s = '%s' (fully formed by crossings)",
                            clue.number, clue.direction, pattern,
                        )
                    except ValueError as e:
                        logger.warning(
                            "Phase 3: Auto-lock conflict %d-%s = '%s': %s",
                            clue.number, clue.direction, pattern, e,
                        )
                    continue

                # Build regex from pattern: S_R__ → ^S.R..$ (case-insensitive)
                regex_str = "^" + pattern.replace("_", ".") + "$"
                try:
                    regex = re.compile(regex_str)
                except re.error:
                    continue

                # Scan the local word list for this length
                word_bucket = self._words_by_length.get(clue.length, [])
                matches = [w for w in word_bucket if regex.match(w)]

                # Safety guard: short words with many blanks are dangerous.
                # Obscure dictionary words (e.g., BIGATE) can beat real answers
                # (BIG APE) that aren't in the word list. Only proceed if the
                # word is long (7+) or nearly complete (≤1 blank).
                num_blanks = pattern.count("_")
                if clue.length < 7 and num_blanks > 1:
                    continue

                # Zero Ambiguity Gate
                if len(matches) == 1:
                    answer = matches[0]
                    try:
                        self.lock_in_answer(clue, answer)
                        newly_solved += 1
                        logger.info(
                            "Phase 3: Locked %d-%s = '%s' (regex: %s)",
                            clue.number, clue.direction, answer, regex_str,
                        )
                    except ValueError as e:
                        logger.warning(
                            "Phase 3: Conflict locking %d-%s = '%s': %s",
                            clue.number, clue.direction, answer, e,
                        )
                elif len(matches) > 1:
                    logger.debug(
                        "Phase 3: %d-%s pattern '%s' has %d dictionary matches: %s",
                        clue.number, clue.direction, pattern,
                        len(matches), matches[:5],
                    )

            total_locked += newly_solved

            logger.info(
                "Phase 3 iteration %d: solved %d new clues (total Phase 3: %d).",
                iteration, newly_solved, total_locked,
            )

            if newly_solved == 0:
                break

        logger.info(
            "Phase 3 complete: solved %d additional clues in %d iterations. "
            "Total solved: %d / %d.",
            total_locked, iteration,
            len(self.solved_clues), len(self.graph.clues),
        )

        # --- Final Unsolved Diagnostic ---
        unsolved = [
            c for c in self.graph.clues
            if (c.number, c.direction) not in self._solved_keys
        ]
        if unsolved:
            logger.info("=== STILL UNSOLVED: %d remaining ===", len(unsolved))
            for clue in unsolved:
                pattern = self.get_known_pattern(clue)
                regex_str = "^" + pattern.replace("_", ".") + "$"
                word_bucket = self._words_by_length.get(clue.length, [])
                try:
                    matches = [w for w in word_bucket if re.match(regex_str, w)]
                except re.error:
                    matches = []
                logger.info(
                    "UNSOLVED: %d-%s: \"%s\" (Len: %d) Pattern: %s | "
                    "Dict matches: %d %s",
                    clue.number, clue.direction, clue.text, clue.length,
                    pattern, len(matches), matches[:5] if matches else "",
                )

        return total_locked




    # ------------------------------------------------------------------
    # Summary / Export
    # ------------------------------------------------------------------
    def get_solve_summary(self) -> Dict:
        """
        Returns a summary of the current solve state for diagnostics.

        Returns:
            Dictionary with solve statistics and per-clue results.
        """
        total = len(self.graph.clues)
        solved = len(self.solved_clues)
        total_cells = self.graph.width * self.graph.height
        filled_cells = len(self.grid_state)

        return {
            "total_clues": total,
            "solved_clues": solved,
            "unsolved_clues": total - solved,
            "solve_percentage": round(solved / total * 100, 1) if total > 0 else 0,
            "total_cells": total_cells,
            "filled_cells": filled_cells,
            "answers": {
                f"{num}-{direction}": answer
                for (num, direction), answer in sorted(self.solved_clues.items())
            },
        }

    def get_enriched_context(self) -> Dict[Tuple[int, str], str]:
        """
        Returns the solved answers in a format ready for injection into
        the Tutor Agent's system prompt.

        Returns:
            Dictionary mapping (clue_number, direction) → answer string
            for all solved clues.
        """
        return dict(self.solved_clues)
