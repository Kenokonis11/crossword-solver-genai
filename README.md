# 🧩 AI Crossword Assistant: Neuro-Symbolic CSP Solver

## Overview
This project is a local, Python-based AI agent that assists users in solving crossword puzzles. It moves beyond standard LLM heuristics by implementing a formal **Constraint Satisfaction Problem (CSP)** architecture. 

It combines the semantic reasoning of Large Language Models (for web-fetching, embedding-based fact validation, and ReAct-driven tutoring) with a rigid algorithmic backend (Arc Consistency, MRV heuristics, and state-backed depth-first backtracking) to solve puzzles with human-like, non-linear intuition.

## Core Architecture & Engine Dynamics

### 1. Formal CSP Formulation
* **Variables:** Every clue (e.g., 1-Across) is a variable.
* **Domains:** The candidate pool for each clue (limited to Top-K via Beam Search to prevent combinatorial explosion).
* **Arc Consistency (AC-3):** When a crossing letter is locked, the engine instantly prunes any candidates in intersecting domains that violate the new constraint. 
* **Variable Selection (MRV):** The engine does not solve linearly (1 to 100). It uses the **Minimum Remaining Values** heuristic, always selecting the unlocked clue with the smallest valid domain to solve next.

### 2. Multiplicative Bayesian Confidence Scoring
Candidate scores are computed dynamically as probabilities, allowing strong crossing matches to exponentially spike confidence:

$$P(Candidate) = P(Base) \times \left( \frac{Matches}{Total Crossings} \right)^2 \times P(Web) \times P(Theme)$$

* **Dynamic Locking:** A candidate is only locked if $P(Candidate)$ exceeds a dynamic threshold: `Base Threshold + 0.2 * (Grid Fill Ratio)`. As the grid fills, the engine requires higher certainty to lock new words.

### 3. Backtracking & Stall Recovery
* **Decision Stack:** Every assignment pushes a snapshot of the grid's domains to a stack. If a conflict occurs (domain size = 0), the engine pops the stack, restores the prior state, penalizes the failed branch, and explores the next candidate.
* **Stall Recovery:** If the solver stalls (no variables can cross the dynamic threshold), it forces an unlock on the lowest-confidence locked variable to unblock the grid.

### 4. Web Semantic Pipeline & Tools
* **Contextual Web Extraction:** Uses the `search_web()` tool. Instead of raw frequency, the backend uses lightweight vector embeddings to compute semantic similarity between the crossword clue and the extracted web phrases.
* **User Learning Prior:** Stores user overrides in SQLite. Applies a soft Bayesian update (`0.9 * General Prior + 0.1 * User Prior`) to personalize vocabulary without suffering from bias drift.

## System Fallbacks & UI
* **Ingestion:** Prioritizes `.puz` (via `puzpy`) and web fetching (via `xword-dl`, supporting authenticated NYT access).
* **OCR/PDF Fallback:** If visual ingestion fails, the UI defaults to a Regex-assisted Manual Coordinate mode (e.g., user inputs `1-A, 5 letters, P...S`).
* **The 3-Hint ReAct Tutor:** The LLM interacts with the user via a strict tool-calling loop, escalating hints (Semantic -> Structural -> Direct) based on the candidate domain structure, only yielding the final answer if explicitly commanded.*Choose the strategy that best fits the complexity of the clue or the available tools.*
