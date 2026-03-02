# 🧩 AI Crossword Tutor: Neuro-Symbolic Puzzle Solver

A Streamlit-based AI crossword assistant that combines **PDF parsing**, a **3-phase autonomous solver**, and a **Gemini-powered conversational tutor**. Upload a crossword PDF and the system pre-solves ~95% of clues before you even ask a question — then tutors you through the rest using structured hint escalation.

---

## Quick Start

### 1. Set your API Key

```bash
# Windows PowerShell
$env:GOOGLE_API_KEY = "your-google-api-key-here"

# macOS / Linux
export GOOGLE_API_KEY="your-google-api-key-here"
```

> **Note:** A [Google AI Studio](https://aistudio.google.com/apikey) API key with access to **Gemini 2.5 Flash** is required. The free tier works but may produce `504 timeout` errors during heavy usage, which reduces solve rates.

### 2. Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

---

## How to Use

### Mode 1: Freestyle Questions (No PDF Required)

Simply type a crossword-style question in the chat:

> *"6-letter word for country with most medals at 2026 Olympics?"*

The tutor will research the answer using web search, verify it with programmatic letter counting, and guide you through hints — never giving the answer outright unless you explicitly ask.

### Mode 2: PDF Puzzle Upload

1. **Source a crossword PDF.** For example, go to the [LA Times Crossword](https://www.latimes.com/games#clx-item-1) and use the print/PDF option to save the puzzle.
2. **Upload the PDF** via the sidebar file uploader.
3. The system automatically:
   - Parses the grid geometry (dimensions, black squares, clue numbers)
   - Extracts and structures all clue text
   - Runs the 3-phase autonomous solver
   - Displays a summary: e.g., `Phase 1: 87 | Phase 2: +48 | Phase 3: +3 | Total: 138/142 (97.2%)`
4. **Ask for help** on any clue by number:
   > *"Can you help me with 14 Across?"*

   The tutor knows the pre-solved answer with certainty and will guide you through hints without ever spoiling it.

### ⚠️ PDF Rendering Caveat

The parser works by analyzing the **vector drawing commands** inside the PDF (rectangles for cells, text spans for clue numbers). Accuracy depends on how the PDF was generated:

- **Best case (~95-97%):** PDFs with clearly drawn cell borders and well-separated clue text. Most newspaper PDF exports work well.
- **Degraded case (~80-90%):** Some PDF generators (particularly browser "Save as PDF") produce ultra-thin CSS borders that vanish during rasterization, causing localized grid mapping gaps. This is a limitation of the PDF source, not the parser.

The parser logs all unsolved clues with their known letter patterns (e.g., `Pattern: S_UTIN`) so you can see exactly where gaps occur.

---

## Architecture

```
PDF Upload
  → PDFTextExtractor    (raw text → structured clue dictionaries)
  → PDFGridParser       (vector drawings → grid geometry)
  → PDFCrosswordParser  (clue dicts + geometry → CrosswordGraph)
  → AutonomousSolver    (3-phase solve → Ground Truth answers)
  → CrosswordTutorAgent (Gemini chat with Ground Truth context)
```

### The CrosswordGraph

The universal data model that bridges raw input and AI reasoning. Contains:
- Grid dimensions and black square positions
- Every clue with its text, word length, and exact cell coordinates
- All intersection points (where Across and Down clues share a cell)

This gives the LLM access to **mathematically precise constraint information** it cannot infer from unstructured text.

---

## The 3-Phase Autonomous Solver

The solver runs **before** the chat session begins, pre-computing Ground Truth answers so the tutor never hallucinates.

### Phase 1 — Absolute Certainty Pass

For each clue, the solver:
1. Sends the clue text + word length to **Gemini** and asks for candidate answers
2. Searches the web via **DuckDuckGo** for supporting evidence
3. Programmatically verifies that each candidate's letter count matches the required length
4. Checks candidates against all **crossing letter constraints** from previously locked answers
5. Locks in a candidate only if it's the **sole match** that satisfies all constraints

**GenAI connection:** This phase uses a **ReAct (Reason + Act)** loop — the LLM reasons about the clue semantics, acts by generating candidates, observes the constraint verification results, and iterates. The tool-calling pattern (search → verify length → check crossings) is a direct implementation of the ReAct framework.

### Phase 2 — Constraint Propagation

After Phase 1 locks in ~60% of answers, Phase 2 exploits the **cascading effect** of crossing letters:
1. For each unsolved clue, reads the known letters from crossing answers (e.g., `S_R_PT`)
2. Runs iterative constraint propagation: each newly locked answer reveals more crossing letters, which may uniquely determine other clues
3. Repeats until no more progress is made

This is a **deterministic, zero-API-call phase** — pure algorithmic constraint propagation. It typically solves an additional 25-35% of clues.

### Phase 3 — Local Dictionary Regex Match

For remaining unsolved clues:
1. Builds a regex from the known pattern (e.g., `^S.R.PT$`)
2. Searches a local 370K-word dictionary for matches
3. Locks in only if exactly **one** dictionary word matches the pattern
4. Safety guard: skips short words with many blanks to prevent incorrect locks

**GenAI connection:** Phase 3 demonstrates the value of **neuro-symbolic hybrid architecture** — combining neural reasoning (Phases 1's LLM calls) with symbolic computation (regex matching against a formal dictionary). The LLM handles semantic understanding; the algorithm handles exhaustive enumeration.

---

## The Conversational Tutor

### Hint Escalation Protocol

When you ask about a clue, the tutor follows a mandatory 3-step escalation:

| Step | Type | Example |
|------|------|---------|
| 1 | **Semantic** — broad thematic clue | *"Think cold. Think fjords."* |
| 2 | **Structural** — relational context | *"This country dominates cross-country skiing."* |
| 3 | **Direct** — near-giveaway | *"It starts with N and ends with Y."* |

The tutor **never reveals the answer** unless explicitly asked. This mirrors the pedagogical structure of **Analogical Prompting** — each hint reframes the problem from a different cognitive angle, encouraging the solver to make the connection themselves.

### Confidence & Ambiguity Handling

- **Ground Truth clues (✅):** The tutor knows the answer with 100% certainty and hints at it confidently.
- **Unsolved clues (❓):** The tutor researches via web search and triggers the **Ambiguity Failsafe** when multiple candidates fit: *"That's a great candidate, but a few other words fit perfectly here. What crossing letters do you have?"*
- **Pen vs. Pencil Rule:** If not 100% confident, the tutor will never confirm a guess.

**GenAI connection:** The system prompt implements **Chain-of-Thought (CoT)** reasoning by requiring the LLM to silently execute a candidate selection protocol (research → verify length → check constraints) before generating any user-facing text. The internal reasoning chain is never exposed — only the final tutor persona speaks.

---

## Project Structure

```
├── app.py                      # Streamlit UI — file upload, chat interface
├── llm_agent.py                # Gemini model config, system prompt, tool definitions
├── crossword_schema.py         # Pydantic models: CrosswordGraph, Clue, Intersection
├── inspect_pdf.py              # CLI test harness for the parsing + solving pipeline
├── words_alpha.txt             # 370K-word English dictionary for Phase 3
├── requirements.txt            # Python dependencies
├── parsers/
│   ├── base_parser.py          # Abstract base class for all input parsers
│   ├── pdf_parser.py           # Master orchestrator: stitches text + grid → graph
│   ├── pdf_text_extractor.py   # Column-aware text extraction + clue parsing
│   └── pdf_grid_parser.py      # Cell-first grid geometry extraction from PDF vectors
└── solvers/
    └── autonomous_solver.py    # 3-phase pre-solve pipeline (Phase 1, 2, 3)
```

## Dependencies

- **Streamlit** — Web UI framework
- **Google Generative AI** (`google-generativeai`) — Gemini 2.5 Flash for candidate generation and tutoring
- **PyMuPDF** (`fitz`) — PDF parsing (vector drawings + text extraction)
- **Pydantic** — Schema validation for the CrosswordGraph
- **DDGS** — DuckDuckGo search for real-time trivia lookup
