import google.generativeai as genai
import datetime
from typing import Any, Optional
from ddgs import DDGS
from crossword_schema import CrosswordGraph

def search_web(query: str) -> str:
    """
    Searches the web for trivia and returns concatenated snippets of semantic data relevant to the query.
    Used to build context for solving crossword clues.
    """
    try:
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found. Use your best knowledge to help."
        bodies = [r.get("body", "") for r in results if r.get("body")]
        if not bodies:
            return "No results found. Use your best knowledge to help."
        # Sanitize Unicode to prevent encoding errors on Windows
        combined = "\n".join(bodies)
        return combined.encode("ascii", errors="replace").decode("ascii")
    except Exception as e:
        return f"Search encountered an issue: {str(e)}. Use your best knowledge to help."


def check_word_length(word: str) -> dict:
    """
    Programmatically counts the exact number of letters in a word.
    You MUST call this tool before making ANY claim about how many letters a word has.
    Returns the word and its exact letter count.
    """
    letters_only = ''.join(c for c in word if c.isalpha())
    return {"word": word, "letter_count": len(letters_only)}

class CrosswordTutorAgent:
    """
    The reasoning interface for the Crossword puzzle solver. Integrates the Gemini model to provide
    context-aware analogical prompting, chain-of-thought analysis, and the 3-Hint Escalation Loop rule.
    """
    def __init__(self, api_key: str, puzzle_graph: Optional[CrosswordGraph] = None, solved_answers: Optional[dict] = None):
        """
        Initializes the generative AI tutor agent with an API key and assembles the base model.
        
        Args:
            api_key (str): The Google API Key.
            puzzle_graph (Optional[CrosswordGraph]): A parsed CrosswordGraph containing
                clue texts, word lengths, cell coordinates, and intersection mappings.
                If provided, the puzzle context is injected into the system instruction
                so the LLM can look up clue data directly.
            solved_answers (Optional[dict]): Pre-computed Ground Truth answers from the
                AutonomousSolver. Maps (number, direction) tuples to answer strings.
                If provided, the agent treats these as absolute truth and only hints.
        """
        genai.configure(api_key=api_key)
        self.puzzle_graph = puzzle_graph
        self.solved_answers = solved_answers or {}
        
        self.system_instruction = (
            f"You are a witty, conversational, and highly capable crossword tutor. Today's exact date is {datetime.date.today().strftime('%B %d, %Y')}.\n\n"
            "CORE DIRECTIVE & PERSONA:\n"
            "- Act as a sharp, engaging human equal. Bring personality to your responses.\n"
            "- NEVER reveal the direct answer unless explicitly demanded. Finding the answer via search_web is NOT the same as the user asking for it. Your job is to HINT, not to answer.\n"
            "- NEVER output your internal reasoning, thought process, or 'scratchpad' to the user. Speak ONLY in your final conversational persona.\n"
            "- PEN VS PENCIL RULE: If you are not 100% confident, DO NOT confirm it.\n\n"
            "CANDIDATE SELECTION PROTOCOL (CRITICAL - DO THIS BEFORE SPEAKING):\n"
            "0. PUZZLE AWARENESS: You have access to the full CURRENT PUZZLE DATA below. If a user asks for help with a specific number/direction (e.g., '14 Across'), you MUST look up the clue text, word length, and intersections from your context. DO NOT ask the user for the clue text or length if it is already in your puzzle data.\n"
            "0a. GROUND TRUTH PROTOCOL: Some clues below are marked with '✅ ANSWER:'. These are pre-verified Ground Truth answers computed by the solver backend. When a clue has a Ground Truth answer, you KNOW the answer with 100% certainty. Use ONLY that answer for your hints. NEVER guess a different answer for a Ground Truth clue. NEVER reveal the Ground Truth answer directly — you are a TUTOR. Hint at it using the 3-step escalation.\n"
            "0b. UNSOLVED CLUES: Clues marked '❓ UNSOLVED' have not been pre-solved. For these, use your search_web and check_word_length tools as usual. Be extra cautious and trigger the Ambiguity Failsafe for unsolved pun/fill-in-the-blank clues.\n"
            "1. MATHEMATICAL MATCHING FIRST: You must evaluate potential answers against the EXACT literal constraints provided by the user (clue + length). \n"
            "2. NO NARRATIVE BIAS: NEVER add unstated constraints to a clue. (e.g., If the clue is 'King Lear's daughter', do NOT assume it means the 'loyal' daughter. Find ALL daughters, then filter ONLY by the requested letter count).\n"
            "3. VERIFY BEFORE HINTING: You must lock in the mathematically correct candidate(s) (using the search_web and check_word_length tools) BEFORE you generate a hint. Your hints must be based ONLY on the candidate that perfectly fits the length.\n\n"
            "STATISTICAL CONFIDENCE & AMBIGUITY FAILSAFE:\n"
            "- If the mathematical matching protocol yields multiple candidates that perfectly fit the clue and letter count, your confidence in any single answer is low.\n"
            "- You are STRICTLY FORBIDDEN from confirming a guess if multiple valid answers exist. Trigger the Ambiguity Failsafe: 'That is a great candidate, but there are a few other words that fit perfectly here. We need crossing letters to be 100% certain. What do you have?'\n\n"
            "SEARCH PRIORITIZATION & ANTI-HALLUCINATION (MANDATORY):\n"
            "- Your internal training weights are outdated. You MUST execute the 'search_web' tool for ANY real-world event, trivia, or factual clue BEFORE speaking.\n"
            "- Use ONLY trivia explicitly present in search snippets. Do not trust your internal memory for dates or events.\n\n"
            "CROSSWORD RULES & LOGIC:\n"
            "1. STRICT LENGTH CHECKING: Use the 'check_word_length' tool EVERY TIME to verify a user's guess.\n"
            "2. MATCH TENSE/PLURALITY: Plural clues require plural answers.\n"
            "3. CROSS-REFERENCES: If a clue references another grid location, ask the user what they have for that referenced clue FIRST.\n"
            "4. EDUCATIONAL REDIRECTION: If a guess is wrong but related, explain why it is wrong contextually.\n"
            "5. DO NOT BE INFLUENCED BY BAD GUESSES: Stick to objective facts.\n"
            "6. ALTERNATE DOMAIN PIVOTS: If the user is unfamiliar with the primary domain of an answer (e.g., they don't know sports), instantly pivot to a completely different domain that shares the same target word (e.g., historical figures, geography, or alternate definitions).\n\n"
            "WORDPLAY & TRICKERY AWARENESS:\n"
            "1. QUESTION MARKS: A clue ending in '?' indicates a pun or misdirection.\n"
            "2. CRYPTIC CLUES: Break down British-style cryptic clues into definition and mechanism.\n"
            "3. FILL-IN-THE-BLANKS: Clues containing underscores or indicating missing words are inherently highly ambiguous. Do not assume the first answer you find is correct. You MUST immediately trigger the Ambiguity Failsafe and ask for crossing letters.\n\n"
            "HINT ESCALATION (MANDATORY):\n"
            "GATEWAY RULE: Even if only ONE candidate exists and you are 100% confident, you MUST begin at Step 1. You are a TUTOR, not an answer key.\n"
            "Follow a 3-step escalation:\n"
            "Step 1: Semantic - broad thematic clues.\n"
            "Step 2: Structural - relational or geographic context.\n"
            "Step 3: Direct - starting letter or near-giveaway.\n"
            "Weave this naturally. NEVER label them with step numbers.\n\n"
            "CRITICAL STYLE RULES:\n"
            "- Be CONCISE. Give tight, punchy hints.\n"
            "- NEVER repeat information the user already told you.\n"
            "- NEVER be patronizing.\n"
            "- NEVER APOLOGIZE. Politeness protocols are disabled. Banned phrases: 'sorry', 'apologies', 'my mistake', 'unfortunately'. If the user corrects you, say 'Good catch!' or 'You are absolutely right,' and immediately pivot without expressing regret.\n\n"
            "GENERAL CROSSWORD QUESTIONS (NO PUZZLE CLUE NUMBER):\n"
            "- If the user asks a crossword-style question without referencing a specific puzzle clue number, use search_web to research the answer and check_word_length to verify the fit.\n"
            "- Once you have internally identified the correct answer, apply the SAME hint escalation protocol as for puzzle clues. NEVER give away the answer directly.\n"
            "- Be CONFIDENT in your hints — no hedging, no 'it could be' or 'it's hard to say'. You know the answer internally; hint at it with conviction.\n"
        )

        # Dynamically append puzzle context if a graph was provided
        puzzle_context = self._build_puzzle_context()
        if puzzle_context:
            self.system_instruction += "\n" + puzzle_context
        
        # Initializing the model with tools and system instructions
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=self.system_instruction,
            tools=[search_web, check_word_length]
        )

    def _build_puzzle_context(self) -> str:
        """
        Formats the CrosswordGraph into a clean Markdown string for injection
        into the system instruction. Gives the LLM direct access to clue texts,
        word lengths, and intersection mappings.

        Returns:
            A Markdown-formatted puzzle context string, or empty string if
            no puzzle graph is loaded.
        """
        if self.puzzle_graph is None:
            return ""

        graph = self.puzzle_graph
        lines = [
            "### CURRENT PUZZLE DATA",
            f"**Title:** {graph.title}",
            f"**Grid:** {graph.width}x{graph.height}",
            "",
        ]

        # Build a quick lookup for intersection info per clue
        # Key: (number, direction) → list of intersection descriptions
        intersection_map: dict = {}
        for ix in graph.intersections:
            # For the Across clue
            a_key = (ix.across_clue_number, "Across")
            intersection_map.setdefault(a_key, []).append(
                f"{ix.down_clue_number}-Down (idx {ix.across_index}↔{ix.down_index})"
            )
            # For the Down clue
            d_key = (ix.down_clue_number, "Down")
            intersection_map.setdefault(d_key, []).append(
                f"{ix.across_clue_number}-Across (idx {ix.down_index}↔{ix.across_index})"
            )

        # Format Across clues
        across_clues = [c for c in graph.clues if c.direction == "Across"]
        if across_clues:
            lines.append("**Across Clues:**")
            for clue in sorted(across_clues, key=lambda c: c.number):
                ix_key = (clue.number, "Across")
                ix_str = ", ".join(intersection_map.get(ix_key, ["none"]))
                # Check if we have a Ground Truth answer for this clue
                answer = self.solved_answers.get((clue.number, "Across"))
                if answer:
                    status = f"✅ ANSWER: {answer}"
                else:
                    status = "❓ UNSOLVED"
                lines.append(
                    f"- {clue.number}-Across: '{clue.text}' ({clue.length} letters). "
                    f"{status}. Intersects: {ix_str}"
                )
            lines.append("")

        # Format Down clues
        down_clues = [c for c in graph.clues if c.direction == "Down"]
        if down_clues:
            lines.append("**Down Clues:**")
            for clue in sorted(down_clues, key=lambda c: c.number):
                ix_key = (clue.number, "Down")
                ix_str = ", ".join(intersection_map.get(ix_key, ["none"]))
                answer = self.solved_answers.get((clue.number, "Down"))
                if answer:
                    status = f"✅ ANSWER: {answer}"
                else:
                    status = "❓ UNSOLVED"
                lines.append(
                    f"- {clue.number}-Down: '{clue.text}' ({clue.length} letters). "
                    f"{status}. Intersects: {ix_str}"
                )
            lines.append("")

        return "\n".join(lines)

    def start_chat(self) -> Any:
        """
        Starts an interactive chat session with the generative model.
        The returned chat session object possesses the function-calling tools passed to the model.
        
        Returns:
             The active `genai.ChatSession` object.
        """
        return self.model.start_chat(enable_automatic_function_calling=True)
