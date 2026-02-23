import google.generativeai as genai
import datetime
from typing import Any
from ddgs import DDGS

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

def get_variable_domain(variable_id: str) -> dict:
    """
    Returns the current candidate pool (domain) and the locked status for a specific crossword clue ID.
    Used to analyze the algorithm's confidence and current path.
    """
    import streamlit as st
    
    if "solver" not in st.session_state:
        return {"error": "No active puzzle."}
        
    var = st.session_state.solver.variables.get(variable_id.upper())
    if not var:
        return {"error": "Variable not found."}
        
    return {
        "clue_text": var.clue_text,
        "locked_candidate": var.locked_candidate.text if var.locked_candidate else None,
        "top_candidates": [
            {"text": c.text, "score": c.score} 
            for c in var.domain[:3]
        ]
    }

def fetch_puzzle_via_xword_dl(source: str) -> str:
    """
    Fetches a puzzle from a publication like NYT or LA Times and returns a success message mapping back into the solver.
    """
    pass

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
    def __init__(self, api_key: str):
        """
        Initializes the generative AI tutor agent with an API key and assembles the base model.
        
        Args:
            api_key (str): The Google API Key.
        """
        genai.configure(api_key=api_key)
        
        self.system_instruction = (
            f"You are a sharp, conversational crossword tutor. Today's exact date is {datetime.date.today().strftime('%B %d, %Y')}.\n\n"
            "CORE DIRECTIVE:\n"
            "You are a TUTOR. NEVER reveal the direct answer unless explicitly demanded. Guide the user through hints.\n"
            "- NEVER list possible answers as a set of choices (e.g., do NOT say 'it is Goneril, Regan, or Cordelia'). Listing the options gives the answer away. Describe the answer using hints only.\n"
            "- Many solvers use PEN, not pencil. A wrong confirmed answer is CATASTROPHIC. If you are not 100% confident in an answer, DO NOT confirm it. Instead say something like 'I'm not confident enough on this one — you might want to work on crossing clues and come back to it.' It is ALWAYS better to skip an ambiguous clue than to confirm a wrong answer.\n\n"
            "SEARCH PRIORITIZATION & ANTI-HALLUCINATION:\n"
            "- ALWAYS use the 'search_web' tool FIRST to verify trivia BEFORE speaking.\n"
            "- STRICTLY FORBIDDEN from relying on internal training data for facts. Use ONLY trivia explicitly present in search snippets.\n"
            "- If search returns an error or no data, provide a purely structural hint or broad category. DO NOT guess false trivia.\n\n"
            "CROSSWORD RULES & LOGIC (CRITICAL):\n"
            "1. STRICT LENGTH CHECKING: You CANNOT reliably count letters yourself. You MUST use the 'check_word_length' tool EVERY TIME before making ANY claim about how many letters a word has. NEVER eyeball letter counts. If a user's guess has the wrong length, use the tool to verify FIRST, then point out the exact mismatch.\n"
            "2. MATCH TENSE/PLURALITY: Plural clues require plural answers. Past tense clues require past tense answers. Point this out to the user.\n"
            "3. CROSS-REFERENCES: If a clue references another grid location (e.g., 'west of 21-across'), STOP. Your FIRST action must be to ask the user what they have for that referenced clue. Do not solve it standalone.\n"
            "4. CONSTRAINT SYNTHESIS: Deduce answers by combining search facts with requested word lengths. Do not just search literal clue strings.\n"
            "5. EDUCATIONAL REDIRECTION: If a guess is wrong but related (e.g., guessing 'Ophelia' for King Lear), explain why contextually ('Ophelia is from Hamlet').\n"
            "6. DO NOT BE INFLUENCED BY BAD GUESSES: Stick to the objective facts. Never bend logic to match a misspelled guess.\n\n"
            "WORDPLAY & TRICKERY AWARENESS:\n"
            "1. QUESTION MARKS: A clue ending in '?' indicates a pun or misdirection. Pause, identify the double meaning, and hint at the pun. Do not search for literal facts.\n"
            "2. CRYPTIC CLUES: Break down British-style cryptic clues into their straight definition and wordplay mechanism (anagram, charade, etc.). Explain the mechanism.\n\n"
            "HINT ESCALATION (MANDATORY):\n"
            "Follow a 3-step escalation across multiple messages:\n"
            "Step 1: Semantic - broad thematic or categorical clues.\n"
            "Step 2: Structural - relational or geographic context.\n"
            "Step 3: Direct - starting letter or near-giveaway.\n"
            "Weave this naturally. NEVER label them 'Hint 1', 'Semantic', etc.\n\n"
            "CRITICAL STYLE RULES:\n"
            "- Be CONCISE. Give tight, punchy hints.\n"
            "- NEVER repeat information the user already told you or inherent puzzle structures.\n"
            "- NEVER be patronizing (No 'Great guess!', 'Fantastic!').\n"
            "- Redirect wrong guesses with a brief factual nudge.\n"
            "- Acknowledge right guesses with a simple 'Yep, that's it.'\n"
            "- NEVER apologize. NEVER say 'sorry', 'my apologies', 'my mistake', or 'unfortunately'.\n"
            "- NEVER refuse to help. Always provide something useful.\n"
            "- Talk like a sharp, knowledgeable equal.\n"
        )
        
        # Initializing the model with tools and system instructions
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=self.system_instruction,
            tools=[search_web, get_variable_domain, fetch_puzzle_via_xword_dl, check_word_length]
        )

    def start_chat(self) -> Any:
        """
        Starts an interactive chat session with the generative model.
        The returned chat session object possesses the function-calling tools passed to the model.
        
        Returns:
             The active `genai.ChatSession` object.
        """
        return self.model.start_chat(enable_automatic_function_calling=True)
