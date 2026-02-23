import re
from difflib import SequenceMatcher
from typing import List, Dict

from models import Candidate

def evaluate_web_candidates(clue_text: str, search_snippets: List[str], regex_pattern: str) -> List[Candidate]:
    """
    Extracts potential candidate answers from text snippets based on a given regex structure,
    scoring each word using semantic similarity to the original clue text.
    
    Args:
        clue_text (str): The text of the crossword clue.
        search_snippets (List[str]): Extracted text blocks/snippets from a web search or local parsing.
        regex_pattern (str): The regex pattern strictly defining the length and known characters
            of the underlying puzzle cell (e.g., '^P...S$').

    Returns:
        List[Candidate]: A sorted list of Potential Candidates, prioritized descending by their
            semantic similarity score.
    """
    compiled_regex = re.compile(regex_pattern, re.IGNORECASE)
    candidate_scores: Dict[str, float] = {}

    for snippet in search_snippets:
        # Strip simple punctuation to isolate distinct web words
        clean_snippet = re.sub(r'[^\w\s]', '', snippet)
        
        words = clean_snippet.split()
        
        for word in words:
            # Enforce the regex structure (e.g. valid length and crossing constraints)
            if compiled_regex.match(word):
                # Calculate simple semantic similarity between the clue and the full snippet the word came from
                similarity_score = SequenceMatcher(None, clue_text, snippet).ratio()
                
                word_upper = word.upper() # Standardize crosswords to uppercase
                
                if word_upper in candidate_scores:
                    candidate_scores[word_upper] = max(candidate_scores[word_upper], similarity_score)
                else:
                    candidate_scores[word_upper] = similarity_score

    # Convert mapping dictionary into Candidate objects, sorted descending by the computed Score
    results = [
        Candidate(text=word, score=score, source="web_pipeline") 
        for word, score in candidate_scores.items()
    ]
    
    results.sort(key=lambda x: x.score, reverse=True)
    
    return results
