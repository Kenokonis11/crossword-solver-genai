from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Cell:
    """
    Represents a single cell in the crossword grid. Supports Rebus puzzles by allowing
    multiple characters in a single cell.
    """
    value: List[str] = field(default_factory=lambda: [""])

@dataclass
class Candidate:
    """
    Represents a potential candidate answer for a Variable.
    Contains the text, its computed confidence score, and the source of the candidate.
    """
    text: str
    score: float
    source: str

@dataclass
class Variable:
    """
    Represents a single clue in the crossword puzzle (e.g., '1A').
    Stores the clue text itself, the length of the expected word, the current domain of possible Candidates,
    and a locked Candidate if the algorithm has committed to an answer.
    """
    id: str
    length: int
    clue_text: str = ""
    domain: List[Candidate] = field(default_factory=list)
    locked_candidate: Optional[Candidate] = None

@dataclass
class DecisionSnapshot:
    """
    Represents a decision state in the CSP's decision stack.
    Stores the specific variable evaluated, the candidate chosen, and a deep copy of all
    variable domains at the time of the decision to handle backtracking.
    """
    variable_id: str
    chosen_candidate: Candidate
    domain_snapshot: Dict[str, List[Candidate]]
