import copy
from typing import Dict, List, Tuple, Optional

from models import Variable, Candidate, DecisionSnapshot

class CSPSolver:
    """
    Core solver engine applying Constraint Satisfaction Problem logic to crossword puzzle grids.
    Implements Minimum Remaining Values (MRV) selection and Arc Consistency (AC-3) inference.
    """
    def __init__(self, variables: Dict[str, Variable], intersections: Dict[str, List[Tuple[str, int, int]]]):
        """
        Initializes the CSP Solver.
        
        Args:
            variables (Dict[str, Variable]): A dictionary mapping a variable ID to its Variable object.
            intersections (Dict[str, List[Tuple[str, int, int]]]): Maps a variable ID to a list of its intersections.
                Each tuple contains:
                (intersecting_var_id, index_in_this_var, index_in_other_var).
        """
        self.variables = variables
        self.intersections = intersections
        self.decision_stack: List[DecisionSnapshot] = []

    def get_mrv_variable(self) -> Optional[Variable]:
        """
        Selects the next variable to process using the Minimum Remaining Values (MRV) heuristic.
        It finds the unlocked variable with the smallest candidate domain.
        
        Returns:
            Optional[Variable]: The unlocked Variable with the fewest remaining candidates,
            or None if all variables have been locked.
        """
        mrv_var = None
        min_domain_size = float('inf')

        for var in self.variables.values():
            if var.locked_candidate is None:
                domain_size = len(var.domain)
                if domain_size < min_domain_size:
                    min_domain_size = domain_size
                    mrv_var = var

        return mrv_var

    def apply_arc_consistency(self, target_var_id: str, locked_candidate: Candidate) -> bool:
        """
        Applies Arc Consistency by pruning the domains of all intersecting variables that violate
        the newly locked candidate's characters.
        
        Args:
            target_var_id (str): The ID of the variable that was just assigned a candidate.
            locked_candidate (Candidate): The candidate that was firmly assigned to the target variable.
            
        Returns:
            bool: True if arc consistency was maintained. False if any intersecting variable's
            domain becomes empty (0 candidates), indicating a fatal constraint conflict.
        """
        if target_var_id not in self.intersections:
            return True

        for intersect_id, idx_self, idx_other in self.intersections[target_var_id]:
            intersect_var = self.variables[intersect_id]
            
            # We only prune domains of variables that are not yet locked.
            if intersect_var.locked_candidate is not None:
                continue
            
            required_char = locked_candidate.text[idx_self]
            
            # Filter the intersecting variable's domain
            intersect_var.domain = [
                cand for cand in intersect_var.domain 
                if len(cand.text) > idx_other and cand.text[idx_other] == required_char
            ]
            
            # If the domain is empty after filtering, a conflict occurred.
            if len(intersect_var.domain) == 0:
                return False
                
        return True

    def backtrack(self) -> bool:
        """
        Reverses the most recent algorithmic decision by popping the last DecisionSnapshot
        and restoring the domains of all variables to their pre-decision state.
        
        The candidate that led to the conflict is permanently removed from the evaluated
        variable's domain to prevent infinite loops.
        
        Returns:
            bool: True if the engine successfully backtracked. False if the decision stack
            is empty, indicating the puzzle is logically unsolvable from the initial state.
        """
        if not self.decision_stack:
            return False
            
        snapshot = self.decision_stack.pop()
        
        # Restore the deep copy of all domains
        for var_id, saved_domain in snapshot.domain_snapshot.items():
            self.variables[var_id].domain = copy.deepcopy(saved_domain)
            
        # The variable that caused the conflict
        conflict_var = self.variables[snapshot.variable_id]
        conflict_var.locked_candidate = None
        
        # Remove the failed candidate from the domain
        conflict_var.domain = [
            cand for cand in conflict_var.domain 
            if cand.text != snapshot.chosen_candidate.text
        ]
        
        return True

    def generate_regex_for_variable(self, var_id: str) -> str:
        """
        Generates a strict regular expression pattern dynamically defining the known layout
        of a variable based on the already locked cells of its intersecting words.
        
        Args:
            var_id (str): The ID of the target variable (e.g., '1A').
            
        Returns:
            str: A Regex string using '^' and '$' anchors and '.' for unknown characters.
        """
        target_var = self.variables[var_id]
        pattern = ['.'] * target_var.length
        
        if var_id in self.intersections:
            for intersect_id, idx_self, idx_other in self.intersections[var_id]:
                intersect_var = self.variables[intersect_id]
                
                # If the intersecting word has a committed answer, pin that character into our regex
                if intersect_var.locked_candidate is not None:
                    locked_char = intersect_var.locked_candidate.text[idx_other]
                    pattern[idx_self] = locked_char
                    
        return f"^{''.join(pattern)}$"
