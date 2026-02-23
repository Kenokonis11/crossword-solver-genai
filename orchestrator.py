import copy
from typing import List

from csp_engine import CSPSolver
from web_pipeline import evaluate_web_candidates
from scoring import calculate_bayesian_score, get_dynamic_threshold
from models import DecisionSnapshot, Candidate
from llm_agent import search_web

def run_solver_loop(solver: CSPSolver) -> bool:
    """
    The main autonomous execution loop for the Neuro-Symbolic Crossword CSP Engine.
    
    It continuously uses MRV to select the best variable, generates its strict regex layout constraints,
    fetches and mathematically evaluates candidates, and tests assignments using Arc Consistency.
    If a dead end is reached or Arc Consistency fails, it autonomously triggers the backtrack mechanism
    until a solution is found or it determines the puzzle is unsolvable.
    
    Args:
        solver (CSPSolver): The initialized stateful CSP solving engine containing grid mappings.
        
    Returns:
        bool: True if the entire crossword grid has been successfully solved. False if unsolvable.
    """
    while True:
        # Step 1: Select the next easiest unassigned variable (Minimum Remaining Values)
        var = solver.get_mrv_variable()
        
        # If no variables are returned, all are locked and the puzzle is mathematically solved.
        if var is None:
            return True
            
        # Step 2: Compile the rigorous regex constraint based on currently locked intersecting variables
        regex = solver.generate_regex_for_variable(var.id)
        
        # Step 3: Fetch candidate snippets from the web
        snippets = [search_web(var.clue_text)]
        
        # Step 4: Extract and semantically rank potential variable candidates from the raw snippets
        candidates = evaluate_web_candidates(var.clue_text, snippets, regex)
        
        candidate_was_locked = False
        
        # Step 5: Iteratively evaluate Candidates against Bayesian Thresholds & CSP Consistency
        for cand in candidates:
            cross_matches = 0
            total_crossings = 0
            
            # Sub-Step 5a: Assess the structural alignment of the Candidate
            if var.id in solver.intersections:
                for intersect_id, idx_self, idx_other in solver.intersections[var.id]:
                    intersect_var = solver.variables[intersect_id]
                    total_crossings += 1
                    
                    if intersect_var.locked_candidate is not None:
                        # Character matches the already locked intersecting variable
                        if cand.text[idx_self] == intersect_var.locked_candidate.text[idx_other]:
                            cross_matches += 1
                            
            # Sub-Step 5b: Calculate Multiplicative Bayesian score
            score = calculate_bayesian_score(
                base_prior=0.5,
                crossing_matches=cross_matches,
                total_crossings=total_crossings,
                web_likelihood=cand.score
            )
            
            # Sub-Step 5c: Compute Dynamic lock threshold (grid ratio fixed to 0.5 for now)
            threshold = get_dynamic_threshold(base_threshold=0.3, grid_fill_ratio=0.5)
            
            # Sub-Step 5d: Execute Assignment if Mathematical Confidence Threshold is breached
            if score >= threshold:
                # 1. State Capture: Save deep copy snapshot before any mutations
                snapshot_domains = {v_id: copy.deepcopy(v.domain) for v_id, v in solver.variables.items()}
                
                snapshot = DecisionSnapshot(
                    variable_id=var.id,
                    chosen_candidate=cand,
                    domain_snapshot=snapshot_domains
                )
                solver.decision_stack.append(snapshot)
                
                # 2. Assignment: Propose locking the candidate
                var.locked_candidate = cand
                
                # 3. Validation: Apply Arc Consistency to check for downstream conflict consequences
                is_consistent = solver.apply_arc_consistency(var.id, cand)
                
                if is_consistent:
                    candidate_was_locked = True
                    break # Success: Move on to the next Variable
                else:
                    # Conflict: Revert the arc consistency pruning and the assignment
                    solver.backtrack()

        # Step 6: Dead End Recovery logic
        # If no candidates met the threshold or survived Arc Consistency (or candidates list was entirely empty)
        if not candidate_was_locked:
            # Revert the *previous* structurally flawed decision
            successful_backtrack = solver.backtrack()
            
            if not successful_backtrack:
                 # If backtracking fails because the stack is empty, the puzzle is structurally unsolvable from root
                 return False
