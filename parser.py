import puz
from typing import Dict, List, Tuple

from models import Variable, Candidate

def parse_puz_file(file_bytes: bytes) -> Tuple[Dict[str, Variable], Dict[str, List[Tuple[str, int, int]]]]:
    """
    Parses a binary .puz file and constructs the foundational Variable and Intersection maps
    required by the Constraint Satisfaction Problem (CSP) Engine.

    The primary complexity of this function is mathematically mapping the 2D crossword grid
    into a 1D intersection dictionary format where Across words and Down words are linked by
    their relative 0-based string indices.

    Args:
        file_bytes (bytes): The raw bytes of the uploaded .puz file.

    Returns:
        tuple: (variables, intersections)
               - variables: Dict mapping clue ID (e.g. '1A') to a instantiated Variable object.
               - intersections: Dict mapping a variable ID to a list of Tuples representing constraints.
                 Tuple structure: (intersecting_var_id, index_in_this_var, index_in_other_var).
    """
    # Load the binary .puz file string
    p = puz.load(file_bytes)
    
    variables: Dict[str, Variable] = {}
    intersections: Dict[str, List[Tuple[str, int, int]]] = {}

    width = p.width
    height = p.height
    
    # 2D Grids to store the ID ('1A', '5D') and the word length of a cell
    across_grid = [[None] * width for _ in range(height)]
    down_grid = [[None] * width for _ in range(height)]
    
    # Track assigned clue numbers
    assignments = p.clue_numbering()
    
    # --- PHASE 1: Variable Extraction and Initial Grid Mapping ---
    for clue in assignments.across:
        num = clue["num"]
        clue_text = clue["clue"]
        var_id = f"{num}A"
        length = clue["len"]
        
        # Create Variable and initialize Intersection list
        variables[var_id] = Variable(id=var_id, length=length, clue_text=clue_text)
        intersections[var_id] = []
        
        # Map the cell ownership across the 2D grid
        start_cell = clue["cell"]
        x = start_cell % width
        y = start_cell // width
        for offset in range(length):
            across_grid[y][x + offset] = (var_id, offset)

    for clue in assignments.down:
        num = clue["num"]
        clue_text = clue["clue"]
        var_id = f"{num}D"
        length = clue["len"]
        
        # Create Variable and initialize Intersection list
        variables[var_id] = Variable(id=var_id, length=length, clue_text=clue_text)
        intersections[var_id] = []
        
        # Map the cell ownership down the 2D grid
        start_cell = clue["cell"]
        x = start_cell % width
        y = start_cell // width
        for offset in range(length):
            down_grid[y + offset][x] = (var_id, offset)

    # --- PHASE 2: Intersection Calculus ---
    # Iterate through the entire 2D grid to find intersecting words.
    for y in range(height):
        for x in range(width):
            across_cell = across_grid[y][x]
            down_cell = down_grid[y][x]
            
            # An intersection occurs mathematically if a cell is claimed by both Across and Down words.
            if across_cell and down_cell:
                across_id, across_index = across_cell
                down_id, down_index = down_cell
                
                # Add the relative constraint mapping for the Across word
                intersections[across_id].append((down_id, across_index, down_index))
                
                # Add the reciprocal constraint mapping for the Down word
                intersections[down_id].append((across_id, down_index, across_index))

    return variables, intersections
