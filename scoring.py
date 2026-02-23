def calculate_bayesian_score(base_prior: float, crossing_matches: int, total_crossings: int, web_likelihood: float, theme_likelihood: float = 1.0) -> float:
    """
    Calculates the confidence score of a candidate using a Multiplicative Bayesian Scoring formula.
    It combines the base probability, how well the candidate fits the grid crossings mathematically,
    web-fetched semantic confidence, and the alignment to the crossword theme.
    
    If there are no intersecting letters (total_crossings = 0), the crossing multiplier defaults to 1.0.
    
    Args:
        base_prior (float): The initial base probability of the candidate.
        crossing_matches (int): The number of intersecting letters matched correctly.
        total_crossings (int): The total number of current intersecting crossings.
        web_likelihood (float): Semantic confidence derived from the Web Semantic Pipeline.
        theme_likelihood (float, optional): Score matching the candidate to the crossword theme. Defaults to 1.0.

    Returns:
        float: The final compound Bayesian confidence score.
    """
    crossing_multiplier = 1.0
    if total_crossings > 0:
        crossing_multiplier = (crossing_matches / total_crossings) ** 2
        
    return base_prior * crossing_multiplier * web_likelihood * theme_likelihood

def get_dynamic_threshold(base_threshold: float, grid_fill_ratio: float) -> float:
    """
    Computes the dynamic threshold required for a candidate to be automatically locked based on grid progress.
    As the grid fills up, a stronger confidence score is required to lock candidates.
    
    Args:
        base_threshold (float): The baseline confidence threshold required to lock a candidate.
        grid_fill_ratio (float): The completion fraction of the crossword puzzle. Must be between 0.0 and 1.0.

    Returns:
        float: The dynamic threshold for candidate lock-in.
    """
    clamped_ratio = max(0.0, min(1.0, grid_fill_ratio))
    return base_threshold + (0.2 * clamped_ratio)
