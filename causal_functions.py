"""
Functions and Classes used in analysis of Causal Inference Research
"""

# Imports

def inv_log_odds(LO):
    """
    args: LO , float or list of floats: representing Log Odds
    returns inverse logit of LO
    """
    return 1/(1+ (np.exp(-LO)))