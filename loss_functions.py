from pennylane import numpy as np

def KL_Loss(P, Q, eps=1E-4):
    """A loss function: Kullback-Leibler divergence AKA relative entropy between two probability distributions.
    Args:
        P, Q: arrays of shape (2**num_wires,), representing probability distributions
        eps: small value, prevents divergence in log if Q = 0"""

    mask = P > 0
    return np.sum(P[mask] * np.log(P[mask] / (Q[mask]+eps)) )

def KL_Loss_dict(p_dict, q_dict, eps=1E-4):
    """Kullback-Leibler divergence, but using dicts as arguments instead"""
    cost = 0.0
    for outcome, prob in p_dict.items():
        cost += prob * np.log(prob / (q_dict.get(outcome,0)+eps))
    return cost

def LL_Loss(P, Q, eps=1E-4):
    """Log-likelihood cost
    Args:
        P, Q: arrays of shape (2**num_wires,), representing probability distributions
        eps: small value, prevents divergence in log if Q = 0"""

    mask = P > 0
    return np.sum(P[mask] * np.log(P[mask] / (Q[mask]+eps)) )

def LL_Loss_dict(p_dict, q_dict, eps=1E-4):
    """Log-likelihood cost, but using dicts as arguments instead"""
    cost = 0.0
    for outcome, prob in p_dict.items():
        cost -= np.log(q_dict.get(outcome,0)+eps)
    return cost