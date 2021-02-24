import remote_cirq
import pennylane as qml
from pennylane import numpy as np
import sys

from dotenv import load_dotenv
import os

num_wires = 8
num_shots = 20_000
#np.random.seed(0)

load_dotenv()
API_KEY = os.environ.get("FLOQ_KEY")

try:
    sim = remote_cirq.RemoteSimulator(API_KEY)
    dev = qml.device("cirq.simulator", wires=num_wires, simulator=sim, analytic=False, shots=num_shots)
    print("Using Floq simulator")
except:
    print("FLOQ_key not found in .env, using default.qubit simulator")
    dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)
    
#dev = qml.device("default.qubit", wires=num_wires, analytic=True, shots=num_shots)
######################

def ansatz(weights, num_wires):
    """U(theta) to prepare the state"""
    qml.templates.layers.StronglyEntanglingLayers(weights,
                                                  wires=range(num_wires))

@qml.qnode(dev)
def qcbm_probs(weights, num_wires):
    """Returns array of probabilities"""
    ansatz(weights, num_wires)
    return qml.probs(range(num_wires))

@qml.qnode(dev)
def qcbm_sample(weights, num_wires):
    """Returns many samples. Array of size (num_qubits, shots) """ 
    ansatz(weights,num_wires)
    return [qml.sample(qml.PauliZ(i)) for i in range(num_wires)]

######################

def KL_Loss(P, Q, eps=0.1/num_shots):
    """A loss function: Kullback-Leibler divergence AKA relative entropy between two probability distributions.
    Args:
        P, Q: arrays of shape (2**num_wires,), representing probability distributions
        eps: small value, prevents divergence in log if Q = 0"""

    mask = P > 0
    return np.sum(P[mask] * np.log(P[mask] / (Q[mask]+eps)) )

#######################

def to_num(x):
    '''Converts array of -1 and +1 to 0 and 1, then expresses as binary number
    Example: x = [1, -1, 1, -1, 1, 1] -> 101011 = 43'''

    bitstr = ''.join(['0' if i==-1 else '1' for i in x])
    return eval('0b' + bitstr)

def qcbm_approx_probs(weights, num_wires):
    """Approximate probabilities for the results. Alternative to qcbm_probs, which needs 2^num_qubit entries and would be
        large if we used 26~32 qubits
        Returns dictionary with the results as the keys and the probabilities as the values"""
    samples = qcbm_sample(weights, num_wires)
    num_samples = samples.shape[1]
    prob_dict = {}
    for i in range(num_samples):
        outcome = to_num(samples[:, i])
        prob_dict[outcome] = prob_dict.get(outcome, 0) + 1/num_samples
    
    return prob_dict

def KL_Loss_dict(p_dict, q_dict, eps=0.1/num_shots):
    """Kullback-Leibler divergence, but using dicts as arguments instead"""
    cost = 0.0
    for outcome, prob in p_dict.items():
        #print(outcome, prob, q_dict.get(outcome))
        cost += prob * np.log(prob / (q_dict.get(outcome, eps)))
    return cost

####################

def SPSA_grad(f, theta, delta=1E-2):
    """gradient of f at theta, approximated using
        Simultaneous Perturbation Stochastic Approximation"""

    dt = np.random.choice([-delta,+delta], theta.shape)
    return (f(theta+dt) - f(theta - dt) ) / dt

def particle_swarm_optim(f, init_params):
    """Find global minimum using the particle swarm optimization algorithm"""
    # Can also use Pyswarm library. This can perhaps give better results than SPSA
    pass

#########################
#########################
if __name__ == "__main__":
    #Test the model
    weights = np.random.random((5, num_wires, 3))
    
    print(f"num_wires: {num_wires}, num_shots: {num_shots}, num_layers: {weights.shape[0]}")

    #For testing, generate an exact probability distribution to learn
    exact_prob_dist = np.random.random(2**num_wires)
    exact_prob_dist /= np.sum(exact_prob_dist)
    
    #Convert the probability distribution into dict form
    exact_prob_dict = {outcome:exact_prob_dist[outcome] for outcome in range(2**num_wires)}

    #Two possible cost functions
    #########
    #Uses exact QCBM probability
    def exact_cost_fn(weights):
        return KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires))
    exact_grad_cost = qml.grad(exact_cost_fn, argnum=0)

    #Uses approximate QCBM probability from samples
    def approx_cost_fn(weights):
        return KL_Loss_dict(exact_prob_dict, qcbm_approx_probs(weights, num_wires))
    #########

    print("Training using approximate sample probabilities")
    for i in range(15_000):
        weights = weights - 0.01* SPSA_grad(approx_cost_fn, weights) #cost using approx sample probabilities
        #weights = weights - 0.01* exact_grad_cost(weights) #cost using exact sample probabilities
        if i % 100 == 0:
            #print("Approx Cost:", KL_Loss_dict(exact_prob_dict, qcbm_approx_probs(weights, num_wires)))
            print("True Cost:", KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires)))

    #current results: analytic probabilities do better than sampled probabilities. Maybe adjust eps?
    #SPSA is fast. will try particle swarm optimization next
    #Still hard to train using approximate probabilities, even with 50000 shots the cost doesn't get low
    #100_000 shots: cost gets down to 0.53, slow on Floq though
    
    #Even with exact, analytic gradient, lower limit to cost seems to be
    #Perhaps changing the ansatz may help?

    print(KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires)))
    print(qcbm_sample(weights, num_wires).shape)
    print(to_num(qcbm_sample(weights, num_wires)[:, 0]))
