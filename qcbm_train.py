import remote_cirq
import pennylane as qml
from pennylane import numpy as np
import sys

from dotenv import load_dotenv
import os

num_layers = 7
num_wires = 8
num_shots = 10_000

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

def to_num(x):
    '''Converts array of -1 and +1 to 0 and 1, then expresses as binary number
    Example: x = [1, -1, 1, -1, 1, 1] -> 101011 = 43'''

    bitstr = ''.join(['0' if i==-1 else '1' for i in x])
    return eval('0b' + bitstr)

######################

def template_ansatz(weights, num_wires):
    """U(theta) to prepare the state"""
    qml.templates.layers.StronglyEntanglingLayers(weights,
                                                  wires=range(num_wires))

def ansatz(weights, num_wires):
    """QCBM ansatz from https://arxiv.org/abs/2012.03924"""
    for layer in range(len(weights)):
        if layer == 0:
            for q in range(num_wires):
                qml.RX(weights[layer][q][0], wires=q)
                qml.RZ(weights[layer][q][1], wires=q)
        elif layer % 2 == 1:
            for q1 in range(num_wires):
                for q2 in range(q1+1, num_wires):
                    qml.CNOT(wires=[q1, q2])
                    qml.RX(weights[layer][q1][q2-q1-1], wires=q1)
                    qml.CNOT(wires=[q1, q2])
        else:
            for q in range(num_wires):
                qml.RZ(weights[layer][q][0], wires=q)
                qml.RX(weights[layer][q][1], wires=q)

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

####################

def SPSA_grad(f, theta, delta=1E-3):
    """gradient of f at theta, approximated using
        Simultaneous Perturbation Stochastic Approximation"""

    dt = np.random.choice([-delta,+delta], theta.shape)
    return (f(theta+dt) - f(theta - dt) ) / dt

#########################
    
def initialize_weights(layers, num_wires):
    """Initialize weights for the QCBM ansatz"""
    a = []
    for l in range(layers):
        a.append([])
        if l % 2 == 0:
            for _ in range(num_wires):
                a[l].append([np.random.random()*np.pi*2, np.random.random()*np.pi*2])
        else:
            for i in range(num_wires-1):
                a[l].append([np.random.random()*np.pi*2 for _ in range(num_wires-1-i)])
    return np.array(a)

if __name__ == "__main__":
    from loss_functions import KL_Loss, KL_Loss_dict

    #Test the model
    weights = initialize_weights(num_layers, num_wires)
    print(f"num_wires: {num_wires}, num_shots: {num_shots}, num_layers: {num_layers}")

    #Generate an exact probability distribution to learn
    exact_prob_dist = np.random.random(2**num_wires)
    exact_prob_dist /= np.sum(exact_prob_dist)
    
    #Convert probability distribution into dict form
    exact_prob_dict = {outcome:exact_prob_dist[outcome] for outcome in range(2**num_wires)}
    #Generate an exact probability distribution to test on

    exact_prob_dist = np.zeros(256)
    exact_prob_dist[0:64] = 1/64
    exact_prob_dict = {i:1/64 for i in range(64)}

    #Exact QCBM probability
    def exact_cost_fn(weights):
        return KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires))
    exact_grad_cost = qml.grad(exact_cost_fn, argnum=0)

    #Approximate QCBM probability, from sampling
    def approx_cost_fn(weights):
        return KL_Loss_dict(exact_prob_dict, qcbm_approx_probs(weights, num_wires))

    print("Training using approximate sample probabilities")
    for i in range(1000):
        weights = weights - 0.01* SPSA_grad(approx_cost_fn, weights) #Approximate gradient, approx. probabilities
        #weights = weights - 0.01* exact_grad_cost(weights) #Exact gradient, exact probabilities
        
        if i % 10 == 0:
            print("Exact cost:", KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires)))

    print("Final exact cost:", KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires)))
