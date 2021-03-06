import remote_cirq
import pennylane as qml
from pennylane import numpy as np
import sys
from loss_functions import KL_Loss, KL_Loss_dict
from particle_swarm import optim_particle_swarm

from dotenv import load_dotenv
import os

def to_num(x):
    '''Converts array of -1 and +1 to 0 and 1, then expresses as binary number
    Example: x = [1, -1, 1, -1, 1, 1] -> 101011 = 43'''

    bitstr = ''.join(['0' if i==-1 else '1' for i in x])
    return eval('0b' + bitstr)

def template_ansatz(weights, num_wires):
    """U(theta) to prepare the state"""
    qml.templates.layers.StronglyEntanglingLayers(weights,
                                                wires=range(num_wires))

def paper_ansatz(weights, num_wires):
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

def prob_array_to_dict(prob_array):
    return {outcome: prob_array[outcome] for outcome in range(prob_array.size) if prob_array[outcome] > 0}

##############################################

class QCBM():
    """Class for the QCBM"""

    def __init__(self, num_layers, num_wires, num_shots, Floq=True, ansatz=paper_ansatz):
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.weights = initialize_weights(num_layers, num_wires)
        self.num_shots = num_shots

        if Floq:
            try:
                load_dotenv()
                API_KEY = os.environ.get("FLOQ_KEY")
                sim = remote_cirq.RemoteSimulator(API_KEY)
                self.dev = qml.device("cirq.simulator", wires=num_wires, simulator=sim, analytic=False, shots=num_shots)
                print("Using Floq simulator")
                self._use_exact_probs = False

            except:
                print("FLOQ_key not found in .env, using default.qubit simulator. Note that probabilities used will be exact instead of sampled.")
                self.dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)
                self._use_exact_probs = True

        else:
            print("Using default.qubit simulator. Note that probabilities used will be exact instead of sampled.")
            self.dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)
            self._use_exact_probs = True

        #Perhaps another option to use default.qubit but with sampled probabilities?

        def qcbm_probs(weights, num_wires):
            """Returns array of probabilities"""
            ansatz(weights, num_wires)
            return qml.probs(range(num_wires))

        def qcbm_sample(weights, num_wires):
            """Returns many samples. Array of size (num_qubits, shots) """ 
            ansatz(weights, num_wires)
            return [qml.sample(qml.PauliZ(i)) for i in range(num_wires)]

        self._qcbm_probs = qml.qnode(self.dev)(qcbm_probs)
        self._qcbm_sample = qml.qnode(self.dev)(qcbm_sample)
        

    def _qcbm_approx_probs(self, params):
        """Approximate probabilities by sampling.
            Returns dictionary with the results as the keys and the probabilities as the values"""
        samples = self._qcbm_sample(params, self.num_wires)
        num_samples = samples.shape[1]
        prob_dict = {}
        for i in range(num_samples):
            outcome = to_num(samples[:, i])
            prob_dict[outcome] = prob_dict.get(outcome, 0) + 1/num_samples
        
        return prob_dict

    #Interface:

    def return_samples(self):
        """Samples in shape (num_shots, num_qubits)"""

        return self._qcbm_sample(self.weights, self.num_wires).T

    def return_probabilities(self):
        """Probability array. Sampled if simulator has analytic=False, otherwise exact"""

        return self._qcbm_probs(self.weights, self.num_wires)

    def return_approx_prob_dict(self):
        return self._qcbm_approx_probs(self.weights, self.num_wires)


    #Optimize KL divergence from a probability distribution
    #n_processes is an argument in PySwarms optimize function. Use >= 2 for parallelism
    def train_on_prob_dict(self, prob_dict, iters=10, n_processes=None):
        
        epsilon = 1/self.num_shots
        if not self._use_exact_probs:
            def _cost_fn(params):
                return KL_Loss_dict(prob_dict, self._qcbm_approx_probs(params), eps=epsilon)
        else:
            def _cost_fn(params):
                q_dict = prob_array_to_dict(self._qcbm_probs(params, self.num_wires))
                return KL_Loss_dict(prob_dict, q_dict, eps=epsilon)

        self.weights = optim_particle_swarm(_cost_fn, self.weights.shape, num_particles=12, iters=iters, init_weight=self.weights)
    
    def train_on_prob_array(self, prob_array, iters=10, n_processes=None):
        
        epsilon = 1/self.num_shots
        def _cost_fn(weights):
            return KL_Loss(prob_array, self._qcbm_probs(weights, self.num_wires), eps=epsilon)

        self.weights = optim_particle_swarm(_cost_fn, self.weights.shape, num_particles=12, iters=iters, init_weight=self.weights)


    def train_qcbm_grad_descent(self, exact_prob_dist, iters=500):
        """Train the QCBM"""
        exact_prob_dict = {outcome:exact_prob_dist[outcome] for outcome in range(2**num_wires)}
        def approx_cost_fn(weights):
            return KL_Loss_dict(exact_prob_dict, qcbm_approx_probs(weights, num_wires))
        
        weights = self.weights
        for i in range(iters):
            weights = weights - 0.01* SPSA_grad(approx_cost_fn, weights) #cost using approx sample probabilities
            #weights = weights - 0.01* exact_grad_cost(weights) #cost using exact sample probabilities
            if i % 100 == 0:
                #print("Approx Cost:", KL_Loss_dict(exact_prob_dict, qcbm_approx_probs(weights, num_wires)))
                print("True Cost:", KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires)))
        self.weights = weights



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


if __name__ == "__main__":
    #Test the model
    #Generate an exact probability distribution to test on

    exact_prob_dist = np.zeros(256)
    exact_prob_dist[0:64] = 1/64
    exact_prob_dict = {i:1/64 for i in range(64)}

    qcbm = QCBM(7, 8, 1000, True)
    qcbm.train_on_prob_dict(exact_prob_dict, iters=100, n_processes=2)

