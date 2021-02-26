import pyswarms as ps
from math import prod
from pennylane import numpy as np

"""Some code to wrap the particle swarm optimization from PySwarms"""

def optim_particle_swarm(cost, param_shape, num_particles=12, iters=100, init_weight=None, n_processes=None):
    """Optimizes the parameters with PySwarms particle swarm optimization
        Args:
            cost: cost function, takes in a numpy array and outputs a scalar
            param_shape: shape of the argument of cost
            iters: number of iterations
            init_weights: optional, initial position for the optimization
        Returns:
            params: parameters with lowest cost found by optimizer
    """
    size = prod(param_shape)

    def f(x):
        """Function to pass into pyswarms"""
        cost_list = [cost(x[i].reshape(param_shape)) for i in range(num_particles)]
        return np.array(cost_list)

    if init_weight != None:
        init_weight = np.broadcast_to(init_weight.reshape(-1), (num_particles, size))

    options = {'c1': 0.5, 'c2': 0.5, 'w':0.5}
    optimizer = ps.single.GlobalBestPSO(
                                n_particles=num_particles, dimensions=size,
                                options=options, init_pos=init_weight)
    final_cost, pos = optimizer.optimize(f, iters=iters, n_processes=n_processes, verbose=True)
    return pos.reshape(param_shape)






if __name__ == "__main__":
    #Tests
    from loss_functions import KL_Loss, KL_Loss_dict

    #Generate an exact probability distribution to test on
    exact_prob_dict = {i:1/64 for i in range(64)}
    
    from qcbm_train import *
    def exact_cost_fn(weights):
        """Loss using exact QCBM probabilities"""
        return KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires))

    def approx_cost_fn(weights):
        """Loss using QCBM probabilities estimated from sampling"""
        return KL_Loss_dict(exact_prob_dict, qcbm_approx_probs(weights, num_wires))

    weights = initialize_weights(num_layers, num_wires)
    print(weights.shape)
    print(f"num_wires: {num_wires}, num_shots: {num_shots}, num_layers: {num_layers}")
    best_weights = optim_particle_swarm(approx_cost_fn, weights.shape, iters=100, init_weight=weights)

    print("Final exact cost:", KL_Loss(exact_prob_dist, qcbm_probs(weights, num_wires)))