import numpy as np

probabilities = {
    'VIN': {'220V': 0.8, '110V': 0.2},
    'TAMB': {'23C': 0.7, '30C': 0.3},
    'STransformer': {'Working': 0.9, 'Defective': 0.1},
    'SRectifier': {'Working': 0.85, 'Defective': 0.15},
    'VOUT': {'normal': 0.6, 'low': 0.25, 'high': 0.15},
    'IOUT': {'normal': 0.5, 'low': 0.3, 'high': 0.2},
    'OPT': {'on': 0.4, 'off': 0.6},
    'NL': {'low': 0.7, 'high': 0.3},
    'OT': {'normal': 0.65, 'high': 0.35}
}

initial_state = {
    'VIN': '220V',
    'TAMB': '23C',
    'STransformer': 'Working',
    'SRectifier': 'Working',
    'VOUT': 'normal',
    'IOUT': 'normal',
    'OPT': 'off',
    'NL': 'low',
    'OT': 'normal'
}

def markov_blanket(state, node):
    if node == 'IOUT':
        return ['VOUT', 'OPT', 'NL', 'OT']
    return []

def sample_given_blanket(state, node):
    if node not in probabilities:
        print(f"Warning: No probabilities found for node '{node}'. Skipping.")
        return state[node]

    blanket = markov_blanket(state, node)
    node_probabilities = probabilities[node]
    values, probs = zip(*node_probabilities.items())
    probs = np.array(probs) / np.sum(probs)
    return np.random.choice(values, p=probs)

def mcmc_sampling(state, target_node, iterations=1000):
    samples = {target_node: []}

    for _ in range(iterations):
        for node in state:
            state[node] = sample_given_blanket(state, node)
        samples[target_node].append(state[target_node])

    unique, counts = np.unique(samples[target_node], return_counts=True)
    posterior_distribution = dict(zip(unique, counts / iterations))

    return posterior_distribution

posterior_iout = mcmc_sampling(initial_state, 'IOUT')
posterior_iout_readable = {str(k): float(v) for k, v in posterior_iout.items()}
print("Posterior distribution for IOUT:", posterior_iout_readable)