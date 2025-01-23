import numpy as np
import json


class TreeGrowthRingDBN:
    def __init__(self, initial_state_dist, transition_matrix, emission_matrix):
        self.initial_state_dist = np.array(initial_state_dist)
        self.transition_matrix = np.array(transition_matrix)
        self.emission_matrix = np.array(emission_matrix)

    def viterbi(self, observations):
        T = len(observations)
        N = len(self.initial_state_dist)

        log_delta = np.zeros((T, N))
        log_delta[0] = np.log(self.initial_state_dist) + np.log(self.emission_matrix[:, observations[0]])

        psi = np.zeros((T, N), dtype=int)

        for t in range(1, T):
            for j in range(N):
                trans_probs = log_delta[t - 1] + np.log(self.transition_matrix[:, j])
                log_delta[t, j] = np.max(trans_probs) + np.log(self.emission_matrix[j, observations[t]])
                psi[t, j] = np.argmax(trans_probs)

        path = [np.argmax(log_delta[T - 1])]
        for t in range(T - 1, 0, -1):
            path.insert(0, psi[t, path[0]])

        return path

    def compute_state_probabilities(self, observations):
        T = len(observations)
        N = len(self.initial_state_dist)

        alpha = np.zeros((T, N))
        alpha[0] = self.initial_state_dist * self.emission_matrix[:, observations[0]]
        alpha[0] /= np.sum(alpha[0])

        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = np.sum(alpha[t - 1] * self.transition_matrix[:, j]) * self.emission_matrix[
                    j, observations[t]]
            alpha[t] /= np.sum(alpha[t])

        return alpha


def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data['observations']


def main():
    initial_state_dist = [0.6, 0.4]

    transition_matrix = [
        [0.6, 0.4],
        [0.3, 0.7]
    ]

    emission_matrix = [
        [0.7, 0.2, 0.1],
        [0.1, 0.4, 0.5]
    ]

    dbn = TreeGrowthRingDBN(initial_state_dist, transition_matrix, emission_matrix)

    try:
        observations = load_data('tree_observations.json')
    except FileNotFoundError:
        observations = [0, 1, 2, 1, 0]

    most_likely_temps = dbn.viterbi(observations)
    print("Most Likely Temperature Sequence:",
          ['Cold' if t == 0 else 'Hot' for t in most_likely_temps])

    state_probs = dbn.compute_state_probabilities(observations)
    print("\nState Probabilities:")
    for t, probs in enumerate(state_probs):
        print(f"Time {t}: Cold = {probs[0]:.2f}, Hot = {probs[1]:.2f}")


if __name__ == "__main__":
    main()