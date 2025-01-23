#Drawing of Bayesian Network

# W---->A---->C<----E

class BayesianNetwork:
    def __init__(self, cpt):
        # Initialize the Bayesian Network with Conditional Probability Tables (CPT).
        self.cpt = cpt

    def probability_of_congestion(self, evidence):
        # Calculate P(C=True | W=True, E=True)
        # Extract the necessary probabilities from the CPT
        P_W = self.cpt.get('P(W=True)', 0)
        P_E = self.cpt.get('P(E=True)', 0)
        P_A_given_W = self.cpt.get(f'P(A=True|W={evidence["W"]})', 0)

        P_C_given_E_A = self.cpt.get(f'P(C=True|E={evidence["E"]},A=True)', 0)
        P_C_given_E_not_A = self.cpt.get(f'P(C=True|E={evidence["E"]},A=False)', 0)

        # Calculate the probability of an accident occurring
        P_A = P_A_given_W

        # Calculate the probability of C=True given evidence
        P_C = P_A * P_C_given_E_A + (1 - P_A) * P_C_given_E_not_A
        return P_C

# Example CPT data
cpt_data = {
    "P(W=True)": 0.3,
    "P(E=True)": 0.2,
    "P(A=True|W=True)": 0.6,
    "P(A=True|W=False)": 0.1,
    "P(C=True|E=True,A=True)": 0.9,
    "P(C=True|E=True,A=False)": 0.7,
    "P(C=True|E=False,A=True)": 0.5,
    "P(C=True|E=False,A=False)": 0.2
}

# Define evidence: it's raining and there's a major event in the city
evidence = {
    "W": True,
    "E": True
}

# Initialize the Bayesian Network with the loaded CPT
bn = BayesianNetwork(cpt_data)

# Calculate the probability of congestion given the evidence
probability_congestion = bn.probability_of_congestion(evidence)
print(f"Probability of Congestion given that it is raining and there is a major event: {probability_congestion:.2f}")
