class BayesianNetwork:
    def __init__(self, cpt):
        # Initialize the Bayesian Network with Conditional Probability Tables (CPT).

        self.cpt = cpt

    def probability_of_admission(self, evidence):
        # Calculate P(A=a^1 | I=i^1, S=s^1, E=e^1)
        # Extract the necessary probabilities from the CPT
        P_E = self.cpt.get('P(E=e^1)', 0.3)
        P_I = self.cpt.get('P(I=i^1)', 0.2)
        P_S_given_I = self.cpt.get(f'P(S=s^1|I=i^1)', 0.6)

        # Get P(M|I, E) values for the given evidence
        P_M0_given_I_E = self.cpt.get(f'P(M=m^0|I=i^1,E=e^1)', 0.8)
        P_M1_given_I_E = self.cpt.get(f'P(M=m^1|I=i^1,E=e^1)', 0.2)

        # Get P(A|M) values
        P_A1_given_M0 = self.cpt.get(f'P(A=a^1|M=m^0)', 0.1)
        P_A1_given_M1 = self.cpt.get(f'P(A=a^1|M=m^1)', 0.1)

        # Calculate the joint probability for M and A
        P_A1 = P_M0_given_I_E * P_A1_given_M0 + P_M1_given_I_E * P_A1_given_M1

        # Calculate the final probability P(A=a^1 | I=i^1, S=s^1, E=e^1)
        probability_admission = P_I * P_S_given_I * P_E * P_A1
        return probability_admission

# Example CPT data
cpt_data = {
    "P(E=e^0)": 0.7, "P(E=e^1)": 0.3,
    "P(I=i^0)": 0.8, "P(I=i^1)": 0.2,
    "P(S=s^0|I=i^0)": 0.75, "P(S=s^1|I=i^0)": 0.25,
    "P(S=s^0|I=i^1)": 0.4, "P(S=s^1|I=i^1)": 0.6,
    "P(M=m^0|I=i^0,E=e^0)": 0.6, "P(M=m^1|I=i^0,E=e^0)": 0.4,
    "P(M=m^0|I=i^0,E=e^1)": 0.9, "P(M=m^1|I=i^0,E=e^1)": 0.1,
    "P(M=m^0|I=i^1,E=e^0)": 0.5, "P(M=m^1|I=i^1,E=e^0)": 0.5,
    "P(M=m^0|I=i^1,E=e^1)": 0.8, "P(M=m^1|I=i^1,E=e^1)": 0.2,
    "P(A=a^0|M=m^0)": 0.6, "P(A=a^1|M=m^0)": 0.4,
    "P(A=a^0|M=m^1)": 0.9, "P(A=a^1|M=m^1)": 0.1
}

# Define the evidence: High IQ, High Aptitude Score, Difficult Exam
evidence = {
    "I": "i^1",
    "S": "s^1",
    "E": "e^1"
}

# Initialize the Bayesian Network with the loaded CPT
bn = BayesianNetwork(cpt_data)

# Calculate the probability of admission given the evidence
probability_admission = bn.probability_of_admission(evidence)
print(f"Probability of Admission given high IQ, high aptitude score, and difficult exam: {probability_admission:.2f}")
