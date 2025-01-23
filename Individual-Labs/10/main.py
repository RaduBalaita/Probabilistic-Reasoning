import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


class StudentPerformanceCBN:
    def __init__(self):
        # Define the network structure
        self.model = BayesianNetwork([
            ('PAP', 'SH'),
            ('MH', 'A'),
            ('ES', 'MH'),
            ('SH', 'Performance'),
            ('A', 'Performance'),
            ('MH', 'Performance'),
            ('PAP', 'Performance')
        ])

        # Create CPDs with correct shape
        cpds = [
            self._create_cpd('PAP', 3, [[0.3], [0.4], [0.3]]),
            self._create_cpd('ES', 3, [[0.4], [0.3], [0.3]]),
            self._create_mental_health_cpd(),
            self._create_study_habits_cpd(),
            self._create_attendance_cpd(),
            self._create_performance_cpd()
        ]

        # Add CPDs to the model
        for cpd in cpds:
            self.model.add_cpds(cpd)

        # Check model consistency
        assert self.model.check_model()

    def _create_cpd(self, variable, card, values):
        return TabularCPD(variable=variable, variable_card=card, values=values)

    def _create_mental_health_cpd(self):
        return TabularCPD(variable='MH', variable_card=3,
                          values=np.array([
                              [0.7, 0.3, 0.1],  # Low ES
                              [0.2, 0.4, 0.2],  # Medium ES
                              [0.1, 0.3, 0.7]  # High ES
                          ]),
                          evidence=['ES'], evidence_card=[3])

    def _create_study_habits_cpd(self):
        return TabularCPD(variable='SH', variable_card=3,
                          values=np.array([
                              [0.7, 0.2, 0.1],  # Low PAP
                              [0.2, 0.4, 0.2],  # Medium PAP
                              [0.1, 0.4, 0.7]  # High PAP
                          ]),
                          evidence=['PAP'], evidence_card=[3])

    def _create_attendance_cpd(self):
        return TabularCPD(variable='A', variable_card=3,
                          values=np.array([
                              [0.7, 0.2, 0.1],  # Low MH
                              [0.2, 0.4, 0.2],  # Medium MH
                              [0.1, 0.4, 0.7]  # High MH
                          ]),
                          evidence=['MH'], evidence_card=[3])

    def _create_performance_cpd(self):
        # Generate a comprehensive CPD for Performance
        performance_values = np.zeros((3, 3 ** 4))

        # Iterate through all combinations of SH, A, MH, PAP
        for sh in range(3):
            for a in range(3):
                for mh in range(3):
                    for pap in range(3):
                        # Calculate index for this combination
                        idx = sh * 3 ** 3 + a * 3 ** 2 + mh * 3 + pap

                        # Simple logic for performance probability
                        if sh == 2 and a == 2 and mh == 2 and pap == 2:
                            # Best case
                            performance_values[2, idx] = 0.9
                            performance_values[1, idx] = 0.1
                            performance_values[0, idx] = 0.0
                        elif sh == 0 and a == 0 and mh == 0 and pap == 0:
                            # Worst case
                            performance_values[0, idx] = 0.9
                            performance_values[1, idx] = 0.1
                            performance_values[2, idx] = 0.0
                        else:
                            # Middle scenarios
                            performance_values[1, idx] = 0.5
                            performance_values[0, idx] = 0.3
                            performance_values[2, idx] = 0.2

        return TabularCPD(variable='Performance', variable_card=3,
                          values=performance_values,
                          evidence=['SH', 'A', 'MH', 'PAP'],
                          evidence_card=[3, 3, 3, 3])

    def predict_performance(self, evidence):
        inference = VariableElimination(self.model)
        return inference.query(variables=['Performance'], evidence=evidence)

    def categorize_risk(self, performance_prob):
        performance_level = np.argmax(performance_prob.values)
        risk_categories = {
            0: "High Risk (Struggling)",
            1: "Moderate Risk",
            2: "Low Risk (Excelling)"
        }
        return risk_categories[performance_level]


def main():
    cbn = StudentPerformanceCBN()

    scenarios = [
        {
            'name': 'Student 1',
            'evidence': {
                'SH': 1, 'A': 0, 'MH': 0, 'PAP': 0, 'ES': 2
            }
        },
        {
            'name': 'Student 2',
            'evidence': {
                'SH': 2, 'A': 2, 'MH': 2, 'PAP': 2, 'ES': 0
            }
        }
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']} Scenario:")
        performance_query = cbn.predict_performance(scenario['evidence'])
        print("Performance Probability Distribution:")
        print(performance_query)
        risk_category = cbn.categorize_risk(performance_query)
        print(f"Risk Category: {risk_category}")


if __name__ == "__main__":
    main()