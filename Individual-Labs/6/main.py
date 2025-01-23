import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def create_disaster_response_network():
    model = BayesianNetwork([
        ('D', 'S'), ('D', 'R'),
        ('S', 'I'), ('S', 'R'), ('S', 'E'),
        ('W', 'R'), ('W', 'I'),
        ('I', 'M'), ('I', 'C'), ('I', 'T'),
        ('R', 'E'),
        ('M', 'T'), ('C', 'T')
    ])

    cpd_d = TabularCPD(variable='D', variable_card=3,
        values=[[0.3], [0.4], [0.3]])

    cpd_s = TabularCPD(variable='S', variable_card=3,
        values=[
            [0.6, 0.3, 0.1],
            [0.3, 0.4, 0.3],
            [0.1, 0.3, 0.6]
        ],
        evidence=['D'], evidence_card=[3])

    cpd_w = TabularCPD(variable='W', variable_card=4,
        values=[[0.25], [0.25], [0.25], [0.25]])

    cpd_r = TabularCPD(variable='R', variable_card=3,
        values=np.ones((3, 36)) * 0.33,  # Fill with uniform probabilities
        evidence=['D', 'S', 'W'], evidence_card=[3, 3, 4])

    cpd_i = TabularCPD(variable='I', variable_card=4,
        values=np.ones((4, 12)) * 0.25,
        evidence=['S', 'W'], evidence_card=[3, 4])

    cpd_m = TabularCPD(variable='M', variable_card=3,
        values=np.ones((3, 4)) * 0.33,
        evidence=['I'], evidence_card=[4])

    cpd_c = TabularCPD(variable='C', variable_card=3,
        values=np.ones((3, 4)) * 0.33,
        evidence=['I'], evidence_card=[4])

    cpd_e = TabularCPD(variable='E', variable_card=2,
        values=np.ones((2, 9)) * 0.5,
        evidence=['S', 'R'], evidence_card=[3, 3])

    cpd_t = TabularCPD(variable='T', variable_card=3,
        values=np.ones((3, 36)) * 0.33,
        evidence=['I', 'M', 'C'], evidence_card=[4, 3, 3])

    model.add_cpds(cpd_d, cpd_s, cpd_r, cpd_w, cpd_i, cpd_m, cpd_c, cpd_e, cpd_t)

    if not model.check_model():
        raise ValueError("Model is not valid")

    return model

def disaster_response_inference():
    model = create_disaster_response_network()
    inference = VariableElimination(model)
    evidence = {'D': 0, 'S': 2, 'W': 2}

    print("Inference Results:")
    response_time = inference.query(variables=['T'], evidence=evidence)
    print("\nEmergency Response Time Probabilities:")
    for idx, prob in enumerate(response_time.values):
        print(f"State {idx}: {prob:.4f}")

    evacuation_need = inference.query(variables=['E'], evidence=evidence)
    print("\nEvacuation Need Probabilities:")
    for idx, prob in enumerate(evacuation_need.values):
        print(f"State {idx}: {prob:.4f}")

if __name__ == "__main__":
    disaster_response_inference()