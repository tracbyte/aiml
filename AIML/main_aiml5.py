import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Define the network structure
model = DiscreteBayesianNetwork([
    ('Earthquake', 'Alarm'),
    ('Alarm', 'JohnCalls'),
    ('Alarm', 'MaryCalls')
])

# Define Conditional Probability Distributions (CPDs)
cpd_earthquake = TabularCPD(
    variable='Earthquake',
    variable_card=2,
    values=[[0.99],
            [0.01]]
)

cpd_alarm = TabularCPD(
    variable='Alarm',
    variable_card=2,
    values=[[0.95, 0.2],
            [0.05, 0.8]],
    evidence=['Earthquake'],
    evidence_card=[2]
)

cpd_john = TabularCPD(
    variable='JohnCalls',
    variable_card=2,
    values=[[0.9, 0.2],
            [0.1, 0.8]],
    evidence=['Alarm'],
    evidence_card=[2]
)

cpd_mary = TabularCPD(
    variable='MaryCalls',
    variable_card=2,
    values=[[0.7, 0.1],
            [0.3, 0.9]],
    evidence=['Alarm'],
    evidence_card=[2]
)

# Add CPDs to the model and check consistency
model.add_cpds(cpd_earthquake, cpd_alarm, cpd_john, cpd_mary)
assert model.check_model()

print("\nBayesian Network structure:")
print(model.edges())

# Perform Inference
inference = VariableElimination(model)

# 1. Probability of Earthquake
q1 = inference.query(variables=['Earthquake'])
print("\n1. Probability of Earthquake:")
print(q1)

# 2. Probability of Earthquake given John calls
q2 = inference.query(
    variables=['Earthquake'],
    evidence={'JohnCalls': 1}
)
print("\n2. Probability of Earthquake given John calls:")
print(q2)

# 3. Probability of Earthquake given John and Mary both call
q3 = inference.query(
    variables=['Earthquake'],
    evidence={'JohnCalls': 1, 'MaryCalls': 1}
)
print("\n3. Probability of Earthquake given John and Mary both call:")
print(q3)