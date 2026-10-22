from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([('Earthquake', 'Alarm'), ('Alarm', 'JohnCalls'), ('Alarm', 'MaryCalls')])
model.add_cpds(
    TabularCPD('Earthquake', 2, [[0.99], [0.01]]),
    TabularCPD('Alarm', 2, [[0.95, 0.2], [0.05, 0.8]], ['Earthquake'], [2]),
    TabularCPD('JohnCalls', 2, [[0.9, 0.2], [0.1, 0.8]], ['Alarm'], [2]),
    TabularCPD('MaryCalls', 2, [[0.7, 0.1], [0.3, 0.9]], ['Alarm'], [2])
)

infer = VariableElimination(model)
print(infer.query(['Earthquake']))
print(infer.query(['Earthquake'], evidence={'JohnCalls': 1}))
print(infer.query(['Earthquake'], evidence={'JohnCalls': 1, 'MaryCalls': 1}))
