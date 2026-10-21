import scipy.optimize as opt

def objective(x):
    return (x - 3) ** 2 - 9

result = opt.minimize(objective, x0=0.0, method='Nelder-Mead')

print("Optimal x:", result.x[0])
print("Maximum Value:", -result.fun)
