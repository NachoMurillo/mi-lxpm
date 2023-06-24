import numpy as np

# Discrete (True) or Continuous Variable (False)
INT_VARS = [True]*10 + [False]*10

# Minimization Problem
MAX = False
L_BOUND = [0]*10 + [298]*10
U_BOUND = [1]*10 + [308]*10

POP_SIZE = 10

def evaluate(x):
    # Minimum servers to meet QoS constraints
    wk = 3
    D = 2
    lk = 1
    min_servers = np.ceil(wk*(1/D + lk))

    constraint1 = -x[:, :10]                           # Servers between 0 and 1 (off, on)
    constraint2 = x[:, :10] - 1
    constraint3 = 298 - x[:, 10:]                     # Tr between 298K and 308K
    constraint4 = x[:, 10:] - 308
    constraint5 = min_servers - np.sum(x[:, :10], axis=1)

    constraints = np.hstack((constraint1, constraint2, constraint3, constraint4, constraint5.reshape(-1, 1)))

    constraints = np.absolute(constraints.clip(min=0))

    y = np.sum(x[:, :10], axis=1)

    return y, np.sum(constraints, axis=1)