import numpy as np

# Discrete (True) or Continuous Variable (False)
INT_VARS = [True]*10 + [False]

# Minimization Problem
MAX = False
L_BOUND = [0]*10 + [298]
U_BOUND = [1]*10 + [308]

POP_SIZE = 3

def evaluate(x, x_act, ti_act):
    # Minimum servers to meet QoS constraints
    wk = 3
    D = 2
    lk = 1
    min_servers = np.ceil(wk*(1/D + lk))

    # Thermal constants 
    cp = 5
    qa = 1
    Kt = 1.6
    ts = 1.0
    a1 = 180
    # Ti = Ti + ts/Kt * ((cp*qa) * (Tc-Ti) + Pi)

    # Assumptions - Tc = Trinput, Pi = Working Pi input, ts = tm
    Pi = a1*x[:, :10]
    ti_act = np.array(ti_act)
    Tr = np.array(x[:, 10])
    ti_act = np.tile(ti_act.reshape(-1, 1), (1, POP_SIZE)).T
    Tc = np.tile(Tr.reshape(-1, 1), (1, 10))

    print('[F_eval] Ti act: ')
    print(ti_act)
    print(np.shape(ti_act))

    print('[F_eval] Tc: ')
    print(Tc)

    print('[F_eval] Tr: ')
    print(Tr)

    Ti = ti_act + ts/Kt *((cp*qa)*(Tc-ti_act) + Pi)

    print('[F_eval] Pi: ')
    print(Pi)

    print('[F_eval] Ti: ')
    print(Ti)

    constraint1 = -x[:, :10]                           # Servers between 0 and 1 (off, on)
    constraint2 = x[:, :10] - 1
    constraint3 = 298 - x[:, 10]                     # Tr between 298K and 308K
    constraint4 = x[:, 10] - 308
    constraint5 = min_servers - np.sum(x[:, :10], axis=1)

    constraints = np.hstack((constraint1, constraint2, constraint3.reshape(-1, 1), constraint4.reshape(-1, 1), constraint5.reshape(-1, 1)))

    constraints = np.absolute(constraints.clip(min=0))

    y = np.sum(x[:, :10], axis=1)

    return y, np.sum(constraints, axis=1)