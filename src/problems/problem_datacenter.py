import numpy as np

# Discrete (True) or Continuous Variable (False)
INT_VARS = [True]*10 + [False]

# Minimization Problem
MAX = False
L_BOUND = [0]*10 + [298]
U_BOUND = [1]*10 + [308]

POP_SIZE = 10

def evaluate(x, x_act, ti_act, tc_act):
    # Minimum servers to meet QoS constraints
    wk = 3
    D = 2
    lk = 1
    min_servers = np.ceil(wk*(1/D + lk))

    # Thermal constants 
    cp = 5
    qa = 1
    Kt = 1.6
    ts = 0.1
    a1 = 180
    tau = 0.18
    
    # Tc prediction
    Tr = np.array(x[:, 10])
    Tc = tc_act + ts/tau * (Tr - tc_act)

    # CoP prediction
    Pi = a1*x[:, :10]
    CoP = 0.0068*Tc**2 + 0.0008*Tc + 0.458

    # Reshape Tc and Ti_act to operate with Ti (servers temperature) size
    Tc = np.tile(Tc.reshape(-1, 1), (1, 10))
    ti_act = np.array(ti_act)
    ti_act = np.tile(ti_act.reshape(-1, 1), (1, POP_SIZE)).T

    # Ti prediction
    Ti = ti_act + ts/Kt *((cp*qa)*(Tc-ti_act) + Pi)

    # Constraints
    constraint1 = -x[:, :10] # Servers between 0 and 1 (off, on)
    constraint2 = x[:, :10] - 1
    constraint3 = 298 - x[:, 10] # Tr between 298K and 308K
    constraint4 = x[:, 10] - 308
    constraint5 = min_servers - np.sum(x[:, :10], axis=1) # QoS constraints
    constraint6 = Ti - 353 # Ti <= 353K

    constraints = np.hstack((constraint1, constraint2, constraint3.reshape(-1, 1), constraint4.reshape(-1, 1), constraint5.reshape(-1, 1), constraint6))

    constraints = np.absolute(constraints.clip(min=0))

    total_power_consumption = np.sum(CoP[:, np.newaxis] * Pi, axis=1)

    # delta U (number off-on commutation)
    servers_act = np.array(x_act[:10])
    servers_control = np.array(x[:, :10])

    delta_u = np.sum(servers_control != servers_act[np.newaxis, :], axis=1)
    lambda_u = 10000

    # delta Tr (difference between Tr_act and candidate Tr)
    delta_r = x_act[10] - Tr
    lambda_r = 10000

    y = total_power_consumption + lambda_u*delta_u + lambda_r*delta_r

    return y, np.sum(constraints, axis=1)