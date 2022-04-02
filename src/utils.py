import numpy as np
import problems.problem1 as p1


def generate(n, int_vars, l_bound, u_bound):
    gen = []

    for i in range(0, n):
        x = np.zeros(len(int_vars))
        for j, int_var in enumerate(int_vars):
            x[j] = np.random.randint(l_bound[j], u_bound[j]) if int_var else \
                l_bound[j] + (u_bound[j] - l_bound[j]) * np.random.rand()
        gen.append(x)

    return np.array(gen)


def evaluate(population, evaluate, max=False):
    fitness = np.zeros(population.shape[0])
    f_fworst = np.min if max else np.max

    score, constraint = evaluate(population)

    feasible_idx = np.where(constraint == 0)
    feasible = score[feasible_idx]
    fworst = f_fworst(feasible) \
        if feasible.size > 0 else 0

    fitness = fworst + constraint
    fitness[feasible_idx] = score[feasible_idx]

    return fitness