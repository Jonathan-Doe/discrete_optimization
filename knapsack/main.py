import numpy as np
knapsack_name = "ks_19_0"

with open(f"data/{knapsack_name}") as df:
    lines = df.readlines()

n, c = [int(x) for x in lines[0].strip().split()]
tmp = np.array([x.strip().split() for x in lines[1:]])


def optimize(capacity, items):

    tmp = np.array(items)
    values, weights = tmp[:, 0].astype(int), tmp[:, 1].astype(int)

    value_per_kilo = values / weights
    sorted_ind = np.argsort(value_per_kilo)

    x = np.zeros_like(values)
    obj = 0
    total_w = 0

    for i in sorted_ind:
        v, w = values[i], weights[i]

        if total_w + w <= capacity:
            x[i] = 1
            total_w += w
            obj += v
        else:
            continue
    return obj, x
