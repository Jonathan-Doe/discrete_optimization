#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight'])


def optimize(capacity, vw_matrix):

    values, weights = vw_matrix[:, 0].astype(int), vw_matrix[:, 1].astype(int)

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


def dp_optimize(capacity, w, v):
    j = len(w)
    m = np.zeros((capacity + 1, j + 1), dtype=int)
    m[:, 0] = 0
    for item_index in range(1, j + 1):
        w_item, v_item = w[item_index - 1], v[item_index - 1]
        if w_item > capacity:
            m[:, item_index] = m[:, item_index - 1]
        else:
            m[:w_item, item_index] = m[:w_item, item_index - 1]
            for tmp_capacity in range(w_item, capacity + 1):
                new_value = v_item + m[tmp_capacity - w_item, item_index - 1]
                m[tmp_capacity, item_index] = max([new_value, m[tmp_capacity, item_index - 1]])

    obj = m[-1, -1]
    x = np.zeros_like(w, dtype=int)
    tmp_capacity = capacity
    for item_index in range(j, 0, -1):
        if m[tmp_capacity, item_index] == m[tmp_capacity, item_index - 1]:
            x[item_index - 1] = 0
        else:
            x[item_index - 1] = 1
            tmp_capacity -= w[item_index - 1]
    return obj, x


def dp_optimize_two(capacity, w, v):
    j = len(w)
    m = np.zeros((capacity + 1, 2), dtype=int)
    m[:, 0] = 0
    m_bool = np.zeros((capacity + 1, j + 1), dtype=bool)

    for item_index in range(1, j + 1):
        w_item, v_item = w[item_index - 1], v[item_index - 1]
        if w_item > capacity:
            m[:, 1] = m[:, 0]
        else:
            m[:w_item, 1] = m[:w_item, 0]
            new_value = v_item + m[:-w_item, 0]
            m[w_item:, 1] = np.max((new_value, m[w_item:, 0]), axis=0)

        m_bool[:, item_index] = (m[:, 1] - m[:, 0]).astype(bool)
        m[:, 0] = m[:, 1]
    obj = m[-1, 1]

    x = np.zeros_like(w, dtype=bool)
    tmp_capacity = capacity
    for item_index in range(j, 0, -1):
        if m_bool[tmp_capacity, item_index]:
            x[item_index - 1] = 1
            tmp_capacity -= w[item_index - 1]
        else:
            x[item_index - 1] = 0

    return obj, x.astype(int)


def solve_it(input_data):
    tmp = np.array(input_data.strip().split()).astype(int).reshape(-1, 2)

    values, weights = tmp[1:, 0].astype(int), tmp[1:, 1].astype(int)
    n, c = tmp[0]
    if n > 1100:
        cutoff = 200
        v_over_w = values / weights
        ind_to_sort = np.argsort(v_over_w)[::-1]
        values_sort, weights_sort = values[ind_to_sort][:cutoff], weights[ind_to_sort][:cutoff]
        obj, x_sort = dp_optimize_two(c, weights_sort, values_sort)
        x = np.zeros_like(values)
        for i, ind in enumerate(ind_to_sort[:cutoff]):
            if x_sort[i]:
                x[ind_to_sort[i]] = 1
    else:
        obj, x = dp_optimize_two(c, weights, values)
    #obj, x = optimize(c, tmp[1:])
    # prepare the solution in the specified output format
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, x))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()

        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

