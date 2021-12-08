import sys
import copy
import re
import math
import numpy as np
import time


def read(data_dir, evid_dir):
    graph_model = {}
    orig_data = []
    orig_evid = []

    with open(data_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line != '':
                orig_data.append(line)
    with open(evid_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line != '':
                orig_evid = line

    # preamble
    graph_model['type_network'] = orig_data[0]
    graph_model['num_variables'] = int(orig_data[1])
    graph_model['cardinalities'] = list(map(int, orig_data[2].split(' ')))
    graph_model['num_cliques'] = int(orig_data[3])
    graph_model['cliques'] = []
    for i in range(4, graph_model['num_cliques'] + 4):
        graph_model['cliques'].append(list(map(int, re.split(' |\t', orig_data[i]))))

    # function tables
    graph_model['probs'] = []
    for i in range(graph_model['num_cliques'] + 4, 3 * graph_model['num_cliques'] + 4, 2):
        graph_model['probs'].append([int(orig_data[i])] + list(map(float, orig_data[i + 1].split(' '))))

    # evidence
    orig_evid = list(map(int, orig_evid.split(' ')))
    graph_model['evid'] = {}
    for i in range(1, orig_evid[0] + 1):
        graph_model['evid'][orig_evid[2 * i - 1]] = orig_evid[2 * i]

    return graph_model


# min-degree ordering
def order(graph_model):
    temp_graph_model = copy.deepcopy(graph_model)
    neighbors = {}
    pi = []

    for i in range(temp_graph_model['num_variables']):
        neighbors[i] = []
    for clique in temp_graph_model['cliques']:
        for var1 in clique[1:]:
            for var2 in clique[1:]:
                if var1 != var2 and var2 not in neighbors[var1] and var1 not in temp_graph_model['evid'].keys() and var2 not in temp_graph_model['evid'].keys():
                    neighbors[var1].append(var2)

    for i in range(temp_graph_model['num_variables']):
        # find a variable in X with smallest number of neighbors in G
        min_num = 10000000000
        min_var = -1
        for key in neighbors:
            if len(neighbors[key]) < min_num and key not in temp_graph_model['evid'].keys():
                min_num = len(neighbors[key])
                min_var = key
        if min_var == -1:
            continue
        pi.append(min_var)
        # add an edge between every pair of non-adjacent neighbors of pi(i) in G
        for j in range(0, len(neighbors[min_var]) - 1):
            for k in range(j+1, len(neighbors[min_var])):
                if neighbors[min_var][k] not in neighbors[neighbors[min_var][j]]:
                    neighbors[neighbors[min_var][j]].append(neighbors[min_var][k])
                    neighbors[neighbors[min_var][k]].append(neighbors[min_var][j])
        # delete variable pi(i) from G and from X
        for j in range(0, len(neighbors[min_var])):
            neighbors[neighbors[min_var][j]].remove(min_var)
        neighbors.pop(min_var)
    print('Elimination order: {0}'.format(pi))

    return pi


# instantiate evidence in the factor
def instantiate(graph_model):
    temp_graph_model = copy.deepcopy(graph_model)
    for key in temp_graph_model['evid']:
        value = temp_graph_model['evid'][key]
        for i in range(temp_graph_model['num_cliques']):
            stride = 1
            for j in range(1, temp_graph_model['cliques'][i][0] + 1):
                if j != 1:
                    stride = temp_graph_model['cardinalities'][temp_graph_model['cliques'][i][j - 2]] * stride
                if key == temp_graph_model['cliques'][i][j]:
                    num_choices = 1
                    for card in temp_graph_model['cliques'][i][1:]:
                        num_choices *= temp_graph_model['cardinalities'][card]
                    for start_index in range(0, num_choices, stride):
                        if start_index / stride % temp_graph_model['cardinalities'][j] != value:
                            for k in range(start_index + 1, start_index + stride + 1):
                                temp_graph_model['probs'][i][k] = 0
                    break

    return temp_graph_model


# compute phi_3 = phi_1 * phi_2, phi_1 = sum_y(phi)
def product_sum_out(graph_model, pi):
    temp_graph_model = copy.deepcopy(graph_model)
    # define stride
    temp_graph_model['stride'] = []
    # define clusters
    temp_graph_model['clusters'] = []
    for i in range(temp_graph_model['num_cliques']):
        temp_graph_model['stride'].append([temp_graph_model['cliques'][i][0], 1])
        for j in range(2, temp_graph_model['cliques'][i][0] + 1):
            temp_graph_model['stride'][i].append(temp_graph_model['cardinalities'][temp_graph_model['cliques'][i][j - 2]] * temp_graph_model['stride'][i][j - 1])

    for pi_i in pi:
        c_1 = -1
        c_2 = 0
        while c_2 != len(temp_graph_model['cliques']) and c_1 != c_2:
            if pi_i in temp_graph_model['cliques'][c_2][1:] and temp_graph_model['cliques'][c_1][-1] != 'visited' and temp_graph_model['cliques'][c_2][-1] != 'visited':
                if c_1 == -1:
                    c_1 = c_2
                elif pi_i in temp_graph_model['cliques'][c_1][1:]:
                    # product
                    phi_1 = {}
                    phi_2 = {}
                    phi_1['cliques'] = temp_graph_model['cliques'][c_1]
                    phi_1['probs'] = temp_graph_model['probs'][c_1]
                    phi_2['cliques'] = temp_graph_model['cliques'][c_2]
                    phi_2['probs'] = temp_graph_model['probs'][c_2]
                    new_clique = sorted(set(phi_1['cliques'][1:] + phi_2['cliques'][1:]), key=(phi_1['cliques'][1:] + phi_2['cliques'][1:]).index)
                    new_clique.insert(0, len(new_clique))
                    temp_graph_model['cliques'].append(new_clique)
                    temp_graph_model['stride'].append([temp_graph_model['cliques'][-1][0], 1])
                    for j in range(2, temp_graph_model['cliques'][-1][0] + 1):
                        temp_graph_model['stride'][-1].append(temp_graph_model['cardinalities'][temp_graph_model['cliques'][-1][j - 2]] * temp_graph_model['stride'][-1][j - 1])
                    num_val = 1
                    for var in new_clique[1:]:
                        num_val *= temp_graph_model['cardinalities'][var]
                    j = 0
                    k = 0
                    temp_graph_model['probs'].append([])
                    assignment = []
                    for l in range(new_clique[0]):
                        assignment.append(0)
                    for i in range(0, num_val):
                        temp_graph_model['probs'][-1].append(phi_1['probs'][j + 1] * phi_2['probs'][k + 1])
                        for l in range(new_clique[0]):
                            assignment[l] += 1
                            if assignment[l] == temp_graph_model['cardinalities'][new_clique[l + 1]]:
                                assignment[l] = 0
                                if new_clique[l + 1] in temp_graph_model['cliques'][c_1][1:]:
                                    phi_1_stride = temp_graph_model['stride'][c_1][1:][temp_graph_model['cliques'][c_1][1:].index(new_clique[l + 1])]
                                else:
                                    phi_1_stride = 0
                                if new_clique[l + 1] in temp_graph_model['cliques'][c_2][1:]:
                                    phi_2_stride = temp_graph_model['stride'][c_2][1:][temp_graph_model['cliques'][c_2][1:].index(new_clique[l + 1])]
                                else:
                                    phi_2_stride = 0
                                j = j - (temp_graph_model['cardinalities'][new_clique[l + 1]] - 1) * phi_1_stride
                                k = k - (temp_graph_model['cardinalities'][new_clique[l + 1]] - 1) * phi_2_stride
                            else:
                                if new_clique[l + 1] in temp_graph_model['cliques'][c_1][1:]:
                                    phi_1_stride = temp_graph_model['stride'][c_1][1:][temp_graph_model['cliques'][c_1][1:].index(new_clique[l + 1])]
                                else:
                                    phi_1_stride = 0
                                if new_clique[l + 1] in temp_graph_model['cliques'][c_2][1:]:
                                    phi_2_stride = temp_graph_model['stride'][c_2][1:][temp_graph_model['cliques'][c_2][1:].index(new_clique[l + 1])]
                                else:
                                    phi_2_stride = 0
                                j = j + phi_1_stride
                                k = k + phi_2_stride
                                break
                    temp_graph_model['cliques'][c_1].append('visited')
                    temp_graph_model['cliques'][c_2].append('visited')
                    temp_graph_model['probs'][-1].insert(0, len(temp_graph_model['probs'][-1]))
                    c_1 = len(temp_graph_model['cliques']) - 1
            c_2 += 1

        # sum_out
        prev = len(temp_graph_model['cliques']) - 1
        # update clusters
        prev_len = len(temp_graph_model['cliques'][prev])
        temp_graph_model['clusters'].append(temp_graph_model['cliques'][prev][:prev_len])
        temp_graph_model['probs'].append([])
        var_index = temp_graph_model['cliques'][-1][1:].index(pi_i)
        start_index = 0
        stride = temp_graph_model['stride'][-1][1:][var_index]
        while start_index < temp_graph_model['probs'][prev][0]:
            temp_graph_model['probs'][-1].append(0)
            for i in range(temp_graph_model['cardinalities'][pi_i]):
                temp_graph_model['probs'][-1][-1] += temp_graph_model['probs'][prev][start_index + i * stride + 1]
            start_index += temp_graph_model['cardinalities'][pi_i] * stride
        temp_graph_model['probs'][-1].insert(0, len(temp_graph_model['probs'][-1]))
        # update clique
        new_clique = temp_graph_model['cliques'][prev][1:]
        del new_clique[var_index]
        new_clique.insert(0, len(new_clique))
        temp_graph_model['cliques'].append(new_clique)
        # update stride
        temp_graph_model['stride'].append([temp_graph_model['cliques'][-1][0], 1])
        for j in range(2, temp_graph_model['cliques'][-1][0] + 1):
            temp_graph_model['stride'][-1].append(temp_graph_model['cardinalities'][temp_graph_model['cliques'][-1][j - 2]] * temp_graph_model['stride'][-1][j - 1])
        temp_graph_model['cliques'][prev].append('visited')

    # add evidence probabilities
    clique = temp_graph_model['cliques'][-1]
    evid = temp_graph_model['evid']
    stride = temp_graph_model['stride'][-1]
    var_index = len(stride) - 1
    index = 0
    if len(evid) > 0:
        while var_index != 0:
            index += evid[clique[var_index]] * stride[var_index]
            var_index -= 1
    for evid_index in range(0, len(temp_graph_model['cliques']) - 1):
        if temp_graph_model['cliques'][evid_index][-1] != 'visited':
            for prob in temp_graph_model['probs'][evid_index][1:]:
                if prob != 0:
                    temp_graph_model['probs'][-1][index + 1] *= prob
                    temp_graph_model['cliques'][evid_index].append('visited')

    return temp_graph_model


# Sampling-based Variable Elimination and Conditioning
def sampling(graph_model, new_graph_model, w, N):
    Z = 0
    X = wcutset(new_graph_model, w)
    Q = []
    for x in X:
        Q.append(1.0 / graph_model['cardinalities'][x])
    for i in range(N):
        temp_graph_model = copy.deepcopy(graph_model)
        # generate sample
        X_x = []
        for x in X:
            X_x_i = np.random.randint(temp_graph_model['cardinalities'][x])
            X_x.append(X_x_i)
            # set X = x as evidence in the PGM
            temp_graph_model['evid'][x] = X_x_i
        # variable elimination
        temp_graph_model = instantiate(temp_graph_model)
        pi = order(temp_graph_model)
        temp_graph_model = product_sum_out(temp_graph_model, pi)
        # weight of the sample
        clique = temp_graph_model['cliques'][-1]
        probs = temp_graph_model['probs'][-1]
        evid = temp_graph_model['evid']
        stride = temp_graph_model['stride'][-1]
        var_index = len(stride) - 1
        index = 0
        if len(evid) > 0:
            while var_index != 0:
                index += evid[clique[var_index]] * stride[var_index]
                var_index -= 1
        VE = math.log10(probs[index + 1])
        Q_X_x = 1
        for q in Q:
            Q_X_x *= q
        w = VE / Q_X_x
        Z += w

    return Z / N


# wCutset algorithm
def wcutset(graph_model, w):
    temp_graph_model = copy.deepcopy(graph_model)
    X = []
    num_max = -1
    X_num = {}
    for cluster in temp_graph_model['clusters']:
        if cluster[0] > num_max:
            num_max = cluster[0]
        for c in cluster[1:]:
            if c not in X_num:
                X_num[c] = 1
            else:
                X_num[c] += 1

    while num_max > (w + 1):
        X_max = max(X_num, key=X_num.get)
        for i in range(len(temp_graph_model['clusters'])):
            if X_max in temp_graph_model['clusters'][i][1:]:
                index_max = temp_graph_model['clusters'][i][1:].index(X_max) + 1
                temp_graph_model['clusters'][i].pop(index_max)
                temp_graph_model['clusters'][i][0] -= 1
        X.append(X_max)
        del X_num[X_max]
        num_max = -1
        for cluster in temp_graph_model['clusters']:
            if cluster[0] > num_max:
                num_max = cluster[0]

    return X


def print_result(graph_model):
    clique = graph_model['cliques'][-1]
    probs = graph_model['probs'][-1]
    evid = graph_model['evid']
    stride = graph_model['stride'][-1]
    var_index = len(stride) - 1
    index = 0
    if len(evid) > 0:
        while var_index != 0:
            index += evid[clique[var_index]] * stride[var_index]
            var_index -= 1
    print('Partition function: {0}'.format(math.log10(probs[index + 1])))


# start time
start = time.time()
# insist on 2 arguments
if len(sys.argv) != 5:
    print(sys.argv[0], "takes 4 arguments. Not ", len(sys.argv) - 1)
    sys.exit()

data_dir = sys.argv[1]
evid_dir = sys.argv[2]
# w denotes the w-cutset bound
w = int(sys.argv[3])
# N denotes the number of samples
N = int(sys.argv[4])

graph_model = read(data_dir, evid_dir)
new_graph_model = instantiate(graph_model)
pi = order(new_graph_model)
new_graph_model = product_sum_out(new_graph_model, pi)
# print_result(new_graph_model)
part_fun = sampling(graph_model, new_graph_model, w, N)
print('Partition function: {0}'.format(part_fun))
# read partition function and compute the error
PR = []
with open(data_dir + '.PR') as f:
    for line in f.readlines():
        line = line.strip()
        if line != '':
            PR.append(line)
part_true = float(PR[-1])
error = (np.log(part_true) - np.log(part_fun)) / np.log(part_true)
print('error: {0}'.format(error))
# end time
end = time.time()
print('time: {0}'.format(end - start))

