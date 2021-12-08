import copy
import sys
import re
import numpy as np
import random

num_iterations = 20
num_initializations = 5


def read(data_dir, train_dir, test_dir):
    # network file
    graph_model = {}
    orig_data = []

    with open(data_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line != '':
                orig_data.append(line)

    # preamble
    graph_model['type_network'] = orig_data[0]
    graph_model['num_variables'] = int(orig_data[1])
    graph_model['cardinalities'] = list(map(int, orig_data[2].split(' ')))
    graph_model['num_functions'] = int(orig_data[3])
    graph_model['functions'] = []
    for i in range(4, graph_model['num_functions'] + 4):
        graph_model['functions'].append(list(map(int, re.split(' |\t', orig_data[i]))))

    # function tables
    graph_model['probs'] = []
    i = graph_model['num_functions'] + 4
    while i != len(orig_data):
        num_probs = int(orig_data[i])
        graph_model['probs'].append([num_probs])
        while len(graph_model['probs'][-1]) != num_probs + 1:
            new_probs = list(map(float, orig_data[i + 1].split(' ')))
            graph_model['probs'][-1].extend(new_probs)
            i += 1
        i += 1

    # train data file
    train_data = []
    with open(train_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            train_data.append(list(map(int, line.split(' '))))

    # test data file
    test_data = []
    with open(test_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            test_data.append(list(map(int, line.split(' '))))

    return graph_model, train_data, test_data


# learn probabilities based on fully observed data
def Mixture_Random_Bayes(graph_model, train_data, test_data, k):
    # construct k random DAG structures over X
    functions = {}
    for i in range(k):
        functions[i] = []
        for j in range(graph_model['num_functions']):
            rand_indexes = np.arange(graph_model['num_functions']).tolist()
            rand_indexes.remove(j)
            for m in range(0, j):
                if j in functions[i][m][1:]:
                    rand_indexes.remove(m)
            num_parents = np.random.randint(4, size=1)[0]
            parents = random.sample(rand_indexes, num_parents)
            functions[i].append(parents)
            functions[i][-1].append(j)
            functions[i][-1].insert(0, len(functions[i][-1]))

    graph_model['strides'] = []
    for function in graph_model['functions']:
        num_var = len(function) - 1
        graph_model['strides'].append([num_var, 1])
        stride = 1
        for i in range(1, num_var):
            stride *= graph_model['cardinalities'][function[num_var - i - 1]]
            graph_model['strides'][-1].insert(1, stride)

    probs_learn = {}
    strides = {}
    train_datas = {}
    lldiffs = []
    for i in range(k):
        train_datas[i] = copy.deepcopy(train_data)
        probs_learn[i] = []
        strides[i] = []
        for function in functions[i]:
            num_var = len(function) - 1
            strides[i].append([num_var, 1])
            stride = 1
            num_prob = graph_model['cardinalities'][num_var - 1]
            for j in range(1, num_var):
                stride *= graph_model['cardinalities'][function[num_var - j - 1]]
                strides[i][-1].insert(1, stride)
                num_prob *= graph_model['cardinalities'][num_var - j]
            probs_learn[i].append([num_prob])
            # random initialization
            probs_learn[i][-1].extend(np.random.rand(num_prob))

    for initialization in range(num_initializations):
        p = np.random.dirichlet(np.ones(k), size=1)[0]
        if initialization > 0:
            for i in range(k):
                for j in range(len(probs_learn[i])):
                    probs_learn[i][j][1:] = np.random.rand(probs_learn[i][j][0])
        for iteration in range(num_iterations):
            # E-step
            for t in range(1, len(train_data)):
                sum_weights = 0
                for k_i in range(k):
                    weight = p[k_i]
                    for f in range(len(functions[k_i])):
                        index = 0
                        function = functions[k_i][f]
                        for j in range(1, len(function)):
                            index += train_data[t][function[j]] * strides[k_i][f][j]
                        weight *= probs_learn[k_i][f][index + 1]
                    if initialization == 0 and iteration == 0:
                        train_datas[k_i][t].append(weight)
                    else:
                        train_datas[k_i][t][-1] = weight
                    sum_weights += weight
                for k_i in range(k):
                    train_datas[k_i][t][-1] /= sum_weights
            # M-step
            for k_i in range(k):
                prob_index = 0
                for function in functions[k_i]:
                    num_prob = probs_learn[k_i][prob_index][0]
                    for i in range(num_prob):
                        weight_child_parents = 0
                        weight_parents = 0
                        for train in train_data[1:]:
                            flag_child_parents = 1
                            flag_parents = 1
                            for j in range(1, len(function)):
                                cur_value = int(i / strides[k_i][prob_index][j]) % graph_model['cardinalities'][function[j]]
                                if cur_value != train[function[j]] and j != len(function) - 1:
                                    flag_child_parents = 0
                                    flag_parents = 0
                                elif cur_value != train[function[j]] and j == len(function) - 1:
                                    flag_child_parents = 0
                            if flag_child_parents == 1:
                                weight_child_parents += train[-1]
                            if flag_parents == 1:
                                weight_parents += train[-1]
                        if weight_child_parents == 0:
                            probs_learn[k_i][prob_index][i + 1] = 1e-5
                        else:
                            probs_learn[k_i][prob_index][i + 1] = weight_child_parents / weight_parents
                    prob_index += 1
                p[k_i] = 0
                for train in train_datas[k_i][1:]:
                    p[k_i] += train[-1]
                p[k_i] /= train_data[0][1]
        lldiff = 0
        for test in test_data[1:]:
            ll_bo = 0
            ll_bl = 0
            for i in range(len(graph_model['functions'])):
                index = 0
                function = graph_model['functions'][i]
                for j in range(1, len(function)):
                    index += test[function[j]] * graph_model['strides'][i][j]
                ll_bo += np.log10(graph_model['probs'][i][index + 1])
            for k_i in range(k):
                for i in range(len(functions[k_i])):
                    index = 0
                    function = functions[k_i][i]
                    for j in range(1, len(function)):
                        index += test[function[j]] * strides[k_i][i][j]
                    ll_bl += np.log10(p[k_i]) + np.log10(probs_learn[k_i][i][index + 1])
            lldiff += np.abs(ll_bo - ll_bl)
        lldiffs.append(lldiff)

    return lldiffs


# insist on 4 arguments
if len(sys.argv) != 5:
    print(sys.argv[0], "takes 4 arguments. Not ", len(sys.argv) - 1)
    sys.exit()

data_dir = sys.argv[1]
train_dir = sys.argv[2]
test_dir = sys.argv[3]
k = int(sys.argv[4])

graph_model, train_data, test_data = read(data_dir, train_dir, test_dir)
lldiffs = Mixture_Random_Bayes(graph_model, train_data, test_data, k)
mean_lldiff = np.mean(lldiffs)
std_lldiff = np.std(lldiffs)

print('./Mixture_Random_Bayes.py <{0}> <{1}> <{2}> <{3}>'.format(data_dir.split('/')[-1], 3, train_dir.split('/')[-1], test_dir.split('/')[-1]))
print('--------------------------------------------')
print('mean of log likelihood difference = {}'.format(mean_lldiff))
print('standard deviation of log likelihood difference = {}'.format(std_lldiff))
print('--------------------------------------------')
