import sys
import re
import numpy as np


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
def FOD_learn(graph_model, train_data):
    probs_learn = []
    graph_model['strides'] = []
    for function in graph_model['functions']:
        num_var = len(function) - 1
        graph_model['strides'].append([num_var, 1])
        stride = 1
        num_prob = graph_model['cardinalities'][num_var - 1]
        for i in range(1, num_var):
            stride *= graph_model['cardinalities'][function[num_var - i - 1]]
            graph_model['strides'][-1].insert(1, stride)
            num_prob *= graph_model['cardinalities'][num_var - i]
        probs_learn.append([num_prob])
        for i in range(num_prob):
            num_child_parents = 0
            num_parents = 0
            for train in train_data[1:]:
                flag_child_parents = 1
                flag_parents = 1
                for j in range(1, len(function)):
                    cur_value = int(i / graph_model['strides'][-1][j]) % graph_model['cardinalities'][function[j]]
                    if cur_value != train[function[j]] and j != len(function) - 1:
                        flag_child_parents = 0
                        flag_parents = 0
                    elif cur_value != train[function[j]] and j == len(function) - 1:
                        flag_child_parents = 0
                if flag_child_parents == 1:
                    num_child_parents += 1
                if flag_parents == 1:
                    num_parents += 1
            if num_child_parents == 0:
                probs_learn[-1].append(1e-5)
            else:
                probs_learn[-1].append(float(num_child_parents) / num_parents)

    return graph_model, probs_learn


# test on test data
def FOD_test(graph_model, probs_learn, test_data):
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
            ll_bl += np.log10(probs_learn[i][index + 1])
        lldiff += np.abs(ll_bo - ll_bl)

    return lldiff


# insist on 3 arguments
if len(sys.argv) != 4:
    print(sys.argv[0], "takes 3 arguments. Not ", len(sys.argv) - 1)
    sys.exit()

data_dir = sys.argv[1]
train_dir = sys.argv[2]
test_dir = sys.argv[3]

graph_model, train_data, test_data = read(data_dir, train_dir, test_dir)
graph_model, probs_learn = FOD_learn(graph_model, train_data)
lldiff = FOD_test(graph_model, probs_learn, test_data)

print('./FOD_learn.py <{0}> <{1}> <{2}> <{3}>'.format(data_dir.split('/')[-1], 1, train_dir.split('/')[-1], test_dir.split('/')[-1]))
print('--------------------------------------------')
print('log likelihood difference = {}'.format(lldiff))
print('--------------------------------------------')
