import copy
import sys
import re
import numpy as np

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
    probs_learn = []
    graph_model['strides'] = []
    new_train = []
    lldiffs = []
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
        # random initialization
        probs_learn[-1].extend(np.random.rand(num_prob))

    strides = []
    num_samples = []
    for initialization in range(num_initializations):
        if initialization > 0:
            for i in range(len(probs_learn)):
                probs_learn[i][1:] = np.random.rand(probs_learn[i][0])
        for iteration in range(num_iterations):
            if initialization == 0 and iteration == 0:
                # E-step
                start_index = 0
                for train in train_data[1:]:
                    unknown_indexes = np.where(np.asarray(train) == '?')
                    num_unknown_var = len(unknown_indexes[0])
                    strides.append([num_unknown_var, 1])
                    stride = 1
                    for i in range(1, num_unknown_var):
                        stride *= graph_model['cardinalities'][unknown_indexes[0][i]]
                        strides[-1].insert(1, stride)
                    sum_weights = 0
                    num_samples.append(strides[-1][1] * graph_model['cardinalities'][unknown_indexes[0][0]])
                    for i in range(num_samples[-1]):
                        new_train.append(copy.deepcopy(train))
                        for j in range(num_unknown_var):
                            value = int(i / strides[-1][j + 1]) % graph_model['cardinalities'][unknown_indexes[0][j]]
                            new_train[-1][unknown_indexes[0][j]] = str(value)
                        new_train[-1] = list(map(int, new_train[-1]))
                        weight = 1
                        for f in range(len(graph_model['functions'])):
                            index = 0
                            function = graph_model['functions'][f]
                            for j in range(1, len(function)):
                                index += new_train[-1][function[j]] * graph_model['strides'][f][j]
                            weight *= probs_learn[f][index + 1]
                        new_train[-1].append(weight)
                        sum_weights += weight
                    for index in range(start_index, start_index + num_samples[-1]):
                        new_train[index][-1] = new_train[index][-1] / sum_weights
                    start_index = start_index + num_samples[-1]
            else:
                # E-step
                start_index = 0
                for s in range(len(strides)):
                    sum_weights = 0
                    for i in range(num_samples[s]):
                        weight = 1
                        for f in range(len(graph_model['functions'])):
                            index = 0
                            function = graph_model['functions'][f]
                            for j in range(1, len(function)):
                                index += new_train[start_index + i][function[j]] * graph_model['strides'][f][j]
                            weight *= probs_learn[f][index + 1]
                        new_train[start_index + i][-1] = weight
                        sum_weights += weight
                    for index in range(start_index, start_index + num_samples[s]):
                        new_train[index][-1] /= sum_weights
                    start_index = start_index + num_samples[s]
            # M-step
            prob_index = 0
            for function in graph_model['functions']:
                num_prob = graph_model['probs'][prob_index][0]
                for i in range(num_prob):
                    weight_child_parents = 0
                    weight_parents = 0
                    for train in new_train:
                        flag_child_parents = 1
                        flag_parents = 1
                        for j in range(1, len(function)):
                            cur_value = int(i / graph_model['strides'][prob_index][j]) % graph_model['cardinalities'][function[j]]
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
                        probs_learn[prob_index][i + 1] = 1e-5
                    else:
                        probs_learn[prob_index][i + 1] = weight_child_parents / weight_parents
                prob_index += 1
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
