Type the following command line in the terminal:

Task 1:

python FOL_learn.py args_1 args_2 args_3

Here,
args_1: dir of .uai file
args_2: dir of train file
args_3: dir of test file

For example:

python FOL_learn.py ./hw5-data/hw5-data/dataset1/1.uai ./hw5-data/hw5-data/dataset1/train-f-1.txt ./hw5-data/hw5-data/dataset1/test.txt

Then, the program will output the log likelihood difference.

Task 2:

python POD_EM_Learn.py args_1 args_2 args_3

Here,
args_1: dir of .uai file
args_2: dir of train file
args_3: dir of test file

For example:

python POD_EM_Learn.py ./hw5-data/hw5-data/dataset1/1.uai ./hw5-data/hw5-data/dataset1/train-p-1.txt ./hw5-data/hw5-data/dataset1/test.txt

Then, the program will output the mean and standard deviation of log likelihood difference.

Task 3:

python Mixture_Random_Bayes.py args_1 args_2 args_3 args_4

Here,
args_1: dir of .uai file
args_2: dir of train file
args_3: dir of test file
args_4: value of k

For example:

python Mixture_Random_Bayes.py ./hw5-data/hw5-data/dataset1/1.uai ./hw5-data/hw5-data/dataset1/train-f-1.txt ./hw5-data/hw5-data/dataset1/test.txt 2

Then, the program will output the mean and standard deviation of log likelihood difference.