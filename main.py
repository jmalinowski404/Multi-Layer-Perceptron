from functions import *
from mlp import *
from train import *
from test import *

import pandas as pd
import random

if __name__ == "__main__":
    name_map = {
        'Iris-setosa': [1, 0, 0],
        'Iris-versicolor': [0, 1, 0],
        'Iris-virginica': [0, 0, 1]
    }
    
    data = pd.read_csv("iris.data", header=None)
    
    nums = data.iloc[:, :4]
    mapped_names = data[4].map(name_map)
    
    iris = pd.concat([nums, mapped_names], axis=1)
    iris_list = iris.values.tolist()
    random.shuffle(iris_list)
    
    patterns = []
    for row in iris_list:
        inputs = row[:4]
        expected = row[4]
        patterns.append((inputs, expected))
    
    train_iris_patterns = patterns[:100]
    test_iris_patterns = patterns[100:]
    
    network = MLP(architecture=[4, 5, 3], use_bias=True)
    trainer = Trainer(mlp=network, lr=0.6, momentum=0.0, shuffle=True)
    tester = Tester(mlp=network)
    
    trainer.train(patterns=train_iris_patterns, max_epochs=10000, target_error=0.01)
    tester.test(patterns=test_iris_patterns)

    # patterns = [
    #     ([1,0,0,0], [1,0,0,0]),
    #     ([0,1,0,0], [0,1,0,0]),
    #     ([0,0,1,0], [0,0,1,0]),
    #     ([0,0,0,1], [0,0,0,1]),
    # ]
    
    # TESTOWANIE ENKODER/DEKODER
    # for i, e in patterns:
    #     output = net.forward(i)
    #     print(f"Wejscie: {i}")
    #     print(f"Wyjscie: {[round(o, 3) for o in output]}")
    #     print(f"Oczekiwane: {e}\n")