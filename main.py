from functions import *
from mlp import *
from train import *
from test import *
import os

import pandas as pd
import random

NETWORK = None
LR = 0.0
EPOCHS = 1000
TARGET_ERROR = 0.01
MOMENTUM = 0.0
LOG_STEP_SIZE = 10
SHUFFLE = True

TRAINER = None
TESTER = None

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

def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

def printTrainMode(network):
    clear()

    global LR
    global EPOCHS
    global TARGET_ERROR
    global MOMENTUM
    global LOG_STEP_SIZE
    global SHUFFLE

    global TRAINER

    printNetworkParams()

    print("=== Tryb nauki ===")
    print("1. Ustaw współczynnik uczenia")
    print("2. Ustaw max liczbę epok")
    print("3. Ustaw docelowy poziom błędu")
    print("4. Ustaw momentum")
    print("5. Ustaw krok epok do logowania błędu")
    print("6. Ustaw przetasowanie")
    print("7. START")
    print("0. Wróć")

    choice = int(input("Wybór: "))

    match choice:
        case 1:
            clear()
            LR = float(input("Podaj współczynnik uczenia: "))
            printTrainMode(NETWORK)
        case 2:
            clear()
            EPOCHS = int(input("Podaj ilość epok: "))
            printTrainMode(NETWORK)
        case 3:
            clear()
            TARGET_ERROR = float(input("Podaj poziom błędu: "))
            printTrainMode(NETWORK)
        case 4:
            clear()
            MOMENTUM = float(input("Podaj momentum: "))
            printTrainMode(NETWORK)
        case 5:
            clear()
            LOG_STEP_SIZE = int(input("Podaj rozmiar kroku logowania błędu: "))
            printTrainMode(NETWORK)
        case 6:
            clear()
            shuffle_input = input("Czy przetasować? (Tak/Nie): ")
            if shuffle_input == "Tak":
                SHUFFLE = True
            else:
                SHUFFLE = False

            printTrainMode(NETWORK)
        case 7:
            clear()

            print("Wybierz zbiór danych do treningu: ")
            print("1. Irysy (zbiór treningowy)")
            print("2. Autoenkoder")
            dataset_choice = int(input("Wybór: "))

            if dataset_choice == 1:
                dataset_to_train = train_iris_patterns
            else:
                dataset_to_train = [
                    ([1, 0, 0, 0], [1, 0, 0, 0]),
                    ([0, 1, 0, 0], [0, 1, 0, 0]),
                    ([0, 0, 1, 0], [0, 0, 1, 0]),
                    ([0, 0, 0, 1], [0, 0, 0, 1]),
                ]

            TRAINER = Trainer(mlp=network, lr=LR, momentum=MOMENTUM, shuffle=SHUFFLE)
            TRAINER.train(patterns=dataset_to_train, max_epochs=EPOCHS, target_error=TARGET_ERROR, log_step=LOG_STEP_SIZE)

            printNetworkMenu()
        case 0:
            clear()
            printNetworkMenu()

def printTestMode():
    clear()

    global TESTER
    global NETWORK

    TESTER = Tester(NETWORK)

    printNetworkParams()

    print("=== Tryb testowania ===")
    print("1. Testuj na wzorcach treningowych")
    print("2. Testuj na wzorcach testowych")
    print("3. Tryb autoenkoder")
    print("4. Załaduj wzorce z pliku")
    print("0. Wróć")

    choice = int(input("Wybór: "))

    match choice:
        case 1:
            TESTER.test(train_iris_patterns)
            printTestMode()
        case 2:
            TESTER.test(test_iris_patterns)
            printTestMode()
        case 3:
            autoEncoder(NETWORK)
            printTestMode()
        case 4:
            filePath = input("Podaj nazwę pliku: ")
            patterns = loadPatternsFromFile(filePath)
            TESTER.test(patterns)
            printTestMode()
        case 0:
            clear()
            printNetworkMenu()

def printSaveMenu():
    clear()

    global NETWORK

    fileName = input("Podaj nazwę pliku: ")

    NETWORK.save(fileName)

def printNetworkMenu():
    clear()

    global NETWORK

    if NETWORK is None:
        print("Brak sieci. Stwórz ją lub wczytaj.")
        printMainMenu()
        return

    printNetworkParams()

    print("=== Menu sieci ===")
    print("1. Trenuj")
    print("2. Testuj")
    print("3. Zapisz sieć do pliku")
    print("0. Wróć")

    choice = int(input("Wybór: "))

    match choice:
        case 1:
            printTrainMode(NETWORK)
        case 2:
            printTestMode()
        case 3:
            printSaveMenu()
        case 0:
            printMainMenu()

def printCreateMLPMenu():
    clear()

    global NETWORK

    print("=== Nowa sieć ===")
    print("1. Podaj architekture")
    arch_input = input("Podaj architekture: ")
    architecture = [int(x.strip()) for x in arch_input.split(',')]

    print("2. Czy używać biasu?")
    bias_input = input("Tak lub Nie: ")
    if bias_input == "Tak":
        use_bias = True
    else:
        use_bias = False

    print("3. Wybierz funkcję aktywacji")
    activ_func_input = int(input("1. unipolarna, 2. bipolarna: "))
    if activ_func_input == 1:
        activ_func = sigmoidalUnipolar
    else:
        activ_func = sigmoidalBipolar

    NETWORK = MLP(architecture=architecture, use_bias=use_bias, activation_func=activ_func)
    printNetworkMenu()

def printLoadMLPMenu():
    clear()

    global NETWORK

    fileName = input("Podaj nazwę pliku: ")

    NETWORK = MLP.load(filepath=fileName)

    if NETWORK is not None:
        print("Pomyślnie wczytano sieć!")

    printNetworkMenu()

def printMainMenu():
    print("=== Menu Główne ===")
    print("1. Stwórz nową sieć")
    print("2. Wczytaj sieć z pliku")
    print("0. Wyjdź\n")

    choice = int(input("Wybór: "))

    match choice:
        case 1:
            printCreateMLPMenu()
        case 2:
            printLoadMLPMenu()
        case 0:
            exit(0)

def printNetworkParams():
    global NETWORK
    global LR
    global EPOCHS
    global TARGET_ERROR
    global MOMENTUM
    global LOG_STEP_SIZE
    global SHUFFLE

    print("=== Network ===")
    print(f"Współczynnik nauki: {LR}")
    print(f"Maks liczba epok: {EPOCHS}")
    print(f"Błąd docelowy: {TARGET_ERROR}")
    print(f"Momentum: {MOMENTUM}")
    print(f"Co ile epok logować błąd: {LOG_STEP_SIZE}")
    print(f"Czy przetasowac: {SHUFFLE}")

def autoEncoder(net):
    patterns = [
        ([1,0,0,0], [1,0,0,0]),
        ([0,1,0,0], [0,1,0,0]),
        ([0,0,1,0], [0,0,1,0]),
        ([0,0,0,1], [0,0,0,1]),
    ]

    with open("autoencoder_test.txt", "w") as f:
        for i, e in patterns:
            output = net.forward(i)

            hidden_layer = net.layers[0]
            hidden_outputs = [float(round(neuron.output, 3)) for neuron in hidden_layer.neurons]

            f.write(f"Wejscie: {i}\n")
            f.write(f"Warstwa ukryta: {hidden_outputs}\n")
            f.write(f"Wyjscie: {[float(round(o, 3)) for o in output]}\n")
            f.write(f"Oczekiwane: {e}\n\n")

def loadPatternsFromFile(filePath):
    patterns = []
    with open(filePath, "r", encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(";")
            inputs = [float(x) for x in parts[0].split(",")]
            expected = [float(x) for x in parts[1].split(",")]

            patterns.append((inputs, expected))

    return patterns

if __name__ == "__main__":
    printMainMenu()
