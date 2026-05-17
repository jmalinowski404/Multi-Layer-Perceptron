import numpy as np;

def sigmoidalUnipolar(x):
    return 1 / (1 + np.exp(-x));

def sigmoidalBipolar(x):
    return np.tanh(x);