import numpy as np;

def sigmoidalUnipolar(x):
    if (x < -700): return 0.0;
    if (x > 700): return 1.0;

    return 1 / (1 + np.exp(-x));

def sigmoidalBipolar(x):
    return np.tanh(x);