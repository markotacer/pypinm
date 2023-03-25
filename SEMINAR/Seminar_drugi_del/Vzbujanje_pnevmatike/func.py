import numpy as np;

def sum(a, b):
    return a + b
def razlika(a, b):
    return a - b

def neodvisna_spremenljivka_t(t):
    ele = t * 1000;
    t = np.linspace(0, ele/1000, ele + 1)
    return t;