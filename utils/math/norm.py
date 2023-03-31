import numpy as np

def l2_norm(v: list):
    return np.linalg.norm(v)

def l1_norm(v: list):
    return sum([np.abs(item) for item in v])

def squared_l2_norm(v: list):
    return np.dot(v, v)

def max_norm(v: list):
    return np.max([np.abs(item) for item in v])
