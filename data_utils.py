import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def make_dataset(seed):
    np.random.seed(seed)
    w = np.random.rand(10)
    b = np.random.rand(1000)
    
    p = len(w); n = len(b)
    
    x = np.random.rand(n, p)
    z = np.dot(x, w) + b
    prob = sigmoid(z)
    y = prob.round()
    print(w)
    return x, y

x, y = make_dataset(seed = 42)