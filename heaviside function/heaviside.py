def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0

def xor(x1, x2):
    h1 = heaviside(x1 + x2 - 1.5)
    h2 = heaviside(x1 + x2 - 0.5)
    y = heaviside(h2 - h1 - 0.5)
    return y
print(xor(1,1))

import numpy as np


x1=np.array([
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    1,
])

x2=np.array([
    0,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
])

X=np.stack([x1,x2],axis=1)

y=np.array([
    [0],
    [0],
    [1],
    [1],
    [0],
    [0],
    [1],
    [1],
])

