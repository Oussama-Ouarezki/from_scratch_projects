import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate data for 3 classes
x0 = np.random.random(10) - 9    
x1 = np.random.random(10)
x2 = np.random.random(10) + 9

yA = np.zeros(10)  # class 0
yB = np.ones(10)   # class 1
yC = np.full(10, 2)  # class 2

X = np.concatenate([x0, x1, x2])[:, np.newaxis]  # shape (30,1)
bias = np.ones((30, 1))
X = np.hstack((X, bias))  # shape (30, 2)

y = np.concatenate([yA, yB, yC]).astype(int)  # shape (30,)

# One-hot encoding
N = X.shape[0]
n_classes = 3
Y = np.zeros((N, n_classes))
for i in range(N):
    Y[i, y[i]] = 1

# Softmax function
def softmax(a):
    z = a - np.max(a, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Gradient descent
max_iter = 100000
eta = 0.01
nbr_parameters = X.shape[1]

W = np.zeros((nbr_parameters, n_classes))
W[:, :2] = np.random.randn(nbr_parameters, 2) + 19900 

for i in range(max_iter):
    a = X @ W
    U = softmax(a)
    lambda_ = 0.1
    grad = (1/N) * X.T @ (U - Y) + lambda_ * W
    grad[:, -1] = 0  # freeze class 2
    W = W - eta * grad

# Final prediction
pred_probs = softmax(X @ W)
pred_labels = np.argmax(pred_probs, axis=1)

print("Final predictions:", pred_labels)
print("True labels:      ", y.astype(int))
