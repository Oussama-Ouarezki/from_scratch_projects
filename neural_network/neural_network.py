
import numpy as np
     
x1 = lambda x: (np.random.random(x) >= 0.5).astype(int)
X = np.column_stack([x1(100), x1(100)])
y = []
x_flat = X.flatten()
for i in range(0, 200, 2):
    if x_flat[i] == x_flat[i+1]:
        y.append(0)
    else:
        y.append(1)
y = np.array(y).reshape(100, 1)

np.random.seed(42)
w1 = np.random.normal(0, 1, (2, 2))
b1 = np.random.normal(0, 1, (2, 1))
w2 = np.random.normal(0, 1, (1, 2))
b2 = np.random.normal(0, 1, (1, 1))

def relu(x):
    return np.maximum(x, 0)

def drelu(x):
    return (x > 0).astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def loss(y, estimate):
    epsilon = 1e-15
    estimate = np.clip(estimate, epsilon, 1 - epsilon)
    return -np.mean(y * np.log(estimate) + (1 - y) * np.log(1 - estimate))

def feedForward(x):
    x1 = x @ w1.T + b1.T
    x2 = relu(x1)
    x3 = x2 @ w2.T + b2.T
    x4 = sigmoid(x3)
    return x4, x3, x2, x1

eta = 0.5
batch_size = X.shape[0]

print(f"Initial loss: {loss(y, feedForward(X)[0]):.4f}")

for epoch in range(5000):
    est, x3, x2, x1 = feedForward(X)
    
    Ldx4 = est - y
    Ldw2 = Ldx4.T @ x2 / batch_size
    Ldb2 = np.mean(Ldx4, axis=0, keepdims=True)
    Ldx2 = (Ldx4 @ w2) * drelu(x1)
    Ldw1 = Ldx2.T @ X / batch_size
    Ldb1 = np.mean(Ldx2, axis=0, keepdims=True).T
    
    w2 -= eta * Ldw2
    b2 -= eta * Ldb2
    w1 -= eta * Ldw1
    b1 -= eta * Ldb1
    
    if epoch % 500 == 0:
        current_loss = loss(y, est)
        accuracy = np.mean((est > 0.5).astype(int) == y)
        print(f"Epoch {epoch}, Loss: {current_loss:.4f}, Accuracy: {accuracy:.4f}")

final_est, _, _, _ = feedForward(X)
final_loss = loss(y, final_est)
accuracy = np.mean((final_est > 0.5).astype(int) == y)
