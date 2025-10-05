

import numpy as np

 


import matplotlib.pyplot as plt




np.random.seed(0)

X0 = np.random.randn(50, 1) + 2 
y0 = np.zeros((50, 1))




X1 = np.random.randn(50, 1) + 7
y1 = np.ones((50, 1))

X = np.vstack((X0, X1))           # Shape: (100, 1)
X = np.hstack((X, np.ones((100, 1))))  # Add bias column




y = np.vstack((y0, y1))           # Shape: (100, 1)

class Logistic:
    def __init__(self, eta=0.1, max_iter=10000):  # Fixed: double underscores
        self.eta = eta
        self.max_iter = max_iter
        self.w = None
    
    def sigmoid(self, x):
        x_clipped = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x_clipped))
    
    def predict(self, x):
        a = x @ self.w
        return self.sigmoid(a)
    
    def fit(self, x, y):
        self.w = np.zeros((x.shape[1], 1))
        for _ in range(self.max_iter):
            U = self.predict(x)
            grad = x.T @ (U - y)
            self.w = self.w - self.eta * grad

model = Logistic()
model.fit(X, y)
predicted_probs = model.predict(X)  # Fixed: use the correct variable name

plt.subplot( , 2, 2)
plt.scatter(X[:, 0], predicted_probs.ravel(), c=(predicted_probs > 0.5).astype(int).ravel(), cmap='bwr', edgecolors='k')
plt.title("Predicted Probabilities")
plt.xlabel("Feature 1")
plt.ylabel("Predicted Probability")
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Final weights: {model.w.ravel()}")
print(f"Accuracy: {np.mean((predicted_probs > 0.5) == y):.3f}")
