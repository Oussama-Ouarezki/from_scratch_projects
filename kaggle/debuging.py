import numpy as np

# Example 1: Your softmax (works for single sample)
def softmax_single(a):
    """Your version - works for single sample"""
    return np.exp(a) / np.sum(np.exp(a))

# Example 2: Multinomial softmax (works for multiple samples)
def softmax_multi(Z):
    """Correct version for multiple samples"""
    # Z has shape (n_samples, n_classes)
    exp_Z = np.exp(Z)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

print("=== SINGLE SAMPLE EXAMPLE ===")
# Single sample with 3 class scores
single_scores = np.array([2.0, 1.0, 0.5])
print("Input scores:", single_scores)
print("Your softmax result:", softmax_single(single_scores))
print("Probabilities sum to:", np.sum(softmax_single(single_scores)))

print("\n=== MULTIPLE SAMPLES EXAMPLE ===")
# 4 samples, each with 3 class scores
multi_scores = np.array([
    [2.0, 1.0, 0.5],  # sample 1
    [0.1, 2.5, 1.0],  # sample 2  
    [1.5, 0.8, 2.2],  # sample 3
    [0.3, 0.7, 1.9]   # sample 4
])
print("Input scores shape:", multi_scores.shape)
print("Input scores:")
print(multi_scores)

# Apply softmax to all samples at once
probs = softmax_multi(multi_scores)
print("\nSoftmax probabilities:")
print(probs)
print("\nEach row sums to:", np.sum(probs, axis=1))

print("\n=== WHY YOUR APPROACH DIDN'T WORK ===")
# Simulate your original setup
np.random.seed(42)  # for reproducible example
X = np.random.randn(5, 2)  # 5 samples, 2 features
w1 = np.random.randn(2, 1)  # weights for single output

print("X shape:", X.shape)
print("w1 shape:", w1.shape)

single_output = X @ w1
print("X @ w1 shape:", single_output.shape)
print("X @ w1 values:", single_output.flatten())

print("\nProblem: You only have 1 score per sample, but need 3 for 3 classes!")
print("You can't apply softmax meaningfully to single numbers.")

print("\n=== CORRECT MULTINOMIAL SETUP ===")
# Need weights for all 3 classes
W_correct = np.random.randn(2, 3)  # 2 features -> 3 classes
print("Correct W shape:", W_correct.shape)

multi_output = X @ W_correct
print("X @ W shape:", multi_output.shape)
print("X @ W values:")
print(multi_output)

# Now softmax works properly
probs_correct = softmax_multi(multi_output) 
print("\nProper softmax probabilities:")
print(probs_correct)
print("Each row sums to:", np.sum(probs_correct, axis=1))

# Make predictions
predictions = np.argmax(probs_correct, axis=1)
print("Predicted classes:", predictions)
