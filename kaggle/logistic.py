import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate data
x0 = np.random.random(10) - 9    
x1 = np.random.random(10)
x2 = np.random.random(10) + 9

yA = np.ones(10) - 1  # class 0
yB = np.ones(10)      # class 1
yC = np.ones(10) + 1  # class 2

X = np.concatenate([x0, x1, x2])    # shape (30,)
bias = np.ones(30)                  # shape (30,)
X = np.column_stack((X, bias))      # shape (30, 2)
y = np.concatenate([yA, yB, yC])    # class labels


n_classes=3
N=30
nbr_parameters=2
Y=np.zeros((30,3 ))

for i in range(N):
    Y[i,int(y[i])]=1


W=np.ones((nbr_parameters,n_classes))

test=np.array([[1,2,3],[4 ,5 ,6]])

def softmax(a):
    z=a-np.max(a,axis=1,keepdims=True)
    Z=np.exp(z)
    return Z/np.sum(Z,axis=1,keepdims=True)

max_iter=100000
eta=0.01

for i in range(max_iter):
    a=X@W
    U=softmax(a)
    grad=(1/N)*X.T@(U-Y)
    W_new=W-eta*grad
    W=W_new


np.round(U)
