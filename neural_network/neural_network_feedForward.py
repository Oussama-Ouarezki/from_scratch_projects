import numpy as np



x1=lambda x: (np.random.random(x) >=0.5).astype(int)

X=np.column_stack([x1(100),x1(100)])

y=[]

x_flat=X.flatten()

for i in range(0,200,2):
    if x_flat[i]==x_flat[i+1]:
        y.append(0)
    else:
        y.append(1)

y=np.array(y)
y=y.reshape(100,1)

X=np.delete(X,2,axis=1)
#all_table=np.column_stack([X,y])

w1=np.array([[ 1,1 ],
    [1,1]])
b1=np.array([[-1.5],
             [-0.5]])

w2=np.array([[-1,1]])
b2=np.array([[-0.5]])

def heaviside(x):
    return (x>=0).astype(int)

def relu(x):
    return np.maximum(x,0)

def feedForward(w1,w2,b1,b2,x):
    x1=w1@x.T+b1
    x2=heaviside(x1)
    x3=w2@x2
    return x3

row1=X[0:1,:]
feedForward(w1,w2,b1,b2,row1)

