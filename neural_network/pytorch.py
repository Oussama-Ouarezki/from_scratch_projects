import torch

import torch
print(torch.cuda.is_available())   # True if GPU available
print(torch.cuda.device_count())   # number of GPUs
print(torch.cuda.get_device_name(0))  # name of first GPU (if exists)

N=10
d_in=1
d_out=1

X=torch.randn(N,d_in)

true_w=torch.tensor([[ 2.0 ]])
true_b=torch.tensor(1.0)

y_true=X@true_w+true_b+torch.randn(N,d_out)*0.1

W=torch.randn(d_in,d_out,requires_grad=True)
b=torch.randn(1,requires_grad=True)

y_hat=X@W+b

loss=((y_hat-y_true)**2).mean()

loss.backward()

W.grad
b.grad

eta=0.01

for i in range(2000):
    y_hat = X @ W + b
    loss=((y_hat-y_true)**2).mean()
    loss.backward()

    with torch.no_grad():# pytorch tracks every operation in tensor when we do this we don't want it to trace it  
        W-=eta*W.grad
        b-=eta*b.grad

    W.grad.zero_() # if we don't do this the gradient just add up to each others
    b.grad.zero_()


linear_layer=torch.nn.Linear(in_features=d_in,out_features=d_out)

linear_layer.weight
linear_layer.bias


y_hat_nn=linear_layer(X)

### relu layer ###
relu=torch.nn.ReLU()
relu(X)
