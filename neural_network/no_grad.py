
import torch
from torchviz import make_dot

# simple weight
W = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
eta = 0.1

# forward
y = W * x
y.backward()

# manual update (tracked in graph)
W_updated = W - eta * W.grad

# visualize
dot = make_dot(W_updated, params={"W": W})
dot.render("without_no_grad", format="png")


