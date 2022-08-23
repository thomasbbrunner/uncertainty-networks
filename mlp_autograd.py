"""
This script is to test if there's a need to use clone or detach when passing
the same input multiple times through a network.

Result of this experiment:
"""
import numpy as np
import torch

device = "cuda"
num_passes = 5

# generate dataset
X = torch.Tensor(np.linspace(-10, 10, 100)[..., None]).to(device)
Y = torch.sin(X).to(device)

class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1)).to(device)

    def forward(self, x):
        self.train()
        predictions = torch.zeros((num_passes, *x.shape), device=device)
        for i in range(num_passes):
            input = x  # TODO clone? detach?
            predictions[i] = self.model(input)
        return predictions


# repeat desired output to match shape of predictions
# (num_passes, ...)
Y_repeated = Y.unsqueeze(0).repeat_interleave(num_passes, 0)
model = MLP()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(100):
    optimizer.zero_grad()
    preds = model(X)
    loss = loss_fn(preds, Y_repeated)
    loss.backward()
    optimizer.step()

"""

# text
# multiple forward passes
x = [...]
outputs = []
for i in range(num_passes):
    input = x  # TODO clone? detach()
    outputs.append(model(input))

# backpropagation through outputs
(...)
"""