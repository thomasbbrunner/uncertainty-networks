
import numpy as np
import torch

class UncertaintyRNN(torch.nn.Module):
    """
    Sample implementations:
    https://github.com/KIT-ISAS/TrackSort_Neural_Public/tree/MFI2020
    """


# test input 1D
# some physics problem with incomplete knowledge
X = np.linspace(-10, 10, 1000).reshape(-1, 1)
train_indices = np.index_exp[:200, 300:700, 800:]
test_indices = np.index_exp[200:300, 700:800]
X_train = np.concatenate([X[i] for i in train_indices])
X_test = np.concatenate([X[i] for i in test_indices])
def y_func(x):
    return x * np.sin(x) + x * np.cos(2*x)
y = y_func(X)
y_train = y_func(X_train)
y_test = y_func(X_test)
