# %% Setup
import copy
import functools
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import List, Callable
import rslgym.algorithm.modules as rslgym_module

# reproducibility
torch.manual_seed(0)


class MonteCarloDropout(torch.nn.Dropout):
    def forward(self, x):
        # always have dropout enabled
        self.train()
        return super().forward(x)


class UncertaintyMLP(torch.nn.Module):
    """
    References:
    - "Dropout as a Bayesian Approximation" https://arxiv.org/abs/1506.02142
    - "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" https://arxiv.org/abs/1612.01474

    Sample implementations:
    https://xuwd11.github.io/Dropout_Tutorial_in_PyTorch/#5-dropout-as-bayesian-approximation


    Args:
        model_precision: user-defined value that encodes how well model should fit the data. This
            parameter is inversely proportional to weight regularization. It is called tau in the
            paper and tau > 0 (see Appendix 4.2).
    """
    # TODO important considerations:
    # - with MC Dropout, network should be larger
    # - weights should be initialized with different values
    # - account for variance in the loss
    # - (ideally) different order for data
    # - weight regularization did not help
    # - mc dropout paper used model precision derived from data, which was omitted here
    # - deep ensemble paper outputted parameters of gaussian to form GMM, which was omitted here
    # - (ideally) variance should be calibrated
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int],
        activation_func: Callable[..., torch.nn.Module],
        init_func: Callable,
        dropout_prob: float,
        num_passes: int,
        num_models: int,
        device: str,
    ):
        super().__init__()

        # TODO do parameter checks
        # assert model_precision > 0
        self._output_size = output_size
        self._dropout_prob = dropout_prob
        self._num_passes = num_passes
        self._num_models = num_models
        self._device = device

        # create model
        model = torch.nn.ModuleList()
        for hidden_size in hidden_sizes:
            model.append(torch.nn.Linear(input_size, hidden_size))
            model.append(activation_func())
            if dropout_prob > 0:
                model.append(MonteCarloDropout(dropout_prob))
            input_size = hidden_size
        model.append(torch.nn.Linear(input_size, self._output_size))

        # create ensemble
        self._models = torch.nn.ModuleList()
        for _ in range(self._num_models):
            self._models.append(copy.deepcopy(model))

        # re-initialize parameters to ensure diversity in each model of the ensemble
        self._models.apply(functools.partial(self._init_weights, init_func=init_func))

        self.to(device)

    @staticmethod
    def _init_weights(layer, init_func):
        if isinstance(layer, torch.nn.Linear):
            # reset weights *and* biases
            layer.reset_parameters()
            # overwrite weights if desired
            if init_func is not None:
                init_func(layer.weight)

    def forward(self, x, return_predictions=False):
        # include batch dimensions in predictions array
        y_preds =  torch.zeros(
            (self._num_models, self._num_passes, *x.shape[:-1], self._output_size),
            device=self._device)

        # iterate over Ensemble models
        for i in range(self._num_models):
            # iterate over passes of single MC Dropout model
            for j in range(self._num_passes):
                input = x
                for layer in self._models[i]:
                    input = layer(input)
                y_preds[i, j] = input

        # calculate mean and variance of models and passes
        y_mean = torch.mean(y_preds, dim=(0, 1))
        y_var = torch.var(y_preds, dim=(0, 1))

        if return_predictions:
            return y_mean, y_var, y_preds

        return y_mean, y_var


def train(X_train, y_train, model, epochs, device, shuffle, loss_type):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=shuffle,  # TODO hyperparameter
        batch_size=len(X_train))
    model.train()
    model.to(device)

    for epoch in range(epochs):
        for batch in dataloader:
            x, y = batch
            y_mean, y_var, y_preds = model(x, return_predictions=True)
            optimizer.zero_grad()

            # TODO hyperparameter
            # loss on mean
            if loss_type == "mean":
                loss = loss_fn(y_mean, y)
            # loss on mean with variance
            elif loss_type == "var":
                loss = loss_fn(y_mean, y) + 0.01*y_var.mean()
            # loss on each prediction
            elif loss_type == "pred":
                y_preds = y_preds.flatten(0, 1)
                y = y.unsqueeze(0).repeat_interleave(y_preds.shape[0], 0)
                loss = loss_fn(y_preds, y)

            loss.backward()
            optimizer.step()

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, loss.item()))


def test(X_test, y_test, model, device):
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        X_test = torch.Tensor(X_test).to(device)
        y_test = torch.Tensor(y_test).to(device)
        model.eval()
        model.to(device)
        mean, var, preds = model(X_test, return_predictions=True)
        loss = loss_fn(mean, y_test)
        return (
            mean.to("cpu").numpy(), 
            var.to("cpu").numpy(), 
            preds.to("cpu").numpy(), 
            loss.to("cpu").numpy())

# %% Training
# parameters
main_shape = (128, 128, 128)
activation_func = torch.nn.LeakyReLU
input_size = 1
output_size = 1
device = "cuda"
training_epochs = 20000
num_std_plot = 2
# TODO hyperparameters
shuffle = True
loss_type = "pred"
image_name_suffix = "_shuffle={} loss={}".format(shuffle, loss_type)

# test input 1D
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

# train Deterministic Baseline
mlp_0 = UncertaintyMLP(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=main_shape,
    activation_func=activation_func,
    init_func=None,
    dropout_prob=0,
    num_passes=1,
    num_models=1,
    device=device)
train(X_train, y_train, mlp_0, training_epochs, device, shuffle, "pred")
y_0, var_0, pred_0, loss_0 = test(X, y, mlp_0, device)

# train MC Dropout
mlp_1 = UncertaintyMLP(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=main_shape,
    activation_func=activation_func,
    init_func=None,
    dropout_prob=0.1,
    num_passes=10,
    num_models=1,
    device=device)
train(X_train, y_train, mlp_1, training_epochs, device, shuffle, loss_type)
y_1, var_1, pred_1, loss_1 = test(X, y, mlp_1, device)

# train Deep Ensamble
mlp_2 = UncertaintyMLP(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=main_shape,
    activation_func=activation_func,
    init_func=None,
    dropout_prob=0,
    num_passes=1,
    num_models=10,
    device=device)
train(X_train, y_train, mlp_2, training_epochs, device, shuffle, loss_type)
y_2, var_2, pred_2, loss_2 = test(X, y, mlp_2, device)

# train Deep Ensamble with MC Dropout
mlp_3 = UncertaintyMLP(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=main_shape,
    activation_func=activation_func,
    init_func=None,
    dropout_prob=0.1,
    num_passes=2,
    num_models=5,
    device=device)
train(X_train, y_train, mlp_3, training_epochs, device, shuffle, loss_type)
y_3, var_3, pred_3, loss_3 = test(X, y, mlp_3, device)

# train Deep Ensamble with MC Dropout
mlp_4 = UncertaintyMLP(
    input_size=input_size,
    output_size=output_size,
    hidden_sizes=main_shape,
    activation_func=activation_func,
    init_func=None,
    dropout_prob=0.1,
    num_passes=5,
    num_models=2,
    device=device)
train(X_train, y_train, mlp_4, training_epochs, device, shuffle, loss_type)
y_4, var_4, pred_4, loss_4 = test(X, y, mlp_4, device)

# %% Plotting
# Dataset and Deterministic Baseline
fig = plt.figure(dpi=300, figsize=(14, 7), constrained_layout=True)
axs = fig.subplots(2)
# plot function
axs[0].set_title("Ground Truth Data", loc="left")
axs[0].grid()
for i in train_indices:
    axs[0].plot(X[i], y[i], color="tab:blue")
for i in test_indices:
    axs[0].plot(X[i], y[i], color="tab:orange")
# plot baseline
axs[1].set_title("Deterministic Baseline" "\nTest Loss = {:.4}".format(loss_0), loc="left")
axs[1].grid()
for i in train_indices:
    axs[1].plot(X[i], y[i], color="tab:blue", linestyle="--")
for i in test_indices:
    axs[1].plot(X[i], y[i], color="tab:orange", linestyle="--")
axs[1].plot(X, y_0.flatten(), color="tab:red")
fig.savefig("uncertainty_mlp_baseline.png")


# Uncertainty MLPs
fig = plt.figure(dpi=300, figsize=(28, 14), constrained_layout=True)
axs = np.array(fig.subplots(4, 4))
y_plot = np.array([y_1.flatten(), y_2.flatten(), y_3.flatten(), y_4.flatten()])
std_plot = num_std_plot*np.sqrt(np.array([var_1.flatten(), var_2.flatten(), var_3.flatten(), var_4.flatten()]))
pred_plot = [pred_1, pred_2, pred_3, pred_4]
# plot function
for ax in axs.flatten():
    ax.grid()
    for i in train_indices:
        ax.plot(X[i], y[i], color="tab:blue", linestyle="--")
    for i in test_indices:
        ax.plot(X[i], y[i], color="tab:orange", linestyle="--")
# plot mean and std
axs[0, 0].set_title(r"Final Predictions ({}$\sigma$)".format(num_std_plot))
axs[0, 0].set_title(r"MC Dropout (10 passes)" "\nTest Loss = {:.4}".format(loss_1), loc="left")
for i, ax in enumerate(axs[:, 0]):
    ax.plot(X, y_plot[i], color="tab:red")
    ax.fill_between(X.flatten(), y_plot[i] - std_plot[i], y_plot[i] + std_plot[i], alpha=0.2, color="tab:grey")
# plot predictions
axs[0, 1].set_title(r"Individual Predictions")
axs[1, 0].set_title(r"Ensemble (10 models)" "\nTest Loss = {:.4}".format(loss_2), loc="left")
for i, ax in enumerate(axs[:, 1]):
    preds = pred_plot[i].reshape(-1, *pred_plot[i].shape[2:])
    for pred in preds:
        ax.plot(X, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
# plot predictions of Ensemlbes (mean over MC Dropout MLPs)
axs[0, 2].set_title(r"Individual Predictions of Ensembles")
axs[2, 0].set_title(r"Ensemble (5 models) MC Dropout (2 passes)" "\nTest Loss = {:.4}".format(loss_3), loc="left")
for i, ax in enumerate(axs[:, 2]):
    preds = pred_plot[i].mean(axis=1)
    for pred in preds:
        ax.plot(X, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
# plot predictions of MC Dropout MLPs (mean over Ensembles)
axs[0, 3].set_title(r"Individual Predictions of MC Dropouts")
axs[3, 0].set_title(r"Ensemble (2 models) MC Dropout (5 passes)" "\nTest Loss = {:.4}".format(loss_4), loc="left")
for i, ax in enumerate(axs[:, 3]):
    preds = pred_plot[i].mean(axis=0)
    for pred in preds:
        ax.plot(X, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
fig.savefig("uncertainty_mlp{}.png".format(image_name_suffix))
