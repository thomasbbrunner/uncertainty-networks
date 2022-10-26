
from bdb import Breakpoint
from uncertainty_networks import UncertaintyMLP
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, TensorDataset

# reproducibility
torch.manual_seed(0)
np.random.seed(0)


def train(T_train, X_train, model, epochs, batch_size, device, var_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
    T_train = torch.Tensor(T_train).to(device)
    X_train = torch.Tensor(X_train).to(device)
    ndims = X_train.shape[-1]
    dataset = TensorDataset(T_train, X_train)
    sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_indices in sampler:
            optimizer.zero_grad()
            T_batch, X_batch = dataset[batch_indices]

            output = model(T_batch)

            if var_type == "split_var":
                preds, logvar = torch.split(output, [ndims, ndims], dim=-1)
                logvar_clip = torch.maximum(logvar, torch.log(torch.tensor(1e-6)))
                loss = torch.exp(-logvar_clip)*(preds - X_batch).square() + logvar_clip
                loss = 0.5*loss.mean()
            elif var_type == "common_var":
                preds, logvar, _ = torch.split(output, [ndims, 1, ndims-1], dim=-1)
                # logvar = logvar.expand_as(preds)
                logvar_clip = torch.maximum(logvar, torch.log(torch.tensor(1e-6)))
                loss = torch.exp(-logvar_clip)*(preds - X_batch).square() + logvar_clip
                loss = 0.5*loss.mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if True:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss))


def test(T_test, X_test, model, device, var_type):
    with torch.no_grad():
        T_test = torch.Tensor(T_test).to(device)
        X_test = torch.Tensor(X_test).to(device)
        ndims = X_test.shape[-1]
        model.eval()
        model.to(device)

        output = model(T_test)

        if var_type == "split_var":
            preds, data_logvar = torch.split(output, [ndims, ndims], dim=-1)
        elif var_type == "common_var":
            preds, data_logvar, _ = torch.split(output, [ndims, 1, ndims-1], dim=-1)
            data_logvar = data_logvar.expand_as(preds)

        model_var, mean = torch.var_mean(preds, dim=(0, 1), unbiased=False)
        data_var = torch.mean(torch.exp(data_logvar), dim=(0, 1))
        var = model_var + data_var
        loss = (mean - X_test).square().mean()

        return (
            mean.to("cpu").numpy(),
            var.to("cpu").numpy(),
            model_var.to("cpu").numpy(),
            data_var.to("cpu").numpy(),
            preds.to("cpu").numpy(),
            loss.to("cpu").numpy())

# Training
# parameters
main_shape = (64, 64, 64, 64)
input_size = 1
output_size = 24
ndims = output_size//2
device = "cuda"
epochs = 1000
batch_size = 10000
num_std_plot = 2
shuffle = True
var_type = "common_var" #"split_var"
use_jit = True
noise_std = 0.5
image_name_suffix = "{}".format(var_type)

# generate dataset
T = np.linspace(-20, 20, 100000).reshape(-1, 1)
missing_region = (np.array([[0.0, 0.1], [0.9, 1.0]])*T.shape[0]).astype(int)
noisy_regions = (np.array([[0.6, 0.8]])*T.shape[0]).astype(int)
missing_mask = np.full(T.shape[0], False, dtype=bool)
for region in missing_region:
    missing_mask[region[0]:region[1]] = True
noisy_mask = np.full(T.shape[0], False, dtype=bool)
for region in noisy_regions:
    noisy_mask[region[0]:region[1]] = True
def norm(x):
    return 2*(x - x.min())/(x.max() - x.min()) - 1
funcs = [
    lambda t: t,
    lambda t: np.sin(0.7*t+1) + 0.5*np.cos(1.2*t-2),
    lambda t: np.exp(0.1*t),
    lambda t: np.cos(0.7*t+1) + 0.5*np.sin(1.2*t-2),
    lambda t: np.square(0.5*t),
    lambda t: np.abs(t),
    lambda t: np.sqrt(np.abs(0.5*t)),
    lambda t: t**3 + 1,
    lambda t: -np.square(t),
    lambda t: np.sign(t)]
def func(t):
    X = np.zeros((T.shape[0], ndims))
    for i in range(ndims):
        X[:, i] = norm(np.random.choice(funcs)(t)).flatten()
    return X
X = func(T)

T_train = T.copy()
X_train = X.copy()
X_train[noisy_mask] += noise_std*np.random.randn(*X_train[noisy_mask].shape)
T_train = T_train[~missing_mask]
X_train = X_train[~missing_mask]

# train Deep Ensamble with MC Dropout
model = UncertaintyMLP(
    input_size=input_size,
    hidden_sizes=main_shape,
    output_size=output_size,
    dropout_prob=0.1,
    num_passes=5,
    num_models=4,
    initialization="sl",
    activation=torch.nn.LeakyReLU,
    device=device)
model = torch.jit.script(model) if use_jit else model
train(T_train, X_train, model, epochs, batch_size, device, var_type)
mean, var, model_var, data_var, preds, loss = test(T, X, model, device, var_type)
stdx = num_std_plot*np.sqrt(var)
model_stdx = num_std_plot*np.sqrt(model_var)
data_stdx = num_std_plot*np.sqrt(data_var)

# Plotting
# predictions 2d wrt T
axis_indices = range(ndims)
fig = plt.figure(dpi=300, figsize=(15, 3*ndims), constrained_layout=True)
axs = fig.subplots(ndims+1, 3, sharex=True, sharey=True)
for ax, i in zip(axs[:,0], axis_indices):
    ax.set_ylabel("x"+str(i))
for ax, i in zip(axs[-1,:], axis_indices):
    ax.set_xlabel("t")
# gt
for row_ax, i in zip(axs, axis_indices):
    for j, ax in enumerate(row_ax):
        ax.grid()
        linestyle = "--"
        ax.plot(
            np.where(np.logical_or(noisy_mask, missing_mask), np.nan, T.flatten()),
            np.where(np.logical_or(noisy_mask, missing_mask), np.nan, X[:,i].flatten()),
            color="tab:blue", linestyle=linestyle)
        ax.plot(
            np.where(missing_mask, T.flatten(), np.nan),
            np.where(missing_mask, X[:,i].flatten(), np.nan),
            color="tab:orange", linestyle=linestyle)
        ax.plot(
            np.where(noisy_mask, T.flatten(), np.nan),
            np.where(noisy_mask, X[:,i].flatten(), np.nan),
            color="tab:green", linestyle=linestyle)
# plot mean and total var
axs[0,0].set_title("Mean and Total Uncertainty\nTest Loss = {:.4}".format(loss), loc="left")
for ax, i in zip(axs[:,0], axis_indices):
    ax.plot(T, mean[:,i], alpha=0.5, color="tab:red")
    ax.fill_between(T.flatten(), mean[:,i] - stdx[:,i], mean[:,i] + stdx[:,i], alpha=0.2, color="tab:grey")
# plot mean and model var
axs[0,1].set_title("Mean and Model Uncertainty", loc="left")
for ax, i in zip(axs[:,1], axis_indices):
    ax.plot(T, mean[:,i], alpha=0.5, color="tab:red")
    ax.fill_between(T.flatten(), mean[:,i] - model_stdx[:,i], mean[:,i] + model_stdx[:,i], alpha=0.2, color="tab:grey")
# plot mean and data var
axs[0,2].set_title("Mean and Data Uncertainty", loc="left")
for ax, i in zip(axs[:,2], axis_indices):
    ax.plot(T, mean[:,i], alpha=0.5, color="tab:red")
    ax.fill_between(T.flatten(), mean[:,i] - data_stdx[:,i], mean[:,i] + data_stdx[:,i], alpha=0.2, color="tab:grey")
# plot total vars
for ax in axs[-1]:
    ax.grid()
    linestyle = None
    ax.plot(
        np.where(np.logical_or(noisy_mask, missing_mask), np.nan, T.flatten()),
        np.where(np.logical_or(noisy_mask, missing_mask), np.nan, 0),
        color="tab:blue", linestyle=linestyle)
    ax.plot(
        np.where(missing_mask, T.flatten(), np.nan),
        np.where(missing_mask, 0, np.nan),
        color="tab:orange", linestyle=linestyle)
    ax.plot(
        np.where(noisy_mask, T.flatten(), np.nan),
        np.where(noisy_mask, 0, np.nan),
        color="tab:green", linestyle=linestyle)
axs[-1,0].set_ylabel(r"{}$\sigma$".format(num_std_plot))
axs[-1,0].plot(T, np.sum(stdx, axis=1), alpha=0.5, color="tab:grey")
axs[-1,1].plot(T, np.sum(model_stdx, axis=1), alpha=0.5, color="tab:grey")
axs[-1,2].plot(T, np.sum(data_stdx, axis=1), alpha=0.5, color="tab:grey")
axs[0,0].set_ylim([-5, 20])
fig.savefig("mlp_model_data_xd_{}.png".format(image_name_suffix))
