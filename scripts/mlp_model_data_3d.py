
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
                preds, logvar = torch.split(output, [3, 3], dim=-1)
                loss = torch.exp(-logvar)*(preds - X_batch).square() + logvar
                loss = loss.mean()
            elif var_type == "common_var":
                preds, logvar, _ = torch.split(output, [3, 1, 2], dim=-1)
                loss = torch.exp(-logvar)*(preds - X_batch).square() + logvar
                loss = loss.mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if True: #epoch % 1000 == 0 or epoch == epochs - 1:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss))


def test(T_test, X_test, model, device, var_type):
    with torch.no_grad():
        T_test = torch.Tensor(T_test).to(device)
        X_test = torch.Tensor(X_test).to(device)
        model.eval()
        model.to(device)

        output = model(T_test)

        if var_type == "split_var":
            preds, data_logvar = torch.split(output, [3, 3], dim=-1)
        elif var_type == "common_var":
            preds, data_logvar, _ = torch.split(output, [3, 1, 2], dim=-1)
            data_logvar = data_logvar.expand_as(preds)

        model_var, mean = torch.var_mean(preds, dim=(0, 1), unbiased=False)
        data_var = torch.exp(torch.mean(data_logvar, dim=(0, 1)))
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
output_size = 6
device = "cuda"
epochs = 500
batch_size = 10000
num_std_plot = 2
shuffle = True
var_type = "common_var" #"split_var"
use_jit = True
noise_std = 0.1
image_name_suffix = "_var={}".format(var_type)

# generate dataset
T = np.linspace(-20, 20, 100000).reshape(-1, 1)
missing_region = (np.array([[0.0, 0.1], [0.9, 1.0]])*T.shape[0]).astype(int)
x1_noisy_region = (np.array([[0.6, 0.8]])*T.shape[0]).astype(int)
x2_noisy_region = (np.array([[0.6, 0.8]])*T.shape[0]).astype(int)
x3_noisy_region = (np.array([[0.6, 0.8], [0.35, 0.5]])*T.shape[0]).astype(int)
missing_mask = np.full(T.shape[0], False, dtype=bool)
for region in missing_region:
    missing_mask[region[0]:region[1]] = True
x1_noisy_mask = np.full(T.shape[0], False, dtype=bool)
x2_noisy_mask = np.full(T.shape[0], False, dtype=bool)
x3_noisy_mask = np.full(T.shape[0], False, dtype=bool)
for region in x1_noisy_region:
    x1_noisy_mask[region[0]:region[1]] = True
for region in x2_noisy_region:
    x2_noisy_mask[region[0]:region[1]] = True
for region in x3_noisy_region:
    x3_noisy_mask[region[0]:region[1]] = True
common_noisy_mask = np.logical_or(np.logical_or(x1_noisy_mask, x2_noisy_mask), x3_noisy_mask)
def norm(x):
    return 2*(x - x.min())/(x.max() - x.min()) - 1
def func(t):
    x = t
    y = np.sin(0.7*t+1) + 0.5*np.cos(1.2*t-2)
    z = np.exp(0.1*t)
    return x, y, z
X1, X2, X3 = func(T)
# normalize ranges
X1, X2, X3 = norm(X1), norm(X2), norm(X3)
# T, X1, X2, X3 = norm(T), norm(X1), norm(X2), norm(X3)

T_train = T.copy()
X1_train = X1.copy()
X2_train = X2.copy()
X3_train = X3.copy()
X1_train[x1_noisy_mask] += noise_std*np.random.randn(*X1_train[x1_noisy_mask].shape)
X2_train[x2_noisy_mask] += noise_std*np.random.randn(*X2_train[x2_noisy_mask].shape)
X3_train[x3_noisy_mask] += noise_std*np.random.randn(*X3_train[x3_noisy_mask].shape)
T_train = T_train[~missing_mask]
X1_train = X1_train[~missing_mask]
X2_train = X2_train[~missing_mask]
X3_train = X3_train[~missing_mask]
# join into single output tensor
X = np.concatenate((X1, X2, X3), axis=-1)
X_train = np.concatenate((X1_train, X2_train, X3_train), axis=-1)

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
# dataset
fig = plt.figure(dpi=300, figsize=(12, 12), constrained_layout=True)
axs = fig.subplots(2, 2, subplot_kw={"projection": "3d"}).flatten()
for i, ax in enumerate(axs):
    ax.grid()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.azim = -120
    if i == 1:
        continue
    linestyle = "--" if i != 0 else None
    ax.plot(
        np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, X1.flatten()),
        np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, X2.flatten()),
        np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, X3.flatten()),
        color="tab:blue", linestyle=linestyle)
    ax.plot(
        np.where(missing_mask, X1.flatten(), np.nan),
        np.where(missing_mask, X2.flatten(), np.nan),
        np.where(missing_mask, X3.flatten(), np.nan),
        color="tab:orange", linestyle=linestyle)
    ax.plot(
        np.where(common_noisy_mask, X1.flatten(), np.nan),
        np.where(common_noisy_mask, X2.flatten(), np.nan),
        np.where(common_noisy_mask, X3.flatten(), np.nan),
        color="tab:green", linestyle=linestyle)
axs[0].set_title("Ground Truth Data", loc="left")
# samples
size = 1
axs[1].set_title("Training Data", loc="left")
axs[1].scatter(X_train[:,0], X_train[:,1], X_train[:,2], color="tab:blue", marker=".", s=size)
# mean
axs[2].set_title("Mean \nTest Loss = {:.4}".format(loss), loc="left")
axs[2].plot(mean[:,0], mean[:,1], mean[:,2], alpha=0.5, color="tab:red")
# preds
axs[3].set_title("Predictions", loc="left")
preds_plot = preds.reshape(-1, *preds.shape[2:])
for pred in preds_plot:
    axs[3].plot(pred[:,0], pred[:,1], pred[:,2], color="tab:red", alpha=np.maximum(0.2, 1/len(preds_plot)))
fig.savefig("mlp_model_data_3d{}.png".format(image_name_suffix))

# predictions 2d
axis_indices = [[0, 1], [2, 1], [0, 2]]
axis_labels = {0: "x", 1: "y", 2: "z"}
fig = plt.figure(dpi=300, figsize=(15, 10), constrained_layout=True)
axs = fig.subplots(3, 3, sharex=True, sharey=True)
# gt
for row_ax, i in zip(axs, axis_indices):
    for ax in row_ax:
        ax.grid()
        ax.set_xlabel(axis_labels[i[0]])
        ax.set_ylabel(axis_labels[i[1]])
        ax.plot(
            np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, X[:,i[0]].flatten()),
            np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, X[:,i[1]].flatten()),
            color="tab:blue", linestyle="--")
        ax.plot(
            np.where(missing_mask, X[:,i[0]].flatten(), np.nan),
            np.where(missing_mask, X[:,i[1]].flatten(), np.nan),
            color="tab:orange", linestyle="--")
        ax.plot(
            np.where(common_noisy_mask, X[:,i[0]].flatten(), np.nan),
            np.where(common_noisy_mask, X[:,i[1]].flatten(), np.nan),
            color="tab:green", linestyle="--")
# plot mean and total var
axs[0,0].set_title("Mean and Total Uncertainty\nTest Loss = {:.4}".format(loss), loc="left")
for ax, i in zip(axs[:,0], axis_indices):
    ax.plot(mean[:,i[0]], mean[:,i[1]], alpha=0.5, color="tab:red")
    ax = ax.twinx()
    ax.set_ylabel(r"{}$\sigma$".format(num_std_plot))
    ax.set_ylim([0, 1])
    ax.plot(mean[:,i[0]], stdx[:,i[1]], alpha=0.2, color="tab:grey")
# plot mean and model var
axs[0,1].set_title("Mean and Model Uncertainty", loc="left")
for ax, i in zip(axs[:,1], axis_indices):
    ax.plot(mean[:,i[0]], mean[:,i[1]], alpha=0.5, color="tab:red")
    ax = ax.twinx()
    ax.set_ylabel(r"{}$\sigma$".format(num_std_plot))
    ax.set_ylim([0, 1])
    ax.plot(mean[:,i[0]], model_stdx[:,i[1]], alpha=0.2, color="tab:grey")
# # plot mean and data var
axs[0,2].set_title("Mean and Data Uncertainty", loc="left")
for ax, i in zip(axs[:,2], axis_indices):
    ax.plot(mean[:,i[0]], mean[:,i[1]], alpha=0.5, color="tab:red")
    ax = ax.twinx()
    ax.set_ylabel(r"{}$\sigma$".format(num_std_plot))
    ax.set_ylim([0, 1])
    ax.plot(mean[:,i[0]], data_stdx[:,i[1]], alpha=0.2, color="tab:grey")
fig.savefig("mlp_model_data_2d{}.png".format(image_name_suffix))

# predictions 2d wrt T
axis_indices = [0, 1, 2]
axis_labels = {0: "x", 1: "y", 2: "z"}
fig = plt.figure(dpi=300, figsize=(15, 10), constrained_layout=True)
axs = fig.subplots(4, 3, sharex=True, sharey=True)
for ax, i in zip(axs[:,0], axis_indices):
    ax.set_ylabel(axis_labels[i])
for ax, i in zip(axs[-1,:], axis_indices):
    ax.set_xlabel("t")
# gt
for row_ax, i in zip(axs, axis_indices):
    for j, ax in enumerate(row_ax):
        ax.grid()
        linestyle = "--" # None
        ax.plot(
            np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, T.flatten()),
            np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, X[:,i].flatten()),
            color="tab:blue", linestyle=linestyle)
        ax.plot(
            np.where(missing_mask, T.flatten(), np.nan),
            np.where(missing_mask, X[:,i].flatten(), np.nan),
            color="tab:orange", linestyle=linestyle)
        ax.plot(
            np.where(common_noisy_mask, T.flatten(), np.nan),
            np.where(common_noisy_mask, X[:,i].flatten(), np.nan),
            color="tab:green", linestyle=linestyle)
# plot mean and total var
axs[0,0].set_title("Mean and Total Uncertainty\nTest Loss = {:.4}".format(loss), loc="left")
for ax, i in zip(axs[:,0], axis_indices):
    ax.plot(T, mean[:,i], alpha=0.5, color="tab:red")
    ax.fill_between(T.flatten(), (mean[:,i] - stdx[:,i]).flatten(), (mean[:,i] + stdx[:,i]).flatten(), alpha=0.2, color="tab:grey")
# plot mean and model var
axs[0,1].set_title("Mean and Model Uncertainty", loc="left")
for ax, i in zip(axs[:,1], axis_indices):
    ax.plot(T, mean[:,i], alpha=0.5, color="tab:red")
    ax.fill_between(T.flatten(), (mean[:,i] - model_stdx[:,i]).flatten(), (mean[:,i] + model_stdx[:,i]).flatten(), alpha=0.2, color="tab:grey")
# plot mean and data var
axs[0,2].set_title("Mean and Data Uncertainty", loc="left")
for ax, i in zip(axs[:,2], axis_indices):
    ax.plot(T, mean[:,i], alpha=0.5, color="tab:red")
    ax.fill_between(T.flatten(), (mean[:,i] - data_stdx[:,i]).flatten(), (mean[:,i] + data_stdx[:,i]).flatten(), alpha=0.2, color="tab:grey")
# plot total vars
for ax in axs[-1]:
    ax.grid()
    linestyle = None
    ax.plot(
        np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, T.flatten()),
        np.where(np.logical_or(common_noisy_mask, missing_mask), np.nan, 0),
        color="tab:blue", linestyle=linestyle)
    ax.plot(
        np.where(missing_mask, T.flatten(), np.nan),
        np.where(missing_mask, 0, np.nan),
        color="tab:orange", linestyle=linestyle)
    ax.plot(
        np.where(common_noisy_mask, T.flatten(), np.nan),
        np.where(common_noisy_mask, 0, np.nan),
        color="tab:green", linestyle=linestyle)
axs[-1,0].set_ylabel(r"{}$\sigma$".format(num_std_plot))
axs[-1,0].plot(T, np.sum(stdx, axis=1), alpha=0.5, color="tab:grey")
axs[-1,1].plot(T, np.sum(model_stdx, axis=1), alpha=0.5, color="tab:grey")
axs[-1,2].plot(T, np.sum(data_stdx, axis=1), alpha=0.5, color="tab:grey")
fig.savefig("mlp_model_data_2d_T{}.png".format(image_name_suffix))
