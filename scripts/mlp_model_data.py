
from uncertainty_networks import UncertaintyMLP
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import BatchSampler, RandomSampler, TensorDataset

# reproducibility
torch.manual_seed(0)
np.random.seed(0)


def train(X_train, y_train, model, epochs, batch_size, device, loss_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    dataset = TensorDataset(X_train, y_train)
    sampler = BatchSampler(RandomSampler(dataset), batch_size, False)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_indices in sampler:
            optimizer.zero_grad()
            X_batch, y_batch = dataset[batch_indices]

            output = model(X_batch)
            preds, logvar = torch.split(output, [1, 1], dim=-1)

            loss = torch.exp(-logvar)*(preds - y_batch).square() + logvar
            loss = loss.mean()

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if True: #epoch % 1000 == 0 or epoch == epochs - 1:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss))


def test(X_test, y_test, model, device):

    with torch.no_grad():
        X_test = torch.Tensor(X_test).to(device)
        y_test = torch.Tensor(y_test).to(device)
        model.eval()
        model.to(device)
        output = model(X_test)
        preds, data_logvar = torch.split(output, [1, 1], dim=-1)

        model_var, mean = torch.var_mean(preds, dim=0, unbiased=False)
        data_var = torch.exp(torch.mean(data_logvar, dim=0))
        var = model_var + data_var
        loss = (mean - y_test).square().mean()

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
output_size = 2
device = "cuda"
epochs = 500
batch_size = 1000
num_std_plot = 1
# hyperparameters
shuffle = True
loss_type = "pred"
use_jit = True
noise_std = 0.5
image_name_suffix = "_loss={}_jit={}".format(loss_type, use_jit)

# generate dataset
X = np.linspace(-20, 20, 10000).reshape(-1, 1)
missing_region = (np.array([[0.0, 0.1], [0.9, 1.0]])*X.shape[0]).astype(int)
noisy_region = (np.array([[0.6, 0.8]])*X.shape[0]).astype(int)
missing_mask = np.full(X.shape[0], False, dtype=bool)
for region in missing_region:
    missing_mask[region[0]:region[1]] = True
noisy_mask = np.full(X.shape[0], False, dtype=bool)
for region in noisy_region:
    noisy_mask[region[0]:region[1]] = True
def y_func(x):
    return np.sin(0.7*x+1) + 0.5*np.cos(1.2*x-2)
y = y_func(X)

X_train = X.copy()
y_train = y.copy()
y_train[noisy_mask] += noise_std*np.random.randn(*y[noisy_mask].shape)
X_train = X_train[~missing_mask]
y_train = y_train[~missing_mask]


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
train(X_train, y_train, model, epochs, batch_size, device, loss_type)
mean, var, model_var, data_var, preds, loss = test(X, y, model, device)
stdx = num_std_plot*np.sqrt(var)
model_stdx = num_std_plot*np.sqrt(model_var)
data_stdx = num_std_plot*np.sqrt(data_var)

# Plotting
fig = plt.figure(dpi=300, figsize=(7, 10), constrained_layout=True)
axs = fig.subplots(6, sharex=True, sharey=True)
# plot gt
for i, ax in enumerate(axs):
    ax.grid()
    if i == 1:
        continue
    linestyle = "--" if i != 0 else None
    ax.plot(np.where(np.logical_or(noisy_mask, missing_mask), np.nan, X.flatten()), np.where(np.logical_or(noisy_mask, missing_mask), np.nan, y.flatten()), color="tab:blue", linestyle=linestyle)
    ax.plot(np.where(missing_mask, X.flatten(), np.nan), np.where(missing_mask, y.flatten(), np.nan), color="tab:orange", linestyle=linestyle)
    ax.plot(np.where(noisy_mask, X.flatten(), np.nan), np.where(noisy_mask, y.flatten(), np.nan), color="tab:green", linestyle=linestyle)
# ground truth
axs[0].set_title("Ground Truth Data", loc="left")
# samples
size = 1
axs[1].set_title("Training Data", loc="left")
axs[1].scatter(X_train, y_train, color="tab:blue", marker=".", s=size)
# mean and total var
axs[2].plot(X, mean, color="tab:red")
axs[2].fill_between(X.flatten(), (mean - stdx).flatten(), (mean + stdx).flatten(), alpha=0.2, color="tab:grey")
axs[2].set_title("Mean and Total Uncertainty (test loss = {:.4})".format(loss), loc="left")
# mean and model var
axs[3].plot(X, mean, color="tab:red")
axs[3].fill_between(X.flatten(), (mean - model_stdx).flatten(), (mean + model_stdx).flatten(), alpha=0.2, color="tab:grey")
axs[3].set_title("Mean and Model Uncertainty", loc="left")
# mean and data var
axs[4].plot(X, mean, color="tab:red")
axs[4].fill_between(X.flatten(), (mean - data_stdx).flatten(), (mean + data_stdx).flatten(), alpha=0.2, color="tab:grey")
axs[4].set_title("Mean and Data Uncertainty", loc="left")
# preds
preds_plot = preds.reshape(-1, *preds.shape[1:])
for pred in preds_plot:
    axs[5].plot(X, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds_plot)))
axs[5].set_title("Predictions", loc="left")
fig.savefig("mlp_model_data.png")
