# %% Setup
from uncertainty_networks import UncertaintyMLP
import matplotlib.pyplot as plt
import numpy as np
import torch

# reproducibility
torch.manual_seed(0)
np.random.seed(0)


def train(X_train, y_train, model, epochs, device, shuffle, loss_type):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=shuffle,
        batch_size=len(X_train))
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            x, y = batch
            means, vars, preds = model(x)

            # loss on mean
            if loss_type == "mean":
                loss = loss_fn(means, y)
            # loss on mean with variance
            elif loss_type == "var":
                loss = loss_fn(means, y) + 0.01*vars.mean()
            # loss on each prediction
            elif loss_type == "pred":
                preds = preds.flatten(0, 1)
                # repeat desired output to match shape of predictions 
                # (num_model*num_passes, ...)
                y = y.unsqueeze(0).repeat_interleave(preds.shape[0], 0)
                loss = loss_fn(preds, y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 1000 == 0 or epoch == epochs - 1:
            print("Epoch: {}, Loss: {:.4f}".format(epoch, epoch_loss))


def test(X_test, y_test, model, device):
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        X_test = torch.Tensor(X_test).to(device)
        y_test = torch.Tensor(y_test).to(device)
        model.eval()
        model.to(device)
        mean, var, preds = model(X_test)
        loss = loss_fn(mean, y_test)
        return (
            mean.to("cpu").numpy(),
            var.to("cpu").numpy(),
            preds.to("cpu").numpy(),
            loss.to("cpu").numpy())


# %% Training
# parameters
main_shape = (128, 128, 128)
input_size = 1
output_size = 1
device = "cuda"
training_epochs = 20000
num_std_plot = 2
# TODO hyperparameters
shuffle = True
loss_type = "pred"
image_name_suffix = "_shuffle={} loss={}".format(shuffle, loss_type)

# generate dataset
X = np.linspace(-10, 10, 1000).reshape(-1, 1)
train_indices = np.index_exp[:200, 300:700, 800:]
test_indices = np.index_exp[200:300, 700:800]
X_train = np.concatenate([X[i] for i in train_indices])
def y_func(x):
    return x * np.sin(x) + x * np.cos(2*x)
y = y_func(X)
y_train = y_func(X_train)

# train Deterministic Baseline
mlp_0 = UncertaintyMLP(
    input_size=input_size,
    hidden_sizes=main_shape,
    output_size=output_size,
    dropout_prob=0,
    # multiple passes to make comparison with other networks fairer
    num_passes=10,
    num_models=1,
    device=device)
train(X_train, y_train, mlp_0, training_epochs, device, shuffle, "pred")
y_0, var_0, pred_0, loss_0 = test(X, y, mlp_0, device)

# train MC Dropout
mlp_1 = UncertaintyMLP(
    input_size=input_size,
    hidden_sizes=main_shape,
    output_size=output_size,
    dropout_prob=0.1,
    num_passes=10,
    num_models=1,
    device=device)
train(X_train, y_train, mlp_1, training_epochs, device, shuffle, loss_type)
y_1, var_1, pred_1, loss_1 = test(X, y, mlp_1, device)

# train Deep Ensamble
mlp_2 = UncertaintyMLP(
    input_size=input_size,
    hidden_sizes=main_shape,
    output_size=output_size,
    dropout_prob=0,
    num_passes=1,
    num_models=10,
    device=device)
train(X_train, y_train, mlp_2, training_epochs, device, shuffle, loss_type)
y_2, var_2, pred_2, loss_2 = test(X, y, mlp_2, device)

# train Deep Ensamble with MC Dropout
mlp_3 = UncertaintyMLP(
    input_size=input_size,
    hidden_sizes=main_shape,
    output_size=output_size,
    dropout_prob=0.1,
    num_passes=2,
    num_models=5,
    device=device)
train(X_train, y_train, mlp_3, training_epochs, device, shuffle, loss_type)
y_3, var_3, pred_3, loss_3 = test(X, y, mlp_3, device)

# train Deep Ensamble with MC Dropout
mlp_4 = UncertaintyMLP(
    input_size=input_size,
    hidden_sizes=main_shape,
    output_size=output_size,
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
# titles
axs[0, 0].set_title(r"MC Dropout (10 passes)" "\nTest Loss = {:.4}".format(loss_1), loc="left")
axs[1, 0].set_title(r"Ensemble (10 models)" "\nTest Loss = {:.4}".format(loss_2), loc="left")
axs[2, 0].set_title(r"Ensemble (5 models) MC Dropout (2 passes)" "\nTest Loss = {:.4}".format(loss_3), loc="left")
axs[3, 0].set_title(r"Ensemble (2 models) MC Dropout (5 passes)" "\nTest Loss = {:.4}".format(loss_4), loc="left")
# plot mean and std
axs[0, 0].set_title(r"Final Predictions ({}$\sigma$)".format(num_std_plot))
for i, ax in enumerate(axs[:, 0]):
    ax.plot(X, y_plot[i], color="tab:red")
    ax.fill_between(X.flatten(), y_plot[i] - std_plot[i], y_plot[i] + std_plot[i], alpha=0.2, color="tab:grey")
# plot predictions
axs[0, 1].set_title(r"Individual Predictions")
for i, ax in enumerate(axs[:, 1]):
    preds = pred_plot[i].reshape(-1, *pred_plot[i].shape[2:])
    for pred in preds:
        ax.plot(X, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
# plot predictions of Ensemlbes (mean over MC Dropout MLPs)
axs[0, 2].set_title(r"Individual Predictions of Ensembles")
for i, ax in enumerate(axs[:, 2]):
    preds = pred_plot[i].mean(axis=1)
    for pred in preds:
        ax.plot(X, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
# plot predictions of MC Dropout MLPs (mean over Ensembles)
axs[0, 3].set_title(r"Individual Predictions of MC Dropouts")
for i, ax in enumerate(axs[:, 3]):
    preds = pred_plot[i].mean(axis=0)
    for pred in preds:
        ax.plot(X, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
fig.savefig("uncertainty_mlp{}.png".format(image_name_suffix))
