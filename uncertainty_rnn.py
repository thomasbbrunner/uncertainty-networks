# %% Setup
from uncertainty_networks import UncertaintyRNN
import matplotlib.pyplot as plt
import numpy as np
import torch

# reproducibility
torch.manual_seed(0)
np.random.seed(0)


def train(X_train, y_train, model, epochs, device, shuffle, loss_type):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    batch_size = len(X_train)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=shuffle,
        batch_size=batch_size)
    model.train()
    model.to(device)
    hidden = model.init_hidden(batch_size)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            x, y = batch
            # format to shape (seq, batch, ...)
            x = np.swapaxes(x, 0, 1)
            y = np.swapaxes(y, 0, 1)
            hidden = model.init_hidden(batch_size)
            means, vars, preds, _ = model(x, hidden, return_predictions=True)

            # loss on mean
            if loss_type == "mean":
                loss = loss_fn(means, y)
            # loss on mean with variance
            elif loss_type == "var":
                loss = loss_fn(means, y) + 0.01*vars.mean()
            # loss on each prediction
            elif loss_type == "pred":
                # flatten dimensions with the different passes
                preds = preds.flatten(0, 1)
                # repeat desired output to match shape of predictions 
                # (num_model*num_passes, ...)
                y = y.unsqueeze(0).repeat_interleave(preds.shape[0], 0)
                assert y.shape == preds.shape
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
        hidden = model.init_hidden(None)
        model.eval()
        model.to(device)
        mean, var, preds, _ = model(X_test, hidden, return_predictions=True)
        loss = loss_fn(mean, y_test)
        return (
            mean.to("cpu").numpy(),
            var.to("cpu").numpy(),
            preds.to("cpu").numpy(),
            loss.to("cpu").numpy())


# %% Training
# parameters
epochs = 10000
seq_length = 100
input_size = 1
hidden_size = 3
output_size = 1
num_layers = 2
num_std_plot = 2
device = "cuda"
# TODO hyperparameters
shuffle = True
loss_type = "pred"
image_name_suffix = "_shuffle={} loss={}".format(shuffle, loss_type)

# dynamics of pendulum with dampening
G = 9.81
MASS = 1.0
POLE_LENGTH = 1.0
MUE = 0.01
VEL_0 = 0
ANG_0 = 1
# dynamics equation:
# m*l**2*acc = -mu*vel + m*g*l*sin(angle)
C1 = -MUE/(MASS*POLE_LENGTH**2)
C2 = G/(POLE_LENGTH)
T, DT = np.linspace(0, 10, 100, retstep=True)
vel = np.ones_like(T)*VEL_0
ang = np.ones_like(T)*ANG_0
def step(velocity, angle):
    acceleration = (C1*velocity + C2*np.sin(angle))
    velocity = velocity + DT*acceleration
    angle = angle + DT*velocity + 0.5*DT**2*acceleration
    return velocity, angle
for i in range(1, T.shape[0]):
    vel[i], ang[i] = step(vel[i-1], ang[i-1])

# generate training dataset
# TODO split test and train
indexer = np.arange(seq_length)[None, ...] + np.arange(T.shape[0] + 1 - seq_length)[..., None]
# dimension (batch, seq_length, input_size)
X_train = ang[indexer][..., None]  # split data into sequences
y_train = vel[indexer[..., None]]  # select last velocity in sequence
# generate training dataset with sequence length of 1
# dimension (batch, 1, input_size)
# repeat dataset by sequence length to have a dataset of similar number of samples
X_train_seq = np.repeat(np.reshape(ang, (-1, 1, 1)), 10, axis=0)
y_train_seq = np.repeat(np.reshape(vel, (-1, 1, 1)), 10, axis=0)
# generate test dataset
# one big sequence (seq_length, input_size)
X_test = ang[..., None]
y_test = vel[..., None]

# train deterministic baseline with sequence length of 1
rnn_seq = UncertaintyRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    num_passes=1,
    num_models=1,
    dropout_prob=0,
    device=device)
train(X_train_seq, y_train_seq, rnn_seq, epochs, device, shuffle, "pred")
mean_seq, var_seq, preds_seq, loss_seq = test(X_test, y_test, rnn_seq, device)

# train deterministic baseline
rnn_0 = UncertaintyRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    num_passes=1,
    num_models=1,
    dropout_prob=0,
    device=device)
train(X_train, y_train, rnn_0, epochs, device, shuffle, "pred")
mean_0, var_0, preds_0, loss_0 = test(X_test, y_test, rnn_0, device)

# train dropout
rnn_1 = UncertaintyRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    num_passes=10,
    num_models=1,
    dropout_prob=0.05,
    device=device)
train(X_train, y_train, rnn_1, epochs, device, shuffle, loss_type)
mean_1, var_1, preds_1, loss_1 = test(X_test, y_test, rnn_1, device)

# train ensemble
rnn_2 = UncertaintyRNN(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    num_passes=1,
    num_models=10,
    dropout_prob=0,
    device=device)
train(X_train, y_train, rnn_2, epochs, device, shuffle, loss_type)
mean_2, var_2, preds_2, loss_2 = test(X_test, y_test, rnn_2, device)

# %% Plotting
# dataset
fig = plt.figure(dpi=300, constrained_layout=True)
axs = fig.subplots(2)
axs[0].set_title("Ground Truth Inputs", loc="left")
axs[0].grid()
axs[0].set_ylabel("Inputs")
axs[0].plot(T, ang, color="tab:blue")
axs[1].set_title("Ground Truth Outputs", loc="left")
axs[1].grid()
axs[1].set_ylabel("Outputs")
axs[1].set_xlabel("Time (Sequence Length)")
axs[1].plot(T, vel, color="tab:blue")
fig.savefig("uncertainty_rnn_dataset.png")

# baselines
fig = plt.figure(dpi=300, constrained_layout=True)
axs = fig.subplots(2)
axs[0].set_title("Deterministic Baseline\nTest Loss = {:.4}".format(loss_0), loc="left")
axs[0].grid()
axs[0].plot(T, vel, color="tab:blue", linestyle="--")
axs[0].plot(T, mean_0, color="tab:red")
axs[1].set_title("Deterministic Baseline with seq_len = 1\nTest Loss = {:.4}".format(loss_seq), loc="left")
axs[1].grid()
axs[1].plot(T, vel, color="tab:blue", linestyle="--")
axs[1].plot(T, mean_seq, color="tab:red")
fig.savefig("uncertainty_rnn_baseline.png")

# models
fig = plt.figure(dpi=300, figsize=(28, 14), constrained_layout=True)
axs = np.array(fig.subplots(2, 4))
mean_plot = np.array([mean_1.flatten(), mean_2.flatten()])
std_plot = num_std_plot*np.sqrt(np.array([var_1.flatten(), var_2.flatten()]))
pred_plot = [preds_1, preds_2]
# plot function
for ax in axs.flatten():
    ax.grid()
    ax.plot(T, vel, color="tab:blue", linestyle="--")
# titles
axs[0, 0].set_title("MC Dropout (10 passes)\nTest Loss = {:.4}".format(loss_1), loc="left")
axs[1, 0].set_title("Ensemble (10 models)\nTest Loss = {:.4}".format(loss_2), loc="left")
# axs[2, 0].set_title(r"Ensemble (5 models) MC Dropout (2 passes)" "\nTest Loss = {:.4}".format(loss_3), loc="left")
# axs[3, 0].set_title(r"Ensemble (2 models) MC Dropout (5 passes)" "\nTest Loss = {:.4}".format(loss_4), loc="left")
# plot mean and std
axs[0, 0].set_title(r"Final Predictions ({}$\sigma$)".format(num_std_plot))
for i, ax in enumerate(axs[:, 0]):
    ax.plot(T, mean_plot[i], color="tab:red")
    ax.fill_between(T, mean_plot[i] - std_plot[i], mean_plot[i] + std_plot[i], alpha=0.2, color="tab:grey")
# plot predictions
axs[0, 1].set_title("Individual Predictions")
for i, ax in enumerate(axs[:, 1]):
    preds = pred_plot[i].reshape(-1, *pred_plot[i].shape[2:])
    for pred in preds:
        ax.plot(T, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
# plot predictions of Ensemlbes (mean over MC Dropout MLPs)
axs[0, 2].set_title("Individual Predictions of Ensembles")
for i, ax in enumerate(axs[:, 2]):
    preds = pred_plot[i].mean(axis=1)
    for pred in preds:
        ax.plot(T, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
# plot predictions of MC Dropout MLPs (mean over Ensembles)
axs[0, 3].set_title("Individual Predictions of MC Dropouts")
for i, ax in enumerate(axs[:, 3]):
    preds = pred_plot[i].mean(axis=0)
    for pred in preds:
        ax.plot(T, pred, color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
fig.savefig("uncertainty_rnn{}.png".format(image_name_suffix))
