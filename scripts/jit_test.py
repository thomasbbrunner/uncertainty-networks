
from uncertainty_networks import UncertaintyNetwork
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

# reproducibility
torch.manual_seed(0)
np.random.seed(0)


def train(X_train, y_train, model, epochs, device, shuffle, loss_type):
    loss_mse = torch.nn.functional.mse_loss
    loss_l1 = torch.nn.functional.l1_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    X_train = torch.Tensor(X_train).to(device)
    y_train = torch.Tensor(y_train).to(device)
    batch_size = len(X_train)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=shuffle,
        batch_size=batch_size)
    model.train()
    model.to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()

            x, y = batch
            # format to shape (seq, batch, ...)
            x = np.swapaxes(x, 0, 1)
            y = np.swapaxes(y, 0, 1)
            hidden = model.init_hidden(batch_size)
            means, vars, preds, _ = model(x, hidden)

            if loss_type == "pred" or loss_type == "elbo":
                # flatten dimensions of forward passes or particles
                # we want our tensor to have shape:
                # (num_model*num_passes, seq_len, batch, output_size)
                preds = preds.flatten(0, preds.dim() - 4)
                # repeat desired output to match shape of predictions 
                y_repeat = y.unsqueeze(0).repeat_interleave(preds.shape[0], 0)
                assert y_repeat.shape == preds.shape

            # loss on mean
            if loss_type == "mean":
                loss = loss_mse(means, y)
            # loss on mean with variance
            elif loss_type == "var":
                loss = loss_mse(means, y) + 0.01*vars.mean()
            # loss on each individual prediction
            elif loss_type == "pred":
                loss = loss_mse(preds, y_repeat)
            # loss as described in the PFRNN paper
            elif loss_type == "elbo":
                # TODO add bpdecay_params
                # TODO paper uses sums instead of means to reduce the loss
                # maybe this could prevent vanishing gradients if it happens?
                l2_weight = 1.
                l1_weight = 0.
                elbo_weight = 1.

                loss_preds_l2 = loss_mse(preds, y_repeat, reduction='none')
                loss_preds_l2 = -torch.log(torch.mean(torch.exp(-loss_preds_l2), dim=0))

                loss_preds_l1 = loss_l1(preds, y_repeat, reduction='none')
                loss_preds_l1 = -torch.log(torch.mean(torch.exp(-loss_preds_l1), dim=0))

                loss_preds = (
                    l2_weight*torch.mean(loss_preds_l2) + 
                    l1_weight*torch.mean(loss_preds_l1))

                loss_mean = (
                    l2_weight*loss_mse(means, y) + 
                    l1_weight*loss_l1(means, y))

                loss = loss_mean + elbo_weight * loss_preds

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
        # add batch dimension
        X_test = X_test[..., None, :]
        y_test = y_test[..., None, :]
        hidden = model.init_hidden(1)
        model.eval()
        model.to(device)
        mean, var, preds, _ = model(X_test, hidden)
        loss = loss_fn(mean, y_test)
        return (
            mean.to("cpu").numpy(),
            var.to("cpu").numpy(),
            preds.to("cpu").numpy(),
            loss.to("cpu").numpy())


# Training
# parameters
epochs = 5000
seq_length = 100
# input sizes of input encoder, gru and latent encoder
input_size = [1, 3, 3]
# hidden sizes of input encoder, gru and latent encoder
hidden_size = [[4, 4], 3, [4, 4]]
# output sizes of input encoder, and latent encoder
output_size = [3, 1]
num_layers = 2
num_std_plot = 2
device = "cuda"
shuffle = True
loss_type = "pred"

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
T, DT = np.linspace(0, 15, 500, retstep=True)
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
train_indices = np.index_exp[:T.shape[0]//3*2]
test_indices = np.index_exp[T.shape[0]//3*2:]
X_train = np.concatenate([ang[i] for i in train_indices])
y_train = np.concatenate([vel[i] for i in train_indices])
# generate sequences
indexer = np.arange(seq_length)[None, ...] + np.arange(X_train.shape[0] + 1 - seq_length)[..., None]
# dimension (batch, seq_length, input_size)
X_train = X_train[indexer][..., None]
y_train = y_train[indexer][..., None]
# generate training dataset with sequence length of 1
# dimension (batch, 1, input_size)
# repeat dataset by sequence length to have a dataset of similar number of samples
X_train_seq = np.repeat(np.reshape(ang, (-1, 1, 1)), 10, axis=0)
y_train_seq = np.repeat(np.reshape(vel, (-1, 1, 1)), 10, axis=0)
# generate test dataset
# one big sequence (seq_length, input_size)
X_test = ang[..., None]
y_test = vel[..., None]


# Train Python model
torch.manual_seed(0)
np.random.seed(0)
start_time = time.time()
model = UncertaintyNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_prob=0.05,
    num_passes=5,
    num_models=2,
    initialization="sl",
    device=device)
init_time = time.time() - start_time

start_time = time.time()
train(X_train, y_train, model, epochs, device, shuffle, loss_type)
mean_1, var_1, preds_1, loss_1 = test(X_test, y_test, model, device)
train_time = time.time() - start_time


# Train Torchscript model
torch.manual_seed(0)
np.random.seed(0)
start_time = time.time()
model_jit = UncertaintyNetwork(
    input_size=input_size,
    hidden_size=hidden_size,
    output_size=output_size,
    num_layers=num_layers,
    dropout_prob=0.05,
    num_passes=5,
    num_models=2,
    initialization="sl",
    device=device)
model_jit = torch.jit.script(model_jit)
init_time_jit = time.time() - start_time

start_time = time.time()
train(X_train, y_train, model_jit, epochs, device, shuffle, loss_type)
mean_2, var_2, preds_2, loss_2 = test(X_test, y_test, model_jit, device)
train_time_jit = time.time() - start_time


# Summary
print("Python model:\n\tinitialization time: {}s\n\ttraining time: {} s\n\tfinal loss: {}".format(init_time, train_time, loss_1))
print("TorchScript model:\n\tinitialization time: {}s\n\ttraining time: {} s\n\tfinal loss: {}".format(init_time_jit, train_time_jit, loss_2))

# Plotting
fig = plt.figure(dpi=300, figsize=(28, 14), constrained_layout=True)
axs = np.array(fig.subplots(2, 2))
mean_plot = np.array([mean_1.flatten(), mean_2.flatten()])
std_plot = num_std_plot*np.sqrt(np.array([var_1.flatten(), var_2.flatten()]))
pred_plot = [preds_1, preds_2]
# plot function
for ax in axs.flatten():
    ax.grid()
    for i in train_indices:
        ax.plot(T[i], vel[i], color="tab:blue", linestyle="--")
    for i in test_indices:
        ax.plot(T[i], vel[i], color="tab:orange", linestyle="--")
# titles
axs[0, 0].set_title("Python Model\nTest Loss: {:.4}\nTraining Time: {:.4}".format(loss_1, train_time), loc="left")
axs[1, 0].set_title("TorchScript Model\nTest Loss: {:.4}\nTraining Time: {:.4}".format(loss_2, train_time_jit), loc="left")
# plot mean and std
axs[0, 0].set_title(r"Final Predictions ({}$\sigma$)".format(num_std_plot))
for i, ax in enumerate(axs[:, 0]):
    ax.plot(T, mean_plot[i], color="tab:red")
    ax.fill_between(T, mean_plot[i] - std_plot[i], mean_plot[i] + std_plot[i], alpha=0.2, color="tab:grey")
# plot predictions
axs[0, 1].set_title("Individual Predictions")
for i, ax in enumerate(axs[:, 1]):
    # flatten dimensions so that predictions have shape (total_passes, ...)
    preds = pred_plot[i].reshape(-1, *pred_plot[i].shape[2:])
    for pred in preds:
        ax.plot(T, pred.flatten(), color="tab:red", alpha=np.maximum(0.2, 1/len(preds)))
fig.savefig("jit_test.png")