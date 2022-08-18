
import torch
from torch import nn
import numpy as np


class PFRNNBaseCell(nn.Module):
    """
    This is the base class for the PF-RNNs. We implement the shared functions here, including
        1. soft-resampling
        2. reparameterization trick
        3. obs_extractor o_t(x_t)
        4. control_extractor u_t(x_t)

        All particles in PF-RNNs are processed in parallel to benefit from GPU parallelization.
    """

    def __init__(self, input_size, hidden_size, num_particles, resamp_alpha):
        """
        :param input_size: the size of input x_t
        :param hidden_size: the size of the hidden particle h_t^i
        :param num_particles: number of particles for a PF-RNN
        :param resamp_alpha: the control parameter \alpha for soft-resampling.
        We use the importance sampling with a proposal distribution q(i) = \alpha w_t^i + (1 - \alpha) (1 / K)
        """
        super(PFRNNBaseCell, self).__init__()
        self.num_particles = num_particles
        self.input_size = input_size
        self.h_dim = hidden_size
        self.resamp_alpha = resamp_alpha

    def resampling(self, particles, prob):
        """
        The implementation of soft-resampling. We implement soft-resampling in a batch-manner.

        :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                        each tensor has a shape: [num_particles * batch_size, h_dim]
        :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
        :return: resampled particles and weights according to soft-resampling scheme.
        """
        resamp_prob = self.resamp_alpha * torch.exp(prob) + (1 -
                                                             self.resamp_alpha) * 1 / self.num_particles
        resamp_prob = resamp_prob.view(self.num_particles, -1)
        indices = torch.multinomial(resamp_prob.transpose(0, 1),
                                    num_samples=self.num_particles, replacement=True)
        batch_size = indices.size(0)
        indices = indices.transpose(1, 0).contiguous()
        offset = torch.arange(batch_size).type(torch.LongTensor).unsqueeze(0)
        if torch.cuda.is_available():
            offset = offset.cuda()
        indices = offset + indices * batch_size
        flatten_indices = indices.view(-1, 1).squeeze()

        particles_new = particles[flatten_indices]

        prob_new = torch.exp(prob.view(-1, 1)[flatten_indices])
        prob_new = prob_new / (self.resamp_alpha * prob_new + (1 -
                                                               self.resamp_alpha) / self.num_particles)
        prob_new = torch.log(prob_new).view(self.num_particles, -1, 1)
        prob_new = prob_new - torch.logsumexp(prob_new, dim=0, keepdim=True)
        prob_new = prob_new.view(-1, 1)

        return particles_new, prob_new

    def reparameterize(self, mu, var):
        """
        Reparameterization trick

        :param mu: mean
        :param var: variance
        :return: new samples from the Gaussian distribution
        """
        std = torch.nn.functional.softplus(var)
        # TODO this is weird
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.shape).normal_()
        else:
            eps = torch.FloatTensor(std.shape).normal_()

        return mu + eps * std


class PFGRUCell(PFRNNBaseCell):
    """
    Based on:
    Particle Filter Recurrent Neural Networks (2020)
    https://github.com/Yusufma03/pfrnns
    """
    # TODO document differences with original implementation
    def __init__(
            self,
            input_size,
            hidden_size,
            num_particles,
            resamp_alpha):

        super().__init__(
            input_size,
            hidden_size,
            num_particles,
            resamp_alpha)

        self.input_size = input_size
        self.h_dim = hidden_size
        self.fc_z = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_r = nn.Linear(self.h_dim + self.input_size, self.h_dim)
        self.fc_n = nn.Linear(self.h_dim + self.input_size, self.h_dim * 2)
        self.fc_obs = nn.Linear(self.input_size + self.h_dim, 1)
        self.batch_norm = nn.BatchNorm1d(self.num_particles)

    # TODO
    # def reset_parameters(self):
    #     std = 1.0 / np.sqrt(self.hidden_size)
    #     for w in self.parameters():
    #         w.data.uniform_(-std, std)

    def forward(self, input, hx):
        # hx is composed of hidden state (h0) and log(weights) (p0)
        h0, p0 = hx

        # update gate
        z = torch.sigmoid(self.fc_z(torch.cat((h0, input), dim=1)))
        # reset gate
        r = torch.sigmoid(self.fc_r(torch.cat((h0, input), dim=1)))
        # memory
        n = self.fc_n(torch.cat((r * h0, input), dim=1))

        mu_n, var_n = torch.split(n, split_size_or_sections=self.h_dim, dim=1)
        n = self.reparameterize(mu_n, var_n)

        # batch norm and relu replace hyperbolic tangen function
        n = n.view(self.num_particles, -1, self.h_dim).transpose(0, 1).contiguous()
        n = self.batch_norm(n)
        n = n.transpose(0, 1).contiguous().view(-1, self.h_dim)
        n = nn.functional.leaky_relu(n)

        # update of hidden state
        h1 = (1 - z) * n + z * h0

        # particle weight update
        att = torch.cat((h1, input), dim=1)
        p1 = self.fc_obs(att) + p0

        # normalize log of weights
        p1 = p1.view(self.num_particles, -1, 1)
        p1 = p1 - torch.logsumexp(p1, dim=0, keepdim=True)

        # soft-resampling
        h1, p1 = self.resampling(h1, p1)

        return h1, p1


class PFGRU(torch.nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_particles: int,
            num_layers: int,
            dropout_prob: float,
            resamp_alpha: float,
            device: str):
        super().__init__()

        # TODO add support for multiple layers
        self._num_layers = num_layers
        self._hidden_size = hidden_size
        self._num_particles = num_particles
        self._device = device

        self.rnn_cell = PFGRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_particles=num_particles,
            resamp_alpha=resamp_alpha)

        self.to(device)

    def forward(self, input, hidden):

        # input has shape (seq_len, batch, input_dim) (standard PyTorch)
        # hidden is tuple with shapes:
        #   (batch_size * num_particles, hidden_size)
        #   (batch_size * num_particles, 1)

        # repeat the batch dimension when using PF-RNN
        input = input.repeat(1, self._num_particles, 1)
        seq_len = input.shape[0]
        # tuples of hidden state and log(weights)
        # TODO make tensor?
        hidden_states = []#torch.zeros((seq_len, ), device=self._device)
        probs = []

        for step in range(seq_len):
            hidden = self.rnn_cell(input[step], hidden)
            hidden_states.append(hidden[0])
            probs.append(hidden[1])

        hidden_states = torch.stack(hidden_states, dim=0)
        # TODO dropout here?
        # hidden_states = self.hnn_dropout(hidden_states)

        # calculate mean hidden-state
        probs = torch.stack(probs, dim=0)
        prob_reshape = probs.view([seq_len, self._num_particles, -1, 1])
        hidden_reshape = hidden_states.view([seq_len, self._num_particles, -1, self._hidden_size])
        y = hidden_reshape * torch.exp(prob_reshape)
        # sum over particles dimension
        # y = torch.sum(y, dim=1)
        
        # y has shape (seq_len, num_particles, batch, hidden_size)
        # we swap the first axes to have shape (num_particles, seq_length, ...)
        y = torch.swapaxes(y, 0, 1)

        # TODO output hidden state
        return y, None
