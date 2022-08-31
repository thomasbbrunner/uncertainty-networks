
from pfrnn import PFGRU

import copy
import functools
import numpy as np
import torch
from typing import List, Tuple, Sequence


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
    # - account for variance in the loss or train with individual predictions
    # - (ideally) different order for data
    # - weight regularization did not help
    # - mc dropout paper used model precision derived from data, which was omitted here
    # - deep ensemble paper outputted parameters of gaussian to form GMM, which was omitted here
    # - (ideally) variance should be calibrated

    def __init__(
            self,
            input_size: int,
            hidden_sizes: List[int],
            output_size: int,
            dropout_prob: float,
            num_passes: int,
            num_models: int,
            device: str):

        super().__init__()

        self._output_size = output_size
        self._init_func = None # TODO
        self._num_passes = num_passes
        self._num_models = num_models
        self._device = device
        activation_func = torch.nn.LeakyReLU

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
        self.reset_parameters()

        self.to(device)

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                # reset weights *and* biases
                layer.reset_parameters()
                # overwrite weights if desired
                if self._init_func is not None:
                    self._init_func(layer.weight)

    def forward(self, input, input_preds=None):

        # use input_preds instead of hidden
        # one of them has to be None (xor)
        assert (input_preds == None) ^ (input == None)
        use_preds = input_preds != None

        # include batch dimensions in predictions array
        if use_preds:
            shape = (*input_preds.shape[:-1], self._output_size)
        else:
            shape = (self._num_models, self._num_passes, *input.shape[:-1], self._output_size)
        preds = torch.zeros(shape, device=self._device)

        # iterate over Ensemble models
        for i in range(self._num_models):
            # iterate over passes of single MC Dropout model
            for j in range(self._num_passes):
                if use_preds:
                    output = input_preds[i, j]
                else:
                    output = input
    
                for layer in self._models[i]:
                    output = layer(output)
                preds[i, j] = output

        # calculate mean and variance of models and passes
        output_mean = torch.mean(preds, dim=(0, 1))
        output_var = torch.var(preds, dim=(0, 1))

        return output_mean, output_var, preds


class UncertaintyGRU(torch.nn.Module):
    """
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_passes: int,
            num_models: int,
            dropout_prob: float,
            device: str):

        super().__init__()

        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_passes = num_passes
        self._num_models = num_models
        self._device = device
        self._init_func = None # TODO

        # create model
        gru = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob)

        # create ensemble
        self._models = torch.nn.ModuleList()
        for _ in range(self._num_models):
            self._models.append(copy.deepcopy(gru))

        # re-initialize parameters to ensure diversity in each model of the ensemble
        self.reset_parameters()

        self.to(device)

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.GRU):
                # reset weights *and* biases
                layer.reset_parameters()

    def forward(self, input, hidden=None, input_preds=None):

        # use input_preds instead of hidden
        # one of them has to be None (xor)
        assert (input_preds == None) ^ (input == None)
        use_preds = input_preds != None

        # always have dropout enabled
        self.train()

        # include sequence length and batch dimensions in predictions array
        if use_preds:
            shape = (*input_preds.shape[:-1], self._hidden_size)
        else:
            shape = (self._num_models, self._num_passes, *input.shape[:-1], self._hidden_size)
        preds = torch.zeros(shape, device=self._device)
        hidden_out = torch.zeros_like(hidden)

        # iterate over Ensemble models
        for i in range(self._num_models):
            # iterate over passes of single MC Dropout model
            for j in range(self._num_passes):
                if use_preds:
                    input = input_preds[i, j]
                preds[i, j], hidden_out[i, j] = self._models[i](input, hidden[i, j])

        # calculate mean and variance of models and passes
        output_mean = torch.mean(preds, dim=(0, 1))
        output_var = torch.var(preds, dim=(0, 1))

        return output_mean, output_var, preds, hidden_out

    def init_hidden(self, batch_size=None):
        if batch_size == None:
            # omit batch dimension
            shape = (self._num_models, self._num_passes, self._num_layers, self._hidden_size)
        else:
            shape = (self._num_models, self._num_passes, self._num_layers, batch_size, self._hidden_size)

        # TODO use random initial values for more diversity
        # init_func = torch.zeros
        init_func = torch.rand
        hidden = init_func(shape, device=self._device)
        return hidden


class UncertaintyPFGRU(torch.nn.Module):
    """
    TODO:
        - enable inference on individual predictions (like the other networks)
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int,
            num_particles: int,
            dropout_prob: float,
            resamp_alpha: float,
            device: str):

        super().__init__()

        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_particles = num_particles
        self._device = device

        self._pfgru = PFGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_particles=num_particles,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            resamp_alpha=resamp_alpha,
            device=device)

        self.reset_parameters()

        self.to(device)

    def forward(self, input: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]):

        preds, hidden = self._pfgru(input, hidden)
        # preds have shape (num_particles, seq_len, batch, output_size)

        # calculate mean and variance of particles
        # Paper's code used sum over particles instead of mean.
        # However, this was before activation and linear layers.
        # Also, when we train on individual predictions the sum of outputs is scaled wrongly
        # TODO maybe with elbo loss we can use the sum?
        # output_sum = torch.sum(preds, dim=0)
        output_mean = torch.mean(preds, dim=0)
        output_var = torch.var(preds, dim=0)

        return output_mean, output_var, preds, hidden

    @torch.jit.ignore
    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                # reset weights *and* biases
                layer.reset_parameters()
            elif isinstance(layer, torch.nn.BatchNorm1d):
                # reset weights *and* biases
                layer.reset_parameters()

    @torch.jit.ignore
    def init_hidden(self, batch_size=None):
        # use random initial values for more diversity
        # func = torch.zeros
        func = torch.rand
        if batch_size is None:
            batch_size = 1
        h0 = func(
            (self._num_layers, batch_size * self._num_particles, self._hidden_size), 
            device=self._device)
        p0 = np.log(1 / self._num_particles) * torch.ones(
            (self._num_layers, batch_size * self._num_particles, 1),
            device=self._device)
        hidden = (h0, p0)

        return hidden


class UncertaintyNetwork(torch.nn.Module):
    # consists of MLP -> RNN -> MLP

    def __init__(
            self,
            input_size: Tuple[int, int, int],
            hidden_size: Tuple[Sequence[int], int, Sequence[int]],
            output_size: Tuple[int, int],
            num_layers: int,
            num_passes: int,
            num_models: int,
            dropout_prob: float,
            device: str):

        super().__init__()

        self.mlp1 = UncertaintyMLP(
            input_size=input_size[0],
            hidden_sizes=hidden_size[0],
            output_size=output_size[0],
            dropout_prob=dropout_prob,
            num_passes=num_passes,
            num_models=num_models,
            device=device)

        self.rnn = UncertaintyGRU(
            input_size=input_size[1],
            hidden_size=hidden_size[1],
            num_layers=num_layers,
            num_passes=num_passes,
            num_models=num_models,
            dropout_prob=dropout_prob,
            device=device)

        self.mlp2 = UncertaintyMLP(
            input_size=input_size[2],
            hidden_sizes=hidden_size[2],
            output_size=output_size[1],
            dropout_prob=dropout_prob,
            num_passes=num_passes,
            num_models=num_models,
            device=device)

    def reset_parameters(self):
        self.mlp1.reset_parameters()
        self.rnn.reset_parameters()
        self.mlp2.reset_parameters()

    def forward(self, input, hidden=None):

        means = [None]*3
        vars = [None]*3
        preds = [None]*3

        means[0], vars[0], preds[0] = self.mlp1(input=input)
        means[1], vars[1], preds[1], hidden = self.rnn(input=None, hidden=hidden, input_preds=preds[0])
        means[2], vars[2], preds[2] = self.mlp2(input=None, input_preds=preds[1])

        return means[-1], vars[-1], preds[-1], hidden

    def init_hidden(self, batch_size=None):
        return self.rnn.init_hidden(batch_size)
