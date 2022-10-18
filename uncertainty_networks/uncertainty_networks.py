
from uncertainty_networks.pfrnn import PFGRU

import copy
import numpy as np
import torch
from torch import Tensor
from typing import Literal, Tuple, Sequence


class MonteCarloDropout(torch.nn.modules.dropout._DropoutNd):
    def __init__(self, p: float):
        super().__init__(p, False)

    def forward(self, input: Tensor) -> Tensor:
        # always have dropout enabled
        return torch.nn.functional.dropout(input, self.p, True, False)


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
            hidden_sizes: Sequence[int],
            output_size: int,
            dropout_prob: float,
            num_passes: int,
            num_models: int,
            initialization: Literal["rl", "sl"],
            activation: torch.nn.Module,
            device: str):

        super().__init__()

        self._output_size = output_size
        self._num_passes = num_passes
        self._num_models = num_models
        self._initialization = initialization
        self._device = device

        assert initialization in ["rl", "sl"]

        # create model
        model = torch.nn.ModuleList()
        for hidden_size in hidden_sizes:
            model.append(torch.nn.Linear(input_size, hidden_size))
            model.append(activation())
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

    @torch.jit.ignore
    @torch.jit.export
    def reset_parameters(self) -> None:
        for layer in self.modules():
            if not isinstance(layer, torch.nn.Linear):
                continue
            if self._initialization == "rl":
                # use orthogonal initialization for weights
                torch.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                # set biases to zero
                torch.nn.init.zeros_(layer.bias)
                # as described in: 
                # https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
            else:
                # use default layer initialization
                layer.reset_parameters()

    @torch.jit.export
    def forward(self, input: Tensor, shared_input: bool=True) -> Tensor:
        # input shape:
        #   (batch, input_size)                           when shared_input is True
        #   (num_models, num_passes, batch, input_size)   when shared_input is False
        # batch dimension can be composed of several dimensions, e.g. (sequence, batch) for RNNs.

        # If shared_input is False, then run inference on individual predictions
        # Can be used to propagate individual predictions through many modules
        if not shared_input:
            assert input.shape[:2] == (self._num_models, self._num_passes)

        # include batch dimensions in shape of predictions array
        if shared_input:
            shape = (self._num_models, self._num_passes) + input.shape[:-1] + (self._output_size,)
        else:
            shape = input.shape[:-1] + (self._output_size,)
        preds = torch.zeros(shape, device=self._device)

        # iterate over Ensemble models
        for i, model in enumerate(self._models):
            # iterate over passes of single MC Dropout model
            for j in range(self._num_passes):
                if shared_input:
                    layer_input = input
                else:
                    layer_input = input[i, j]

                for layer in model:
                    layer_input = layer(layer_input)

                preds[i, j] = layer_input

        return preds


class UncertaintyGRU(torch.nn.Module):
    """
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout_prob: float,
            num_passes: int,
            num_models: int,
            initialization: Literal["rl", "sl"],
            device: str):

        super().__init__()

        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._num_passes = num_passes
        self._num_models = num_models
        self._initialization = initialization
        self._device = device

        assert initialization in ["rl", "sl"]

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

    @torch.jit.ignore
    @torch.jit.export
    def reset_parameters(self) -> None:
        for module in self.modules():
            if not isinstance(module, torch.nn.GRU):
                continue
            if self._initialization == "rl":
                for name, param in module.named_parameters():
                    if "weight" in name:
                        torch.nn.init.orthogonal_(param, 1.0)
                    elif "bias" in name:
                        torch.nn.init.zeros_(param)
            else:
                # use default layer initialization
                module.reset_parameters()

    @torch.jit.export
    def forward(self, input: Tensor, hidden: Tensor=None, shared_input: bool=True) -> Tuple[Tensor, Tensor]:
        # input shape:
        # (seq_len, batch, input_size)                          when shared_input is True
        # (num_models, num_passes, seq_len, batch, input_size)  when shared_input is False
        # (for hidden see init_hidden)

        # If shared_input is False, then run inference on individual predictions
        # Can be used to propagate individual predictions through many modules
        if shared_input:
            assert input.ndim == 3
        else:
            assert input.ndim == 5
            assert input.shape[:2] == (self._num_models, self._num_passes)

        if hidden is None:
            hidden = self.init_hidden(input.shape[-2])

        # include sequence length and batch dimensions in shape of predictions array
        if shared_input:
            shape = (self._num_models, self._num_passes) + input.shape[:-1] + (self._hidden_size,)
        else:
            shape = input.shape[:-1] + (self._hidden_size,)
        preds = torch.zeros(shape, device=self._device)
        hidden_out = torch.zeros_like(hidden)

        # iterate over Ensemble models
        for i, model in enumerate(self._models):
            # iterate over passes of single MC Dropout model
            for j in range(self._num_passes):
                if shared_input:
                    model_input = input
                else:
                    model_input = input[i, j]
                
                # always have dropout enabled
                model.training = True
                preds[i, j], hidden_out[i, j] = model(model_input, hidden[i, j])

        return preds, hidden_out

    @torch.jit.export
    def init_hidden(self, batch_size: int) -> Tensor:
        if False: #batch_size == None:
            # not supported currently
            # omit batch dimension
            shape = (self._num_models, self._num_passes, self._num_layers, self._hidden_size)
        else:
            shape = (self._num_models, self._num_passes, self._num_layers, batch_size, self._hidden_size)

        # use random initial values for more diversity in uncertainty
        # some sources claim zeros is better
        # (https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
        init_func = torch.zeros
        # init_func = torch.rand
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
        output_var, output_mean = torch.var_mean(preds, dim=0, unbiased=False)

        return output_mean, output_var, preds, hidden

    def reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                # reset weights *and* biases
                layer.reset_parameters()
            elif isinstance(layer, torch.nn.BatchNorm1d):
                # reset weights *and* biases
                layer.reset_parameters()

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
            dropout_prob: float,
            num_passes: int,
            num_models: int,
            initialization: Literal["rl", "sl"],
            device: str):

        super().__init__()

        activation = torch.nn.LeakyReLU

        self.mlp1 = UncertaintyMLP(
            input_size=input_size[0],
            hidden_sizes=hidden_size[0],
            output_size=output_size[0],
            dropout_prob=dropout_prob,
            num_passes=num_passes,
            num_models=num_models,
            initialization=initialization,
            activation=activation,
            device=device)

        self.rnn = UncertaintyGRU(
            input_size=input_size[1],
            hidden_size=hidden_size[1],
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            num_passes=num_passes,
            num_models=num_models,
            initialization=initialization,
            device=device)

        self.mlp2 = UncertaintyMLP(
            input_size=input_size[2],
            hidden_sizes=hidden_size[2],
            output_size=output_size[1],
            dropout_prob=dropout_prob,
            num_passes=num_passes,
            num_models=num_models,
            initialization=initialization,
            activation=activation,
            device=device)

    @torch.jit.export
    def reset_parameters(self) -> None:
        self.mlp1.reset_parameters()
        self.rnn.reset_parameters()
        self.mlp2.reset_parameters()

    @torch.jit.export
    def forward(self, input: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

        # include sequence dimension if not present
        added_seq_dim = False
        if input.ndim == 2:
            input = input[None, ...]
            added_seq_dim = True
        assert input.ndim == 3
        assert input.shape[-2] == hidden.shape[-2]

        preds = self.mlp1(input=input)
        preds, hidden = self.rnn(preds, hidden=hidden, shared_input=False)
        preds = self.mlp2(preds, shared_input=False)

        var, mean = torch.var_mean(preds, dim=(0, 1), unbiased=False)

        # remove sequence length dimension only if it was not present
        if added_seq_dim:
            mean = torch.squeeze(mean, dim=0)
            var = torch.squeeze(var, dim=0)
            preds = torch.squeeze(preds, dim=-3)

        return mean, var, preds, hidden

    @torch.jit.export
    def init_hidden(self, batch_size: int) -> Tensor:
        return self.rnn.init_hidden(batch_size)
