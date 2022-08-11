
import copy
import functools
import torch
from typing import List, Callable


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
            activation_func: Callable[..., torch.nn.Module],
            init_func: Callable,
            dropout_prob: float,
            num_passes: int,
            num_models: int,
            device: str):

        super().__init__()

        # TODO do parameter checks
        # assert model_precision > 0
        self._output_size = output_size
        self._init_func = init_func
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
        self.reset_parameters()

        self.to(device)

    @staticmethod
    def _init_weights(layer, init_func):
        if isinstance(layer, torch.nn.Linear):
            # reset weights *and* biases
            layer.reset_parameters()
            # overwrite weights if desired
            if init_func is not None:
                init_func(layer.weight)

    def reset_parameters(self):
        self._models.apply(functools.partial(self._init_weights, init_func=self._init_func))

    def forward(self, input, return_predictions=False):

        # include batch dimensions in predictions array
        preds = torch.zeros(
            (self._num_models, self._num_passes, *input.shape[:-1], self._output_size),
            device=self._device)

        # iterate over Ensemble models
        for i in range(self._num_models):
            # iterate over passes of single MC Dropout model
            for j in range(self._num_passes):
                layer_input = input
                for layer in self._models[i]:
                    layer_input = layer(layer_input)
                preds[i, j] = layer_input

        # calculate mean and variance of models and passes
        output_mean = torch.mean(preds, dim=(0, 1))
        output_var = torch.var(preds, dim=(0, 1))

        if return_predictions:
            return output_mean, output_var, preds

        return output_mean, output_var


class UncertaintyRNN(torch.nn.Module):
    """
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            num_layers: int,
            num_passes: int,
            num_models: int,
            dropout_prob: float,
            device: str):

        super().__init__()

        self._hidden_size = hidden_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._num_passes = num_passes
        self._num_models = num_models
        self._device = device

        # create model
        model = torch.nn.Module()
        model.rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_prob)
        model.activation = torch.nn.LeakyReLU()
        model.linear = torch.nn.Linear(hidden_size, output_size)

        # create ensemble
        self._models = torch.nn.ModuleList()
        for _ in range(self._num_models):
            self._models.append(copy.deepcopy(model))

        # re-initialize parameters to ensure diversity in each model of the ensemble
        self.reset_parameters()

        self.to(device)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, torch.nn.GRU):
            # reset weights *and* biases
            layer.reset_parameters()
            # overwrite weights if desired
            # TODO add custom weight initialization
        elif isinstance(layer, torch.nn.Linear):
            # reset weights *and* biases
            layer.reset_parameters()
            # overwrite weights if desired
            # TODO add custom weight initialization

    def reset_parameters(self):
        self._models.apply(self._init_weights)

    def forward(self, input, hidden=None, return_predictions=False):

        # always have dropout enabled
        self.train()

        # hidden has to be of shape (num_models, ...)
        if hidden is None:
            hidden = self.init_hidden()

        # include sequence length and batch dimensions in predictions array
        preds = torch.zeros(
            (self._num_models, self._num_passes, *input.shape[:-1], self._output_size),
            device=self._device)

        # iterate over Ensemble models
        for i in range(self._num_models):
            # iterate over passes of single MC Dropout model
            for j in range(self._num_passes):
                layer_input = input#.detach()
                # calling clone is needed to avoid an in-place operation (breaks autograd)
                layer_input, hidden[i, j] = self._models[i].rnn(layer_input, hidden[i, j].clone())
                layer_input = self._models[i].activation(layer_input)
                layer_input = self._models[i].linear(layer_input)
                preds[i, j] = layer_input

        # calculate mean and variance of models and passes
        output_mean = torch.mean(preds, dim=(0, 1))
        output_var = torch.var(preds, dim=(0, 1))

        if return_predictions:
            return output_mean, output_var, preds, hidden

        return output_mean, output_var, hidden

    def init_hidden(self, batch_size=None):
        if batch_size == None:
            # omit batch dimension
            shape = (self._num_models, self._num_passes, self._num_layers, self._hidden_size)
        else:
            shape = (self._num_models, self._num_passes, self._num_layers, batch_size, self._hidden_size)

        hidden = torch.zeros(shape, device=self._device)
        return hidden
