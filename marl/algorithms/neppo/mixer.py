import torch as th
from torch import nn


class NPLambda(nn.Module):
    def __init__(self, *, output_dim, **_):
        super().__init__()
        self.logits = nn.Parameter(th.zeros(output_dim))
        # self.scale = nn.Parameter(th.tensor(1.0))
        # self.logits = nn.Parameter(th.ones(output_dim))
        # self.logits = nn.Parameter(th.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 10.0]))

    def forward(self, state, reward):
        # lambdas = th.softmax(self.logits, dim=0) * self.scale
        lambdas = th.softmax(self.logits, dim=0)
        # lambdas = self.logits
        return lambdas @ reward


class NPMixer(nn.Module):
    def __init__(self, *, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, state, reward):
        logits = self.linear(state)
        lambdas = th.softmax(logits, dim=-1)
        return th.sum(lambdas.T * reward, dim=0)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Key line: apply orthogonal initialization to weights
            th.nn.init.orthogonal_(module.weight)
            # Initialize biases to zeros
            if module.bias is not None:
                th.nn.init.zeros_(module.bias)


class _NonNegWeight(nn.Module):
    """Parametrization to ensure non-negative weights via ReLU."""

    def forward(self, w):
        return th.relu(w)


class NPConvexNN(nn.Module):
    def __init__(self, *, input_dim, output_dim, hidden_size=16):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.output_layer_y = nn.Linear(input_dim, 1)
        self.output_layer_z = nn.Linear(hidden_size, 1, bias=False)
        th.nn.utils.parametrize.register_parametrization(
            self.output_layer_z,
            "weight",
            _NonNegWeight(),
        )

        self.softplus = nn.Softplus()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Key line: apply orthogonal initialization to weights
            th.nn.init.orthogonal_(module.weight)
            # Initialize biases to zeros
            if module.bias is not None:
                th.nn.init.zeros_(module.bias)

    def forward(self, state, reward):
        out_z = self.input_layer(state)
        out_z = self.softplus(out_z)
        out_z = self.output_layer_z(out_z)

        out_y = self.output_layer_y(state)

        logits = out_y + out_z
        # lambdas = th.softmax(logits, dim=-1)
        return -1 * logits.squeeze(-1)


class NPMixerWithDelta(nn.Module):
    def __init__(self, *, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        self.linear_delta = nn.Linear(input_dim, 1)
        nn.init.zeros_(self.linear_delta.weight)
        nn.init.zeros_(self.linear_delta.bias)

    def forward(self, state, reward):
        logits = self.linear(state)
        lambdas = th.softmax(logits, dim=-1)
        return th.sum(lambdas.T * reward, dim=0) + self.linear_delta(state).squeeze(-1)


class NPNoMix(nn.Module):
    def __init__(self, *, input_dim, hidden_size=64, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Key line: apply orthogonal initialization to weights
            th.nn.init.orthogonal_(module.weight)
            # Initialize biases to zeros
            if module.bias is not None:
                th.nn.init.zeros_(module.bias)

    def forward(self, state, reward):
        return self.net(state).squeeze(-1)
