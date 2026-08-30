import torch as th
import torch.nn as nn


class PopArt(nn.Module):
    """PopArt normalization wrapper for a value network.

    Maintains running mean (mu) and std (sigma) of value targets.
    Rescales output layer weights when statistics update to preserve function.
    """

    def __init__(
        self,
        value_network: nn.Module,
        beta: float = 0.0001,
        eps: float = 1e-5,
        value_key="state_value",
    ):
        super().__init__()
        self.value_network = value_network
        self.beta = beta
        self.eps = eps
        self.value_key = value_key

        self.register_buffer("mu", th.zeros(1))
        self.register_buffer("nu", th.ones(1))  # E[V^2]
        self.register_buffer("sigma", th.ones(1))
        self.register_buffer("count", th.zeros(1))

        self._output_layer = self._find_output_layer()

    @property
    def in_keys(self):
        """Proxy in_keys from wrapped value network for TorchRL compatibility."""
        return self.value_network.in_keys

    @property
    def out_keys(self):
        """Proxy out_keys from wrapped value network for TorchRL compatibility."""
        return self.value_network.out_keys

    def _find_output_layer(self) -> nn.Linear:
        """Find the final Linear layer in the value network."""
        module = self.value_network
        if hasattr(module, "module"):
            module = module.module
        last_linear = None
        for m in module.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ValueError("Could not find Linear output layer")
        return last_linear

    @property
    def std(self) -> th.Tensor:
        return th.sqrt(self.nu - self.mu**2).clamp(min=self.eps)

    def _get_mu_sigma(self, device):
        """Get mu and sigma on the specified device."""
        return self.mu.to(device), self.sigma.to(device)

    def forward(self, tensordict):
        """Forward pass - returns denormalized values for GAE."""
        tensordict = self.value_network(tensordict)
        normalized_value = tensordict[self.value_key]
        mu, sigma = self._get_mu_sigma(normalized_value.device)
        tensordict[self.value_key] = normalized_value * sigma + mu
        return tensordict

    def normalize_target(self, target: th.Tensor) -> th.Tensor:
        mu, sigma = self._get_mu_sigma(target.device)
        return (target - mu) / sigma

    def denormalize(self, normalized_value: th.Tensor) -> th.Tensor:
        mu, sigma = self._get_mu_sigma(normalized_value.device)
        return normalized_value * sigma + mu

    @th.no_grad()
    def update(self, targets: th.Tensor) -> None:
        """Update running statistics and rescale output layer."""
        device = targets.device
        mu, sigma = self._get_mu_sigma(device)
        nu = self.nu.to(device)
        old_mu, old_sigma = mu.clone(), sigma.clone()

        batch_mean = targets.mean()
        batch_nu = (targets**2).mean()

        if self.count == 0:
            new_mu = batch_mean
            new_nu = batch_nu
        else:
            new_mu = (1 - self.beta) * mu + self.beta * batch_mean
            new_nu = (1 - self.beta) * nu + self.beta * batch_nu

        self.mu.copy_(new_mu)
        self.nu.copy_(new_nu)
        self.count += 1
        new_sigma = th.sqrt(new_nu - new_mu**2).clamp(min=self.eps)
        self.sigma.copy_(new_sigma)

        # Rescale output layer to preserve function
        if self.count > 1:
            scale = old_sigma / new_sigma
            shift = (old_mu - new_mu) / new_sigma
            self._output_layer.weight.mul_(scale.to(self._output_layer.weight.device))
            self._output_layer.bias.mul_(scale.to(self._output_layer.bias.device)).add_(
                shift.to(self._output_layer.bias.device)
            )

    def parameters(self):
        return self.value_network.parameters()
