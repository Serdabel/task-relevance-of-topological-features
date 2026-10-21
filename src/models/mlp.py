import torch
import torch.nn as nn
from typing import Literal


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation_cls: type[nn.Module],
    ):
        super().__init__()

        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation_cls()

    def forward(
        self,
        x: torch.Tensor,
        kind: Literal["pre", "post"] = "post",
    ) -> torch.Tensor:
        z = self.linear(x)

        if kind == "pre":
            return z
        if kind == "post":
            return self.activation(z)

        raise ValueError("kind must be 'pre' or 'post'")


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 3,
        hidden_dim: int = 5,
        n_layers: int = 5,
        out_dim: int = 1,
        activation_cls: type[nn.Module] = nn.Tanh,
        include_output_layer: bool = True,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.include_output_layer = include_output_layer

        blocks = []

        current_dim = in_dim
        for _ in range(n_layers):
            blocks.append(
                MLPBlock(
                    in_dim=current_dim,
                    out_dim=hidden_dim,
                    activation_cls=activation_cls,
                )
            )
            current_dim = hidden_dim

        self.blocks = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x

        for block in self.blocks:
            h = block(h, kind="post")

        if self.include_output_layer:
            h = self.output_layer(h)

        return h

    def forward_until(
        self,
        x: torch.Tensor,
        layer_id: int,
        kind: Literal["pre", "post"] = "post",
    ) -> torch.Tensor:
        if layer_id < 0 or layer_id >= self.n_layers:
            raise ValueError(
                f"Invalid layer_id={layer_id}. Expected 0 <= layer_id < {self.n_layers}"
            )

        h = x

        for i, block in enumerate(self.blocks):
            z = block.linear(h)
            h_post = block.activation(z)

            if i == layer_id:
                return z if kind == "pre" else h_post

            h = h_post

        raise RuntimeError("Unreachable code")

    def get_all_layer_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = {}
        h = x

        for i, block in enumerate(self.blocks):
            z = block.linear(h)
            h = block.activation(z)

            outputs[f"layer_{i}_pre"] = z
            outputs[f"layer_{i}_post"] = h

        outputs["logits"] = self.output_layer(h)

        return outputs

    def hidden_output_dim(self, layer_id: int) -> int:
        if layer_id < 0 or layer_id >= self.n_layers:
            raise ValueError(f"Invalid layer_id={layer_id}")

        return self.blocks[layer_id].linear.out_features