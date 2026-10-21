import torch
import torch.nn as nn
from typing import Literal, Optional


class TruncatedMLPFunction:
    """
    Represents:
        F_j = f_j ∘ f_{j-1} ∘ ... ∘ f_1

    Returns vector-valued output:
        F_j(x) ∈ R^{hidden_dim}
    """

    def __init__(
        self,
        model,
        layer_id: int,
        kind: Literal["pre", "post"] = "post",
    ):
        self.model = model
        self.layer_id = layer_id
        self.kind = kind

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_until(
            x,
            layer_id=self.layer_id,
            kind=self.kind,
        )

    @property
    def output_dim(self) -> int:
        return self.model.hidden_output_dim(self.layer_id)

    @property
    def name(self) -> str:
        return f"F_{self.layer_id}_{self.kind}"


class ScalarProjection:
    """
    Turns vector-valued F_j(x) into scalar function.

    Examples:
        coordinate i:
            x -> F_j(x)[i]

        max coordinate:
            x -> max_i F_j(x)[i]

        norm:
            x -> ||F_j(x)||

        logsumexp:
            x -> logsumexp_i F_j(x)[i]

        mean:
            x -> mean_i F_j(x)[i]
    """

    def __init__(
        self,
        vector_function: TruncatedMLPFunction,
        mode: Literal["coord", "max", "norm", "logsumexp", "mean"] = "coord",
        coord_id: Optional[int] = None,
    ):
        self.vector_function = vector_function
        self.mode = mode
        self.coord_id = coord_id

        if self.mode == "coord" and self.coord_id is None:
            raise ValueError("coord_id must be provided for mode='coord'")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = self.vector_function(x)

        if self.mode == "coord":
            if self.coord_id < 0 or self.coord_id >= y.shape[1]:
                raise ValueError(f"Invalid coord_id={self.coord_id}, output dim is {y.shape[1]}")
            return y[:, self.coord_id]

        if self.mode == "max":
            return y.max(dim=1).values

        if self.mode == "norm":
            return y.norm(dim=1)

        if self.mode == "logsumexp":
            return torch.logsumexp(y, dim=1)

        if self.mode == "mean":
            return y.mean(dim=1)

        raise ValueError(f"Unknown projection mode: {self.mode}")

    @property
    def name(self) -> str:
        if self.mode == "coord":
            return f"{self.vector_function.name}_coord_{self.coord_id}"

        return f"{self.vector_function.name}_{self.mode}"


class SubspaceRestrictedFunction:
    """
    Restricts a scalar field f: R^d -> R to a k-dimensional affine subspace:

        g(u) = f(origin + u @ basis),   u in R^k.

    The input-side dual of ScalarProjection (which projects the OUTPUT of a
    truncated MLP). This is the PCA-slice enabler for higher dimensions
    (PLAN.md §Sampling): fix the subspace FIRST (PCA on ambient-space anchor
    points, e.g. a gradient-ascent trajectory to a neuron's local extremum,
    Wang-Ponce style), then compute level sets of the restricted function
    SECOND — g is again a scalar field on R^k, so the whole 2D toolkit (grid
    criteria, heatmaps) applies unchanged in subspace coordinates.

    origin: Tensor [d]
    basis:  Tensor [k, d], rows should be orthonormal (as returned by
            analysis.geometry.fit_affine_subspace).
    """

    def __init__(
        self,
        function,
        origin: torch.Tensor,
        basis: torch.Tensor,
    ):
        assert origin.ndim == 1, "origin must have shape [dim]"
        assert basis.ndim == 2, "basis must have shape [k, dim]"
        assert basis.shape[1] == origin.shape[0], (
            "basis and origin must share the ambient dimension"
        )

        self.function = function
        self.origin = origin
        self.basis = basis

    @classmethod
    def from_points(cls, function, points: torch.Tensor, k: int = 2):
        """Fits the subspace by PCA on ambient-space anchor points."""
        from analysis.geometry import fit_affine_subspace

        subspace = fit_affine_subspace(points, k=k)

        restricted = cls(
            function=function,
            origin=subspace["origin"],
            basis=subspace["basis"],
        )
        restricted.explained_variance_ratio = subspace["explained_variance_ratio"]

        return restricted

    @property
    def subspace_dim(self) -> int:
        return self.basis.shape[0]

    @property
    def ambient_dim(self) -> int:
        return self.basis.shape[1]

    def embed(self, u: torch.Tensor) -> torch.Tensor:
        """Subspace coordinates [B, k] -> ambient points [B, dim]."""
        assert u.ndim == 2 and u.shape[1] == self.subspace_dim, (
            f"expected input of shape [B, {self.subspace_dim}]"
        )
        return self.origin.to(u.device, u.dtype) + u @ self.basis.to(u.device, u.dtype)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """Ambient points [B, dim] -> subspace coordinates [B, k]."""
        assert x.ndim == 2 and x.shape[1] == self.ambient_dim, (
            f"expected input of shape [B, {self.ambient_dim}]"
        )
        centered = x - self.origin.to(x.device, x.dtype)
        return centered @ self.basis.to(x.device, x.dtype).T

    def bounds_for_points(self, x: torch.Tensor, margin: float = 0.5):
        """Subspace bounds [(lo, hi)] * k covering the projection of x."""
        u = self.project(x)
        mins = u.min(dim=0).values
        maxs = u.max(dim=0).values
        return [
            (float(lo.item()) - margin, float(hi.item()) + margin)
            for lo, hi in zip(mins, maxs)
        ]

    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return self.function(self.embed(u))

    @property
    def name(self) -> str:
        inner = getattr(self.function, "name", type(self.function).__name__)
        return f"{inner}_slice{self.subspace_dim}d"


# -------------------------
# MLP structure introspection
# -------------------------

def validate_mlp_layer_id(model, layer_id: int):
    """
    Checks that model has the expected MLP structure and that layer_id is valid.
    """
    assert hasattr(model, "n_layers"), "model must have attribute n_layers"
    assert hasattr(model, "blocks"), "model must have attribute blocks"

    assert isinstance(layer_id, int), "layer_id must be an int"
    assert 0 <= layer_id < model.n_layers, (
        f"Invalid layer_id={layer_id}. Expected 0 <= layer_id < {model.n_layers}"
    )

    assert len(model.blocks) == model.n_layers, (
        "Expected len(model.blocks) == model.n_layers"
    )

    block = model.blocks[layer_id]

    assert hasattr(block, "linear"), "Each MLPBlock must have attribute linear"
    assert hasattr(block, "activation"), "Each MLPBlock must have attribute activation"

    return block


def infer_activation_name_from_model(model, layer_id: int) -> str | None:
    """
    Infers activation name from model.blocks[layer_id].activation.

    Intended for post-activation neurons.
    For pre-activation neurons, activation should usually be treated as None.
    """
    block = validate_mlp_layer_id(model, layer_id)

    activation = block.activation

    if isinstance(activation, nn.Tanh):
        return "tanh"

    if isinstance(activation, nn.Sigmoid):
        return "sigmoid"

    if isinstance(activation, nn.ReLU):
        return "relu"

    return None


def get_hidden_dim(model, layer_id: int) -> int:
    """
    Returns the number of scalar neurons in a hidden layer.
    """
    block = validate_mlp_layer_id(model, layer_id)

    return block.linear.out_features


def infer_level_activation_for_neuron(model, layer_id: int, kind: str) -> str | None:
    """
    Returns activation name only for post-activation neurons.

    kind="post":
        use activation range, e.g. tanh -> [-1, 1].

    kind="pre":
        return None, because pre-activations are not bounded by activation range.
    """
    assert kind in {"pre", "post"}, "kind must be 'pre' or 'post'"

    if kind == "pre":
        return None

    return infer_activation_name_from_model(model, layer_id)
