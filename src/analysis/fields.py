"""Evaluated scalar fields: the shared input object of all criteria.

A ``FieldSample`` is a scalar function evaluated on a finite point sample —
either a regular grid (the 2D/3D default) or an arbitrary point cloud
(gradient-sampled level sets, data clouds, PCA-slice grids). Criteria consume a
``FieldSample`` instead of building their own grid, which

  * removes the hard grid dependency from the criteria (higher-dimensional
    samplers produce point clouds, not grids), and
  * lets ``evaluate_neurons_on_feature`` evaluate each neuron once and share
    the values across all criteria.

Producers:

    evaluate_field_on_grid(function, bounds, resolution)  -> grid FieldSample
    field_from_points(function, points)                   -> cloud FieldSample
"""

from dataclasses import dataclass

import torch

from analysis.geometry import (
    make_grid,
    grid_step_from_bounds_resolution,
    evaluate_scalar_function_on_points,
)


@dataclass
class FieldSample:
    """A scalar field evaluated on a finite point sample.

    points:
        Tensor [N, dim] — where the field was evaluated.

    values:
        Tensor [N] — field values at ``points``.

    grid_shape:
        Tuple (N1, ..., Nd) if the sample is a regular grid in row-major
        (indexing="ij") order, else None. Grid structure enables the
        image-labeling connected-components path and adaptive level_eps.

    grid_step:
        Maximal axis step of the grid, if the sample is a grid, else None.

    bounds:
        [(a1, b1), ..., (ad, bd)] the sample was built from, if known.
    """

    points: torch.Tensor
    values: torch.Tensor
    grid_shape: tuple[int, ...] | None = None
    grid_step: float | None = None
    bounds: list[tuple[float, float]] | None = None

    def __post_init__(self):
        assert self.points.ndim == 2, "points must have shape [N, dim]"
        assert self.values.ndim == 1, "values must have shape [N]"
        assert len(self.points) == len(self.values), (
            "points and values must have the same length"
        )

        if self.grid_shape is not None:
            expected = 1
            for n in self.grid_shape:
                expected *= n
            assert expected == len(self.points), (
                f"grid_shape {self.grid_shape} does not match {len(self.points)} points"
            )

    @property
    def is_grid(self) -> bool:
        return self.grid_shape is not None

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def values_nd(self) -> torch.Tensor:
        """Values reshaped to the grid, [N1, ..., Nd]. Grid samples only."""
        assert self.is_grid, "values_nd is only defined for grid samples"
        return self.values.reshape(self.grid_shape)

    @property
    def points_nd(self) -> torch.Tensor:
        """Points reshaped to the grid, [N1, ..., Nd, dim]. Grid samples only."""
        assert self.is_grid, "points_nd is only defined for grid samples"
        return self.points.reshape(*self.grid_shape, self.dim)

    def to(self, device=None, dtype=None) -> "FieldSample":
        return FieldSample(
            points=self.points.to(device=device, dtype=dtype),
            values=self.values.to(device=device, dtype=dtype),
            grid_shape=self.grid_shape,
            grid_step=self.grid_step,
            bounds=self.bounds,
        )


def evaluate_field_on_grid(
    function,
    bounds,
    resolution: int | list[int] | tuple[int, ...] = 100,
    batch_size: int = 8192,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> FieldSample:
    """Evaluates ``function`` on a regular grid over ``bounds``."""
    assert len(bounds) > 0, "bounds must be non-empty"

    grid_points, grid_shape = make_grid(
        bounds=bounds,
        resolution=resolution,
        device=device,
        dtype=dtype,
    )

    values = evaluate_scalar_function_on_points(
        function=function,
        points=grid_points,
        batch_size=batch_size,
    ).to(device=device, dtype=dtype)

    assert values.shape[0] == grid_points.shape[0], (
        "function returned wrong number of values"
    )

    grid_step = max(float(grid_step_from_bounds_resolution(bounds, resolution)), 1e-12)

    return FieldSample(
        points=grid_points,
        values=values,
        grid_shape=grid_shape,
        grid_step=grid_step,
        bounds=[tuple(b) for b in bounds],
    )


def field_from_points(
    function,
    points: torch.Tensor,
    batch_size: int = 8192,
    bounds=None,
) -> FieldSample:
    """Evaluates ``function`` on an arbitrary point cloud (gridless sample)."""
    assert points.ndim == 2, "points must have shape [N, dim]"

    values = evaluate_scalar_function_on_points(
        function=function,
        points=points,
        batch_size=batch_size,
    ).to(device=points.device)

    return FieldSample(
        points=points,
        values=values,
        grid_shape=None,
        grid_step=None,
        bounds=None if bounds is None else [tuple(b) for b in bounds],
    )
