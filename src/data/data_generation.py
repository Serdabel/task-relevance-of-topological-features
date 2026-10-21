import torch
from abc import ABC


class Object(ABC):
    def __init__(self, dim: int, name: str | None = None, metadata: dict | None = None):
        self.dim = dim
        self.data = None
        self.labels = None
        self.name = name
        self.metadata = {} if metadata is None else dict(metadata)

    def get_data(self):
        return self.data

    def get_labels(self):
        return self.labels

    def apply(self, linear: torch.Tensor, bias: torch.Tensor):
        assert self.data is not None
        assert linear.shape == (self.dim, self.dim)
        assert bias.shape == (self.dim,)

        self.data = self.data @ linear.T + bias

        transforms = self.metadata.setdefault("transforms", [])
        transforms.append(
            {
                "linear": linear.detach().cpu(),
                "bias": bias.detach().cpu(),
            }
        )

        return self

class PointCloudObject(Object):
    """
    Object built directly from already sampled points.

    Useful for reconstructing Data from stored components.
    """

    def __init__(
        self,
        data: torch.Tensor,
        labels: torch.Tensor | None = None,
        label: int | None = None,
        name: str | None = None,
        metadata: dict | None = None,
    ):
        assert torch.is_tensor(data), "data must be a torch.Tensor"
        assert data.ndim == 2, "data must have shape [n_points, dim]"

        super().__init__(dim=data.shape[1], name=name, metadata=metadata)

        self.data = data.detach().clone()

        if labels is not None:
            assert torch.is_tensor(labels), "labels must be a torch.Tensor"
            assert labels.ndim == 1, "labels must have shape [n_points]"
            assert len(labels) == len(data), "labels and data must have same length"
            self.labels = labels.detach().clone().long()
        else:
            assert label is not None, "Either labels or label must be provided"
            self.labels = torch.full((len(data),), int(label), dtype=torch.long)


class Circle2D(Object):
    def __init__(
        self,
        n_points: int,
        sample_mode: str = "param_uniform",
        label: int = 0,
    ):
        super().__init__(2)

        self.n_points = n_points
        self.sample_mode = sample_mode

        self.data = self.sample_data_()
        self.labels = torch.full((self.n_points,), label, dtype=torch.long)

    def sample_data_(self):
        if self.sample_mode in ["param_uniform", "uniform"]:
            angles = torch.linspace(0, 2 * torch.pi, self.n_points + 1)[:-1]

            x = torch.cos(angles)
            y = torch.sin(angles)

            return torch.stack([x, y], dim=1)

        raise ValueError(f"Unknown sample_mode: {self.sample_mode}")


class Circle3D(Object):
    def __init__(
        self,
        n_points: int,
        sample_mode: str = "param_uniform",
        label: int = 0,
    ):
        super().__init__(3)

        self.n_points = n_points
        self.sample_mode = sample_mode

        self.data = self.sample_data_()
        self.labels = torch.full((self.n_points,), label, dtype=torch.long)

    def sample_data_(self):
        if self.sample_mode in ["param_uniform", "uniform"]:
            angles = torch.linspace(0, 2 * torch.pi, self.n_points + 1)[:-1]

            x = torch.cos(angles)
            y = torch.sin(angles)
            z = torch.zeros_like(x)

            return torch.stack([x, y, z], dim=1)

        raise ValueError(f"Unknown sample_mode: {self.sample_mode}")


class Data(Object):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.objects = []
        self.object_names = []
        self.features = {}

    def _make_default_name(self):
        return f"object_{len(self.objects)}"

    def _validate_new_name(self, name: str):
        assert isinstance(name, str) and len(name) > 0, "name must be non-empty string"
        assert name not in self.object_names, f"Duplicate object name: {name}"

    def add(
        self,
        obj: Object,
        name: str | None = None,
        is_feature: bool = False,
        metadata: dict | None = None,
    ):
        if obj.dim != self.dim:
            raise ValueError(f"Only {self.dim}D objects are allowed, got {obj.dim}D")

        if name is None:
            name = obj.name if obj.name is not None else self._make_default_name()

        self._validate_new_name(name)

        obj.name = name

        if not hasattr(obj, "metadata") or obj.metadata is None:
            obj.metadata = {}

        if metadata is not None:
            obj.metadata.update(dict(metadata))

        self.objects.append(obj)
        self.object_names.append(name)

        if is_feature:
            self.add_feature(
                name=name,
                object_names=[name],
                metadata={
                    "source": "component",
                    "component_name": name,
                },
            )

        return self

    def add_feature(
        self,
        name: str,
        points: torch.Tensor | None = None,
        object_names: list[str] | None = None,
        metadata: dict | None = None,
    ):
        """
        Registers an analysis feature.

        Feature can be either:

        1. alias to one or several named objects:
            object_names=["C_mid"]

        2. explicit point cloud:
            points=feature_points

        For your current use case, the main feature is usually C_mid.
        """
        assert isinstance(name, str) and len(name) > 0, "feature name must be non-empty"
        assert name not in self.features, f"Duplicate feature name: {name}"

        if points is None:
            assert object_names is not None and len(object_names) > 0, (
                "Either points or object_names must be provided"
            )

        if object_names is not None:
            for obj_name in object_names:
                assert obj_name in self.object_names, f"Unknown object name: {obj_name}"

        if points is not None:
            assert torch.is_tensor(points), "points must be a torch.Tensor"
            assert points.ndim == 2, "points must have shape [n_points, dim]"
            assert points.shape[1] == self.dim, "feature dimension mismatch"
            points = points.detach().clone()

        self.features[name] = {
            "name": name,
            "points": points,
            "object_names": None if object_names is None else list(object_names),
            "metadata": {} if metadata is None else dict(metadata),
        }

        return self

    def get_object(self, name: str):
        assert name in self.object_names, f"Unknown object name: {name}"
        idx = self.object_names.index(name)
        return self.objects[idx]

    def get_component_data(self, name: str):
        return self.get_object(name).get_data()

    def get_component_labels(self, name: str):
        return self.get_object(name).get_labels()

    def get_feature_points(self, name: str):
        assert name in self.features, f"Unknown feature name: {name}"

        feature = self.features[name]

        if feature["points"] is not None:
            return feature["points"]

        object_names = feature["object_names"]
        return torch.cat(
            [self.get_component_data(obj_name) for obj_name in object_names],
            dim=0,
        )

    def get_data(self):
        if len(self.objects) == 0:
            raise ValueError("No objects added")

        return torch.cat([obj.get_data() for obj in self.objects], dim=0)

    def get_labels(self):
        if len(self.objects) == 0:
            raise ValueError("No objects added")

        return torch.cat([obj.get_labels() for obj in self.objects], dim=0)

    def get_component_ids(self):
        if len(self.objects) == 0:
            raise ValueError("No objects added")

        ids = []
        for component_id, obj in enumerate(self.objects):
            ids.append(
                torch.full(
                    (len(obj.get_data()),),
                    component_id,
                    dtype=torch.long,
                )
            )

        return torch.cat(ids, dim=0)

    def apply(self, linear: torch.Tensor, bias: torch.Tensor):
        for obj in self.objects:
            obj.apply(linear, bias)

        # Explicit feature point clouds must also be transformed.
        # Alias-features via object_names need no update.
        for feature in self.features.values():
            if feature["points"] is not None:
                feature["points"] = feature["points"] @ linear.T + bias

        return self

    def plot(self, interactive=True, size=4, title="Data", elev=20, azim=0):
        # Imported lazily so the data layer stays usable without plotting deps.
        from visualization.plot_point_clouds import (
            plot_point_cloud_3d,
            plot_point_cloud_3d_interactive,
            plot_point_cloud_2d_interactive,
        )

        X = self.get_data()
        y = self.get_labels()

        if self.dim == 2:
            if interactive:
                plot_point_cloud_2d_interactive(
                    X,
                    y,
                    level_points=None,
                    size=size,
                    title=title,
                )
            else:
                raise NotImplementedError()

        elif self.dim == 3:
            if interactive:
                plot_point_cloud_3d_interactive(
                    X,
                    y,
                    level_points=None,
                    size=size,
                    title=title,
                )
            else:
                plot_point_cloud_3d(
                    X,
                    y,
                    level_points=None,
                    size=size,
                    title=title,
                    elev=elev,
                    azim=azim,
                )

        return self


class Disk2D(Object):
    """
    Unit disk in R^2:

        D = {x^2 + y^2 <= 1}
    """

    def __init__(
        self,
        n_points: int,
        label: int = 0,
    ):
        super().__init__(2)

        self.n_points = n_points

        self.data = self.sample_data_()
        self.labels = torch.full((self.n_points,), label, dtype=torch.long)

    def sample_data_(self):
        angles = 2 * torch.pi * torch.rand(self.n_points)

        # sqrt нужен для равномерности по площади диска
        radii = torch.sqrt(torch.rand(self.n_points))

        x = radii * torch.cos(angles)
        y = radii * torch.sin(angles)

        return torch.stack([x, y], dim=1)

class Rectangle2D(Object):
    """
    Thin rectangle boundary or its tubular neighborhood in R^2.

    Base thin rectangle boundary:

        [-width/2, width/2] x [-height/2, height/2]

    If boundary_thickness == 0:
        samples the thin rectangle boundary.

    If boundary_thickness > 0:
        samples the set of points whose Euclidean distance
        to the thin rectangle boundary is <= boundary_thickness.

    So boundary_thickness is a radius around the thin rectangle boundary,
    not a one-sided inward thickness.
    """

    def __init__(
        self,
        n_points: int,
        width: float = 2.0,
        height: float = 2.0,
        boundary_thickness: float = 0.0,
        label: int = 0,
    ):
        super().__init__(2)

        assert n_points > 0, "n_points must be positive"
        assert width > 0, "width must be positive"
        assert height > 0, "height must be positive"
        assert boundary_thickness >= 0, "boundary_thickness must be non-negative"

        self.n_points = n_points
        self.width = width
        self.height = height
        self.boundary_thickness = boundary_thickness

        self.data = self.sample_data_()
        self.labels = torch.full((self.n_points,), label, dtype=torch.long)

    def sample_data_(self):
        if self.boundary_thickness == 0.0:
            return self._sample_thin_boundary()

        return self._sample_tubular_boundary()

    def _sample_thin_boundary(self):
        """
        Uniform random sampling by arclength from the thin rectangle boundary.
        """

        w = self.width
        h = self.height

        half_w = w / 2
        half_h = h / 2

        perimeter = 2 * (w + h)
        s = perimeter * torch.rand(self.n_points)

        x = torch.empty(self.n_points)
        y = torch.empty(self.n_points)

        # bottom side
        mask = (0 <= s) & (s < w)
        x[mask] = -half_w + s[mask]
        y[mask] = -half_h

        # right side
        mask = (w <= s) & (s < w + h)
        x[mask] = half_w
        y[mask] = -half_h + (s[mask] - w)

        # top side
        mask = (w + h <= s) & (s < 2 * w + h)
        x[mask] = half_w - (s[mask] - (w + h))
        y[mask] = half_h

        # left side
        mask = (2 * w + h <= s) & (s < perimeter)
        x[mask] = -half_w
        y[mask] = half_h - (s[mask] - (2 * w + h))

        return torch.stack([x, y], dim=1)

    def _distance_to_thin_boundary(self, points: torch.Tensor) -> torch.Tensor:
        """
        Euclidean distance from points to the thin rectangle boundary.

        Boundary consists of four line segments.
        """

        x = points[:, 0]
        y = points[:, 1]

        half_w = self.width / 2
        half_h = self.height / 2

        # Distance to top and bottom horizontal segments.
        dx_horizontal = torch.clamp(x.abs() - half_w, min=0.0)

        d_top = torch.sqrt(dx_horizontal ** 2 + (y - half_h) ** 2)
        d_bottom = torch.sqrt(dx_horizontal ** 2 + (y + half_h) ** 2)

        # Distance to left and right vertical segments.
        dy_vertical = torch.clamp(y.abs() - half_h, min=0.0)

        d_right = torch.sqrt((x - half_w) ** 2 + dy_vertical ** 2)
        d_left = torch.sqrt((x + half_w) ** 2 + dy_vertical ** 2)

        return torch.minimum(
            torch.minimum(d_top, d_bottom),
            torch.minimum(d_left, d_right),
        )

    def _sample_tubular_boundary(self):
        """
        Uniform rejection sampling from:

            {p : dist(p, thin_rectangle_boundary) <= boundary_thickness}

        The proposal distribution is uniform on the bounding box:

            [-width/2 - t, width/2 + t]
            x
            [-height/2 - t, height/2 + t]
        """

        t = self.boundary_thickness

        half_w = self.width / 2
        half_h = self.height / 2

        x_min = -half_w - t
        x_max = half_w + t
        y_min = -half_h - t
        y_max = half_h + t

        accepted_chunks = []
        n_accepted = 0

        while n_accepted < self.n_points:
            remaining = self.n_points - n_accepted
            n_candidates = max(4096, 10 * remaining)

            x = x_min + (x_max - x_min) * torch.rand(n_candidates)
            y = y_min + (y_max - y_min) * torch.rand(n_candidates)

            candidates = torch.stack([x, y], dim=1)

            distances = self._distance_to_thin_boundary(candidates)
            accepted = candidates[distances <= t]

            if len(accepted) > 0:
                accepted_chunks.append(accepted)
                n_accepted += len(accepted)

        points = torch.cat(accepted_chunks, dim=0)
        return points[: self.n_points]

# ---------------------------------------------------------------------------
# Objects and transforms for PLAN.md experiments 9-14. `with_gaussian_noise`
# (experiment 14) is still a scaffold; the rest is implemented.
# ---------------------------------------------------------------------------

class SphereND(Object):
    """
    Sphere S^{dim-1} in R^dim (experiments 9-10: S^2 in R^3, S^3 in R^4).

    sample_mode="uniform": iid uniform on the sphere (normalized Gaussians) —
    NOT the deterministic linspace of Circle2D; per-point density is then a
    genuine random variable, which the density experiments need.

    seed=None draws from the global torch RNG (the recipe-level
    ``torch.manual_seed(seed)`` convention); an explicit seed uses its own
    generator, independent of global state.
    """

    def __init__(
        self,
        n_points: int,
        dim: int,
        sample_mode: str = "uniform",
        label: int = 0,
        seed: int | None = None,
    ):
        super().__init__(dim)

        assert n_points > 0, "n_points must be positive"
        assert dim >= 2, "dim must be >= 2"

        if sample_mode != "uniform":
            raise ValueError(f"Unknown sample_mode: {sample_mode}")

        self.n_points = n_points
        self.sample_mode = sample_mode

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        gaussian = torch.randn(n_points, dim, generator=generator)
        norms = gaussian.norm(dim=1, keepdim=True).clamp_min(1e-12)

        self.data = gaussian / norms
        self.labels = torch.full((n_points,), label, dtype=torch.long)


class BallND(Object):
    """
    Filled ball B^dim in R^dim — the interior/obstruction counterpart of
    SphereND (uniform in volume: Gaussian direction x U^{1/dim} radius).

    seed semantics match SphereND.
    """

    def __init__(
        self,
        n_points: int,
        dim: int,
        label: int = 0,
        seed: int | None = None,
    ):
        super().__init__(dim)

        assert n_points > 0, "n_points must be positive"
        assert dim >= 2, "dim must be >= 2"

        self.n_points = n_points

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        gaussian = torch.randn(n_points, dim, generator=generator)
        norms = gaussian.norm(dim=1, keepdim=True).clamp_min(1e-12)
        directions = gaussian / norms

        radii = torch.rand(n_points, 1, generator=generator) ** (1.0 / dim)

        self.data = directions * radii
        self.labels = torch.full((n_points,), label, dtype=torch.long)


class WavyCircle2D(Object):
    """
    Circle with controllable high-frequency radial perturbation
    (experiment 13, scale/locality):

        r(t) = 1 + amplitude * sin(frequency * t + phase)

    ``amplitude``/``frequency`` are the scale knobs: at which amplitude/
    frequency do neuron level sets stop following the feature's details?

    Sampling is deterministic in the parameter t (like Circle2D), uniform in
    t — NOT arclength-uniform: high-curvature wave crests get slightly lower
    linear density, which is acceptable at the amplitudes of exp. 13.
    """

    def __init__(
        self,
        n_points: int,
        amplitude: float = 0.1,
        frequency: int = 8,
        phase: float = 0.0,
        label: int = 0,
        seed: int | None = None,
    ):
        super().__init__(2)

        assert n_points > 0, "n_points must be positive"
        assert amplitude >= 0, "amplitude must be non-negative"
        assert frequency >= 1, "frequency must be >= 1"
        assert amplitude < 1.0, "amplitude must be < 1 (radius stays positive)"

        self.n_points = n_points
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

        t = torch.linspace(0, 2 * torch.pi, n_points + 1)[:-1]
        r = 1.0 + amplitude * torch.sin(frequency * t + phase)

        self.data = torch.stack([r * torch.cos(t), r * torch.sin(t)], dim=1)
        self.labels = torch.full((n_points,), label, dtype=torch.long)


class RadialCurve2D(Object):
    """
    Closed curve given in polar form ``r = radial(theta)``, experiment 5.

    Two things make this different from ``Circle2D().apply(ellipse_matrix(...))``,
    and both matter:

    1. **Sampling is uniform in ARC LENGTH, not in the parameter.** An ellipse
       traced as ``(a cos t, b sin t)`` at equal steps in ``t`` piles points up
       at the ends of the major axis, because the speed there is smallest. That
       is a density perturbation on the same harmonic and in the same phase as
       the shape perturbation — shape and density become locked together, and
       the experiment that is supposed to separate them cannot. (The same defect
       sits in ``WavyCircle2D``.)

    2. **The angular density is a separate, explicit knob.** Passing
       ``angular_density`` makes the LINEAR density of points along the curve
       proportional to it, leaving the support — the curve itself — untouched.
       So a perfectly round curve can carry an elliptic point distribution, and
       an ellipse can carry a perfectly even one. Those are the two cells that
       separate the geometric effect from the density effect.

    ``radial`` and ``angular_density`` are callables on a torch tensor of angles
    (vectorised); ``rotation`` turns the finished curve, so ``radius(theta)``
    reports the world-frame radius.
    """

    def __init__(
        self,
        n_points: int,
        radial,
        angular_density=None,
        rotation: float = 0.0,
        label: int = 0,
        n_quadrature: int = 8192,
        name: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(2, name=name, metadata=metadata)

        assert n_points > 0, "n_points must be positive"
        assert n_quadrature >= 64, "n_quadrature must be >= 64"

        self.n_points = n_points
        self.radial = radial
        self.angular_density = angular_density
        self.rotation = float(rotation)

        t = torch.linspace(0.0, 2 * torch.pi, n_quadrature + 1, dtype=torch.float64)
        r = radial(t).to(torch.float64)
        assert torch.all(r > 0), "radial(theta) must be strictly positive"

        # ds = sqrt(r^2 + r'^2) dt; r' by central differences on the fine grid,
        # periodic at the seam.
        dt = t[1] - t[0]
        r_periodic = torch.cat([r[-2:-1], r, r[1:2]])
        dr = (r_periodic[2:] - r_periodic[:-2]) / (2 * dt)
        speed = torch.sqrt(r ** 2 + dr ** 2)

        self.perimeter = float(torch.trapezoid(speed, t))

        weight = speed if angular_density is None else speed * angular_density(t).to(torch.float64)
        assert torch.all(weight > 0), "angular_density(theta) must be strictly positive"

        # Cumulative measure along the curve, normalised to [0, 1]; points are
        # placed at equal increments of it.
        cumulative = torch.cat([
            torch.zeros(1, dtype=torch.float64),
            torch.cumulative_trapezoid(weight, t),
        ])
        cumulative = cumulative / cumulative[-1]

        targets = torch.linspace(0.0, 1.0, n_points + 1, dtype=torch.float64)[:-1]
        angles = _interp_increasing(targets, cumulative, t)

        radii = radial(angles.to(torch.float32)).to(torch.float64)
        points = torch.stack([radii * torch.cos(angles), radii * torch.sin(angles)], dim=1)

        if self.rotation != 0.0:
            c, s = torch.cos(torch.tensor(self.rotation, dtype=torch.float64)), torch.sin(
                torch.tensor(self.rotation, dtype=torch.float64))
            R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
            points = points @ R.T

        self.data = points.to(torch.float32)
        self.labels = torch.full((n_points,), label, dtype=torch.long)
        self.metadata.setdefault("curve", {}).update({
            "perimeter": self.perimeter,
            "rotation": self.rotation,
            "arclength_uniform": angular_density is None,
        })

    def radius(self, theta):
        """World-frame radius: the curve is ``radial`` turned by ``rotation``."""
        return self.radial(theta - self.rotation)


class EllipseRegion2D(Object):
    """
    Filled ellipse with semiaxes ``(a, b)`` turned by ``angle`` — the interior
    partner of ``RadialCurve2D`` in experiment 5.

    Uniform in AREA by default (a linear map sends a uniform disk to a uniform
    ellipse). ``angular_density`` modulates the areal density by angle while
    leaving the support exactly the ellipse — the region-side counterpart of the
    curve's density knob. The modulation is applied in the round pre-image
    frame, so for a circular region (a == b) it is exactly a density in the
    plane's own angle.
    """

    def __init__(
        self,
        n_points: int,
        a: float = 1.0,
        b: float | None = None,
        angle: float = 0.0,
        angular_density=None,
        label: int = 1,
        seed: int | None = None,
        n_quadrature: int = 8192,
        name: str | None = None,
        metadata: dict | None = None,
    ):
        super().__init__(2, name=name, metadata=metadata)

        assert n_points > 0, "n_points must be positive"
        assert a > 0, "a must be positive"
        b = a if b is None else b
        assert b > 0, "b must be positive"

        self.n_points = n_points
        self.a, self.b, self.angle = float(a), float(b), float(angle)

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        u = torch.rand(n_points, generator=generator, dtype=torch.float64)
        v = torch.rand(n_points, generator=generator, dtype=torch.float64)

        if angular_density is None:
            angles = 2 * torch.pi * v
        else:
            t = torch.linspace(0.0, 2 * torch.pi, n_quadrature + 1, dtype=torch.float64)
            density = angular_density(t).to(torch.float64)
            assert torch.all(density > 0), "angular_density(theta) must be strictly positive"
            cumulative = torch.cat([
                torch.zeros(1, dtype=torch.float64),
                torch.cumulative_trapezoid(density, t),
            ])
            cumulative = cumulative / cumulative[-1]
            angles = _interp_increasing(v, cumulative, t)

        radii = torch.sqrt(u)  # uniform in area
        points = torch.stack([
            self.a * radii * torch.cos(angles),
            self.b * radii * torch.sin(angles),
        ], dim=1)

        if self.angle != 0.0:
            c, s = torch.cos(torch.tensor(self.angle, dtype=torch.float64)), torch.sin(
                torch.tensor(self.angle, dtype=torch.float64))
            R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
            points = points @ R.T

        self.data = points.to(torch.float32)
        self.labels = torch.full((n_points,), label, dtype=torch.long)
        self.metadata.setdefault("region", {}).update({
            "a": self.a, "b": self.b, "angle": self.angle,
            "area": float(torch.pi * self.a * self.b),
        })

    def radius(self, theta):
        """World-frame boundary radius of the ellipse."""
        t = theta - self.angle
        return self.a * self.b / torch.sqrt(
            (self.b * torch.cos(t)) ** 2 + (self.a * torch.sin(t)) ** 2
        )


def _interp_increasing(targets, xs, ys):
    """Linear interpolation of ys(xs) at `targets`; xs must be increasing."""
    idx = torch.searchsorted(xs, targets.contiguous()).clamp(1, len(xs) - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = ys[idx - 1], ys[idx]
    w = torch.where(x1 > x0, (targets - x0) / (x1 - x0), torch.zeros_like(targets))
    return y0 + w * (y1 - y0)



def embed_in_ambient(
    data: Data,
    ambient_dim: int,
    mode: str = "linear_isometry",
    seed: int = 0,
) -> Data:
    """
    Embeds a low-dimensional Data into R^{ambient_dim} (experiments 10-11).
    ``Object.apply`` is square-only by design, so embedding is a separate
    operation that produces a NEW Data of dim=ambient_dim (PointCloudObject
    components; features transformed alongside, the frame recorded in
    ``metadata["embedding"]`` of the new Data and of every component).

    mode:
        "linear_isometry" — random orthonormal frame Q [ambient_dim, dim]
            (columns orthonormal, seeded), x -> x @ Q^T. Distances and hence
            topology are exactly preserved; the clean baseline.
        "nonlinear" — smooth non-flat embedding (to be specified) —
            the case where the exp-11 caveat bites: verify preserved homology
            via analysis.homological.persistence_of_point_cloud before/after.

    Alias features (object_names) stay aliases to the embedded components;
    explicit feature point clouds are mapped through the same frame.
    """
    assert ambient_dim >= data.dim, (
        f"ambient_dim={ambient_dim} must be >= data.dim={data.dim}"
    )

    if mode == "nonlinear":
        raise NotImplementedError(
            "Experiment-11 scaffolding: nonlinear embeddings are not "
            "specified yet (use mode='linear_isometry')."
        )

    if mode != "linear_isometry":
        raise ValueError(f"Unknown embedding mode: {mode!r}")

    generator = torch.Generator()
    generator.manual_seed(seed)

    # Random orthonormal frame: QR of a Gaussian matrix, signs fixed so the
    # distribution is Haar (same convention as analysis.controls).
    M = torch.randn(ambient_dim, data.dim, generator=generator)
    Q, R = torch.linalg.qr(M)
    Q = Q * torch.sign(torch.diagonal(R))

    embedding_info = {
        "mode": mode,
        "source_dim": data.dim,
        "ambient_dim": ambient_dim,
        "seed": seed,
        "frame": Q.detach().cpu(),
    }

    def _embed_points(points: torch.Tensor) -> torch.Tensor:
        return points.to(Q.dtype) @ Q.T

    embedded = Data(dim=ambient_dim)
    embedded.metadata["embedding"] = dict(embedding_info)

    for obj in data.objects:
        component = PointCloudObject(
            data=_embed_points(obj.get_data()),
            labels=obj.get_labels(),
            name=obj.name,
            metadata={**(obj.metadata or {}), "embedding": dict(embedding_info)},
        )
        embedded.add(component, name=obj.name)

    for feature in data.features.values():
        embedded.add_feature(
            name=feature["name"],
            points=(
                None
                if feature["points"] is None
                else _embed_points(feature["points"])
            ),
            object_names=(
                None
                if feature["object_names"] is None
                else list(feature["object_names"])
            ),
            metadata=dict(feature["metadata"]),
        )

    return embedded


def with_gaussian_noise(
    data: Data,
    std: float,
    seed: int = 0,
) -> Data:
    """
    [SCAFFOLD] Returns a NEW Data with iid Gaussian jitter added to every
    component (experiment 14: tubular thickening — a 1D feature acquires a
    full-dimensional representation; noise params recorded in metadata).
    Feature point clouds are jittered consistently with their source objects
    (alias features stay aliases; explicit clouds get the same offsets).
    """
    raise NotImplementedError("Experiment-14 scaffolding.")
