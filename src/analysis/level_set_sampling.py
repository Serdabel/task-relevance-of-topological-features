"""Level-set sampling strategies: approximating {x : f(x) ~ c} as a point set.

The grid sampler backs the 2D/3D experiments; the gradient-based samplers are
the d >= 3 route (a dense grid is exponentially dead there, see PLAN.md
§Sampling Methods). Samplers integrate with the criteria through
``analysis.criterions.structural_criterion_sampled`` (per-level sampled clouds,
components via ``analysis.geometry.connected_components_pointcloud``), and any
sampled cloud can be scored directly by passing a gridless
``analysis.fields.FieldSample`` to ``structural_criterion``/``tubular_criterion``.

Coverage caveat for the optimization samplers: gradient descent is pulled to
the nearest sheet of the level surface, so many restarts and diverse inits are
required; report coverage — under-sampling masquerades as "witness not found".
"""

from abc import ABC, abstractmethod
import torch

from analysis.geometry import make_grid, evaluate_scalar_function_on_points


class BaseLevelSetSampler(ABC):
    def __init__(self, function, device="cpu"):
        self.function = function
        self.device = device

    @abstractmethod
    def sample(self, level: float, n_points: int, **kwargs) -> torch.Tensor:
        pass


class GridLevelSetSampler(BaseLevelSetSampler):
    def sample(
        self,
        level: float,
        n_points: int,
        bounds,
        resolution: int = 50,
        eps: float = 0.05,
        batch_size: int = 8192,
        **kwargs,
    ):
        '''
        :param level:
        :param n_points:
        :param bounds: нижние и верхние границы интервалов [(a, b), (c, d), (e, f)]
        :param resolution: сколько точек берётся по каждой оси, то есть всего будет n_bounds^k точек
        :param eps: какая погрешность допустима для понятия "лежать на линии уровня"
        :param batch_size: сколько за раз пропускать точек через нейронку
        :param kwargs:
        :return:
        '''
        grid, _ = make_grid(
            bounds=bounds,
            resolution=resolution,
            device=self.device,
        )

        values = evaluate_scalar_function_on_points(
            function=self.function,
            points=grid,
            batch_size=batch_size,
        )

        mask = (values - level).abs() < eps

        points = grid[mask]

        if len(points) > n_points:
            idx = torch.randperm(len(points), device=self.device)[:n_points]
            points = points[idx]

        return points

class OptimizationLevelSetSampler(BaseLevelSetSampler):
    def sample(
        self,
        level: float,
        n_points: int,
        bounds,
        n_steps: int = 500,
        lr: float = 1e-2,
        eps: float = 1e-2,
        **kwargs,
    ):
        dim = len(bounds)

        lows = torch.tensor([b[0] for b in bounds], device=self.device)
        highs = torch.tensor([b[1] for b in bounds], device=self.device)

        # init
        x = lows + (highs - lows) * torch.rand(
            n_points,
            dim,
            device=self.device,
        )

        x.requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)
        target = torch.tensor(level, device=self.device)

        # optimization
        for _ in range(n_steps):
            values = self.function(x)

            loss = ((values - target) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x.clamp_(lows, highs)

        # 🔥 фильтрация
        with torch.no_grad():
            values = self.function(x)
            mask = (values - level).abs() < eps
            points = x[mask]

        return points

class DataConditionedLevelSetSampler(BaseLevelSetSampler):
    def sample(
        self,
        level: float,
        n_points: int,
        data: torch.Tensor,
        n_steps: int = 300,
        lr: float = 1e-2,
        lambda_anchor: float = 1.0,
        noise_std: float = 0.01,
        **kwargs,
    ):
        data = data.to(self.device)

        idx = torch.randint(
            low=0,
            high=len(data),
            size=(n_points,),
            device=self.device,
        )

        x0 = data[idx].detach()
        x = x0 + noise_std * torch.randn_like(x0)

        x.requires_grad_(True)

        optimizer = torch.optim.Adam([x], lr=lr)
        target = torch.tensor(level, device=self.device)

        for _ in range(n_steps):
            values = self.function(x)

            level_loss = ((values - target) ** 2).mean()
            anchor_loss = ((x - x0) ** 2).sum(dim=1).mean()

            loss = level_loss + lambda_anchor * anchor_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return x.detach()

class TangentWalkLevelSetSampler(BaseLevelSetSampler):
    def sample(self, level: float, n_points: int, **kwargs):
        raise NotImplementedError("Tangent walk sampler will be implemented later")


class MCMCLevelSetSampler(BaseLevelSetSampler):
    def sample(self, level: float, n_points: int, **kwargs):
        raise NotImplementedError("MCMC sampler will be implemented later")

class LevelSetSampler:
    def __init__(
        self,
        function,
        method: str = "optimization",
        device: str = "cpu",
    ):
        self.function = function
        self.method = method
        self.device = device

        samplers = {
            "grid": GridLevelSetSampler,
            "optimization": OptimizationLevelSetSampler,
            "data_conditioned": DataConditionedLevelSetSampler,
            "tangent_walk": TangentWalkLevelSetSampler,
            "mcmc": MCMCLevelSetSampler,
        }

        if method not in samplers:
            raise ValueError(f"Unknown level set sampling method: {method}")

        self.sampler = samplers[method](
            function=function,
            device=device,
        )

    def sample(self, level: float, n_points: int, **kwargs):
        return self.sampler.sample(
            level=level,
            n_points=n_points,
            **kwargs,
        )

