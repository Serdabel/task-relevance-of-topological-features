from pathlib import Path
import warnings

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from data.data_generation import Data, PointCloudObject
from models.mlp import MLP


def _to_cpu_tensor(x):
    if x is None:
        return None

    if torch.is_tensor(x):
        return x.detach().cpu()

    return torch.as_tensor(x)


def _single_label_or_none(y: torch.Tensor):
    unique = torch.unique(y)

    if len(unique) == 1:
        return int(unique.item())

    return None


def _extract_dataset_structure(data: Data):
    """
    Extracts named components and named analysis features from Data.
    """

    components = {}
    component_id_parts = []

    object_names = getattr(
        data,
        "object_names",
        [f"object_{i}" for i in range(len(data.objects))],
    )

    offset = 0

    for component_id, (name, obj) in enumerate(zip(object_names, data.objects)):
        X_i = obj.get_data().detach().cpu()
        y_i = obj.get_labels().detach().cpu()

        n_i = len(X_i)
        indices = torch.arange(offset, offset + n_i, dtype=torch.long)

        metadata = dict(getattr(obj, "metadata", {}))

        components[name] = {
            "id": component_id,
            "name": name,
            "X": X_i,
            "y": y_i,
            "indices": indices,
            "label": _single_label_or_none(y_i),
            "metadata": metadata,
        }

        component_id_parts.append(
            torch.full((n_i,), component_id, dtype=torch.long)
        )

        offset += n_i

    if len(component_id_parts) > 0:
        component_id = torch.cat(component_id_parts, dim=0)
    else:
        component_id = None

    features = {}

    if hasattr(data, "features"):
        for feature_name, feature in data.features.items():
            points = data.get_feature_points(feature_name).detach().cpu()

            features[feature_name] = {
                "name": feature_name,
                "points": points,
                "object_names": feature.get("object_names", None),
                "metadata": dict(feature.get("metadata", {})),
            }

    return {
        "components": components,
        "features": features,
        "component_id": component_id,
    }


def reconstruct_data_from_structure(structure: dict):
    """
    Reconstructs a Data object from saved component point clouds.

    This does not reconstruct the original sampling recipe.
    It reconstructs the exact sampled point clouds, which is what we need
    for reproducible experiments.
    """
    components = structure.get("components", {})
    features = structure.get("features", {})

    assert len(components) > 0, "No components in structure"

    first_component = next(iter(components.values()))
    dim = first_component["X"].shape[1]

    data = Data(dim)

    sorted_components = sorted(
        components.items(),
        key=lambda kv: kv[1]["id"],
    )

    for name, component in sorted_components:
        obj = PointCloudObject(
            data=component["X"],
            labels=component["y"],
            name=name,
            metadata=component.get("metadata", {}),
        )

        data.add(
            obj,
            name=name,
            is_feature=False,
        )

    for feature_name, feature in features.items():
        data.add_feature(
            name=feature_name,
            points=feature["points"],
            object_names=feature.get("object_names", None),
            metadata=feature.get("metadata", {}),
        )

    return data


def get_feature_points(loaded_dataset: dict, feature_name: str):
    features = loaded_dataset.get("features", {})

    assert feature_name in features, (
        f"Unknown feature '{feature_name}'. "
        f"Available features: {list(features.keys())}"
    )

    return features[feature_name]["points"]


def get_component_points(loaded_dataset: dict, component_name: str):
    components = loaded_dataset.get("components", {})

    assert component_name in components, (
        f"Unknown component '{component_name}'. "
        f"Available components: {list(components.keys())}"
    )

    return components[component_name]["X"]



def store_dataset_instance(
    name,
    id=None,
    data: Data = None,
    X=None,
    y=None,
    train_ratio=0.8,
    split_seed=None,
    extra_metadata: dict | None = None,
    stratify_by: str | None = None,
):
    """
    Stores a fixed dataset instance in:

        data/dataset_instances/{name}/{name}_{id}/

    Saved files:

        data.pt
            {
                "X": X,
                "y": y,
                "component_id": component_id or None,
            }

        split.pt
            {
                "train_idx": train_idx,
                "test_idx": test_idx,
            }

        metadata.pt
            {
                "name": name,
                "id": final_id,
                "train_ratio": train_ratio,
                "split_seed": split_seed,
                "format_version": 2,
                ...
            }

        structure.pt
            {
                "components": {
                    component_name: {
                        "id": int,
                        "name": str,
                        "X": Tensor,
                        "y": Tensor,
                        "indices": LongTensor,
                        "label": int | None,
                        "metadata": dict,
                    },
                    ...
                },
                "features": {
                    feature_name: {
                        "name": str,
                        "points": Tensor,
                        "object_names": list[str] | None,
                        "metadata": dict,
                    },
                    ...
                },
            }

    Returns:
        str:
            "/data/dataset_instances/name/name_id/"
    """

    assert isinstance(name, str) and len(name) > 0, "name must be a non-empty string"

    assert data is not None or (X is not None and y is not None), (
        "Either data or both X and y must be provided"
    )

    if id is not None:
        assert isinstance(id, int) and id >= 0, "id must be a non-negative int"

    assert 0.0 < train_ratio < 1.0, "train_ratio must be between 0 and 1"

    if split_seed is not None:
        assert isinstance(split_seed, int), "split_seed must be an int or None"

    structure = {
        "components": {},
        "features": {},
    }
    component_id = None

    if data is not None:
        X = data.get_data()
        y = data.get_labels()

        structure_info = _extract_dataset_structure(data)

        structure = {
            "components": structure_info["components"],
            "features": structure_info["features"],
        }

        component_id = structure_info["component_id"]

    X = _to_cpu_tensor(X)
    y = _to_cpu_tensor(y).long()

    if component_id is not None:
        component_id = component_id.detach().cpu().long()

    assert len(X) == len(y), "X and y must have the same length"

    if component_id is not None:
        assert len(component_id) == len(X), "component_id must have same length as X"

    root = Path("data") / "dataset_instances"
    dataset_dir = root / name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def instance_dir(instance_id: int) -> Path:
        return dataset_dir / f"{name}_{instance_id}"

    def find_first_free_id() -> int:
        current_id = 0
        while instance_dir(current_id).exists():
            current_id += 1
        return current_id

    if id is None:
        final_id = find_first_free_id()
    else:
        if instance_dir(id).exists():
            warnings.warn(
                f"Dataset instance {name}_{id} already exists. "
                f"Using the first free id instead.",
                UserWarning,
            )
            final_id = find_first_free_id()
        else:
            final_id = id

    save_dir = instance_dir(final_id)
    save_dir.mkdir(parents=False, exist_ok=False)

    n = len(X)
    n_train = int(train_ratio * n)

    if split_seed is None:
        generator = None
    else:
        generator = torch.Generator()
        generator.manual_seed(split_seed)

    def _perm(k):
        return torch.randperm(k) if generator is None else torch.randperm(k, generator=generator)

    if stratify_by is None:
        perm = _perm(n)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
    else:
        # Stratified split, one stratum per component.
        #
        # The unstratified draw above is fine while every component has
        # hundreds of points, and silently useless once one of them has a
        # handful: experiment 4's thinning sweep goes down to 7 points on the
        # target circle, where a plain 80/20 draw leaves 1-2 of them in the
        # test set and the "100 % test accuracy" gate stops saying anything
        # about the circle at all. Stratifying keeps each component's share of
        # train and test fixed, so the gate keeps meaning what it says.
        assert stratify_by == "component", (
            f"Unknown stratify_by: {stratify_by!r} (only 'component')"
        )
        assert component_id is not None, (
            "stratify_by='component' needs component_id, i.e. a Data instance"
        )

        train_parts, test_parts = [], []
        for c in component_id.unique(sorted=True).tolist():
            idx = (component_id == c).nonzero(as_tuple=True)[0]
            shuffled = idx[_perm(len(idx))]
            k = int(round(train_ratio * len(idx)))
            if len(idx) >= 2:
                k = min(max(k, 1), len(idx) - 1)  # never starve either side
            train_parts.append(shuffled[:k])
            test_parts.append(shuffled[k:])

        train_idx = torch.cat(train_parts)
        test_idx = torch.cat(test_parts)
        # de-correlate ordering from component id
        train_idx = train_idx[_perm(len(train_idx))]
        test_idx = test_idx[_perm(len(test_idx))]

    torch.save(
        {
            "X": X,
            "y": y,
            "component_id": component_id,
        },
        save_dir / "data.pt",
    )

    torch.save(
        {
            "train_idx": train_idx,
            "test_idx": test_idx,
        },
        save_dir / "split.pt",
    )

    metadata = {
        "name": name,
        "id": final_id,
        "train_ratio": train_ratio,
        "split_seed": split_seed,
        "stratify_by": stratify_by,
        "format_version": 2,
        "has_structure": len(structure["components"]) > 0,
        "feature_names": list(structure["features"].keys()),
        "component_names": list(structure["components"].keys()),
    }

    if extra_metadata is not None:
        metadata.update(dict(extra_metadata))

    torch.save(metadata, save_dir / "metadata.pt")
    torch.save(structure, save_dir / "structure.pt")

    return f"/data/dataset_instances/{name}/{name}_{final_id}/"


def load_dataset_instance(path, map_location="cpu", reconstruct_data=True):
    """
    Loads a fixed dataset instance saved by store_dataset_instance.

    Supports both old format and new structured format.

    Returns:
        dict with keys:
            X
            y
            component_id
            train_idx
            test_idx
            train_dataset
            test_dataset
            metadata
            components
            features
            data
            path
    """

    path = Path(path)

    # If path is "/data/...", convert to "data/..."
    # because this function is called from the project root.
    if path.is_absolute():
        path = Path(str(path).lstrip("/"))

    assert path.exists(), f"Dataset instance path does not exist: {path}"
    assert path.is_dir(), f"Dataset instance path is not a directory: {path}"

    data_path = path / "data.pt"
    split_path = path / "split.pt"
    metadata_path = path / "metadata.pt"
    structure_path = path / "structure.pt"

    assert data_path.exists(), f"Missing file: {data_path}"
    assert split_path.exists(), f"Missing file: {split_path}"
    assert metadata_path.exists(), f"Missing file: {metadata_path}"

    data_obj = torch.load(data_path, map_location=map_location)
    split_obj = torch.load(split_path, map_location=map_location)
    metadata = torch.load(metadata_path, map_location=map_location)

    assert "X" in data_obj, "data.pt must contain key 'X'"
    assert "y" in data_obj, "data.pt must contain key 'y'"
    assert "train_idx" in split_obj, "split.pt must contain key 'train_idx'"
    assert "test_idx" in split_obj, "split.pt must contain key 'test_idx'"

    X = data_obj["X"]
    y = data_obj["y"]

    component_id = data_obj.get("component_id", None)

    train_idx = split_obj["train_idx"]
    test_idx = split_obj["test_idx"]

    train_dataset = TensorDataset(
        X[train_idx],
        y[train_idx],
    )

    test_dataset = TensorDataset(
        X[test_idx],
        y[test_idx],
    )

    if structure_path.exists():
        structure = torch.load(structure_path, map_location=map_location)
    else:
        structure = {
            "components": {},
            "features": {},
        }

    components = structure.get("components", {})
    features = structure.get("features", {})

    reconstructed_data = None

    if reconstruct_data and len(components) > 0:
        reconstructed_data = reconstruct_data_from_structure(structure)

    return {
        "X": X,
        "y": y,
        "component_id": component_id,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "metadata": metadata,
        "components": components,
        "features": features,
        "data": reconstructed_data,
        "path": str(path),
    }


def activation_to_name(activation: nn.Module) -> str:
    if isinstance(activation, nn.Tanh):
        return "tanh"
    if isinstance(activation, nn.ReLU):
        return "relu"
    if isinstance(activation, nn.Sigmoid):
        return "sigmoid"
    if isinstance(activation, nn.LeakyReLU):
        return "leaky_relu"

    raise ValueError(f"Unknown activation type: {type(activation)}")


def extract_mlp_config(model) -> dict:
    """
    Extracts architecture config from your MLP object.
    Assumes model has:
        in_dim
        hidden_dim
        n_layers
        out_dim
        include_output_layer
        blocks
    """

    assert hasattr(model, "in_dim"), "model must have attribute in_dim"
    assert hasattr(model, "hidden_dim"), "model must have attribute hidden_dim"
    assert hasattr(model, "n_layers"), "model must have attribute n_layers"
    assert hasattr(model, "out_dim"), "model must have attribute out_dim"
    assert hasattr(model, "include_output_layer"), "model must have attribute include_output_layer"
    assert hasattr(model, "blocks"), "model must have attribute blocks"

    assert len(model.blocks) > 0, "model.blocks must be non-empty"

    activation_name = activation_to_name(model.blocks[0].activation)

    return {
        "model_type": "mlp",
        "in_dim": model.in_dim,
        "hidden_dim": model.hidden_dim,
        "n_layers": model.n_layers,
        "out_dim": model.out_dim,
        "activation": activation_name,
        "include_output_layer": model.include_output_layer,
    }


def _validate_model_name(model_name: str | None):
    if model_name is None:
        return

    assert isinstance(model_name, str), "model_name must be a string or None"
    assert len(model_name) > 0, "model_name must be non-empty if provided"
    assert "/" not in model_name, "model_name must not contain '/'"
    assert "\\" not in model_name, "model_name must not contain '\\'"
    assert " " not in model_name, "model_name must not contain spaces; use '_' instead"


def store_model_instance(
    dataset_instance_path,
    model,
    model_id=None,
    model_name: str | None = None,
    train_config: dict | None = None,
    metrics: dict | None = None,
    extra: dict | None = None,
):
    """
    Stores a trained MLP model corresponding to a fixed dataset instance.

    Expected dataset path:

        /data/dataset_instances/{data_type}/{instance_name}/

    Saves to:

        models/runs/{data_type}/{instance_name}/{instance_name}_model{model_id}/

    if model_name is None, or to:

        models/runs/{data_type}/{instance_name}/{instance_name}_{model_name}_model{model_id}/

    if model_name is provided.

    Examples:

        /models/runs/linked_circles_3d/linked_circles_3d_0/linked_circles_3d_0_model0/

        /models/runs/linked_circles_3d/linked_circles_3d_0/linked_circles_3d_0_tanh_width8_depth5_model0/

    Saved files:
        checkpoint.pt
        model_config.pt
        train_config.pt
        metrics.pt
        dataset_ref.pt
        extra.pt, if extra is not None

    Returns:
        str path of saved model instance.
    """

    assert model is not None, "model must be provided"

    if model_id is not None:
        assert isinstance(model_id, int) and model_id >= 0, (
            "model_id must be a non-negative int"
        )

    _validate_model_name(model_name)

    dataset_instance_path = Path(dataset_instance_path)

    # If path is given as "/data/...", convert to "data/..."
    # because the function is called from the project root.
    if dataset_instance_path.is_absolute():
        dataset_instance_path = Path(str(dataset_instance_path).lstrip("/"))

    assert dataset_instance_path.exists(), (
        f"Dataset instance path does not exist: {dataset_instance_path}"
    )
    assert dataset_instance_path.is_dir(), (
        f"Dataset instance path is not a directory: {dataset_instance_path}"
    )

    # Expected:
    # data/dataset_instances/{data_type}/{instance_name}
    parts = dataset_instance_path.parts

    assert len(parts) >= 4, (
        "dataset_instance_path must look like "
        "'data/dataset_instances/{data_type}/{instance_name}'"
    )

    assert parts[-3] == "dataset_instances", (
        "dataset_instance_path must be inside 'data/dataset_instances/'"
    )

    data_type = parts[-2]
    instance_name = parts[-1]

    runs_root = Path("models") / "runs"
    instance_run_dir = runs_root / data_type / instance_name
    instance_run_dir.mkdir(parents=True, exist_ok=True)

    def make_model_dir_name(current_model_id: int) -> str:
        if model_name is None:
            return f"{instance_name}_model{current_model_id}"

        return f"{instance_name}_{model_name}_model{current_model_id}"

    def current_model_dir(current_model_id: int) -> Path:
        return instance_run_dir / make_model_dir_name(current_model_id)

    def find_first_free_model_id() -> int:
        current_id = 0
        while current_model_dir(current_id).exists():
            current_id += 1
        return current_id

    if model_id is None:
        final_model_id = find_first_free_model_id()
    else:
        if current_model_dir(model_id).exists():
            warnings.warn(
                f"Model instance {make_model_dir_name(model_id)} already exists. "
                f"Using the first free model_id instead.",
                UserWarning,
            )
            final_model_id = find_first_free_model_id()
        else:
            final_model_id = model_id

    save_dir = current_model_dir(final_model_id)
    save_dir.mkdir(parents=False, exist_ok=False)

    model_config = extract_mlp_config(model)

    dataset_ref = {
        "dataset_instance_path": str(dataset_instance_path),
        "data_type": data_type,
        "instance_name": instance_name,
    }

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
        "train_config": train_config,
        "metrics": metrics,
        "dataset_ref": dataset_ref,
        "model_id": final_model_id,
        "model_name": model_name,
        "model_instance_name": make_model_dir_name(final_model_id),
        "extra": extra,
    }

    torch.save(checkpoint, save_dir / "checkpoint.pt")

    torch.save(model_config, save_dir / "model_config.pt")
    torch.save(train_config, save_dir / "train_config.pt")
    torch.save(metrics, save_dir / "metrics.pt")
    torch.save(dataset_ref, save_dir / "dataset_ref.pt")

    if extra is not None:
        torch.save(extra, save_dir / "extra.pt")

    return f"/models/runs/{data_type}/{instance_name}/{make_model_dir_name(final_model_id)}/"

def name_to_activation(activation_name: str) -> type[nn.Module]:
    if activation_name == "tanh":
        return nn.Tanh
    if activation_name == "relu":
        return nn.ReLU
    if activation_name == "sigmoid":
        return nn.Sigmoid
    if activation_name == "leaky_relu":
        return nn.LeakyReLU

    raise ValueError(f"Unknown activation name: {activation_name}")


def build_mlp_from_config(model_config: dict) -> MLP:
    """
    Reconstructs your MLP from saved model_config.
    """

    assert model_config["model_type"] == "mlp", (
        f"Expected model_type='mlp', got {model_config.get('model_type')}"
    )

    return MLP(
        in_dim=model_config["in_dim"],
        hidden_dim=model_config["hidden_dim"],
        n_layers=model_config["n_layers"],
        out_dim=model_config["out_dim"],
        activation_cls=name_to_activation(model_config["activation"]),
        include_output_layer=model_config["include_output_layer"],
    )

def load_model_instance(model_instance_path, device="cpu", eval_mode=True):
    """
    Loads a trained MLP model saved by store_model_instance.

    Expected path:

        /models/runs/{data_type}/{instance_name}/{instance_name}_model{model_id}/

    Expected file:

        checkpoint.pt

    Parameters
    ----------
    model_instance_path : str | Path
        Path to saved model instance.

    device : str | torch.device
        Device where model should be loaded.

    eval_mode : bool
        If True, calls model.eval() after loading.

    Returns
    -------
    model : MLP
        Reconstructed model with loaded weights.

    checkpoint : dict
        Full saved checkpoint with model_config, train_config, metrics, dataset_ref, etc.
    """

    model_instance_path = Path(model_instance_path)

    # If path is given as "/models/...", convert to "models/..."
    # because the function is called from the project root.
    if model_instance_path.is_absolute():
        model_instance_path = Path(str(model_instance_path).lstrip("/"))

    assert model_instance_path.exists(), (
        f"Model instance path does not exist: {model_instance_path}"
    )
    assert model_instance_path.is_dir(), (
        f"Model instance path is not a directory: {model_instance_path}"
    )

    checkpoint_path = model_instance_path / "checkpoint.pt"

    assert checkpoint_path.exists(), f"Missing checkpoint file: {checkpoint_path}"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    assert "model_config" in checkpoint, "checkpoint.pt must contain key 'model_config'"
    assert "model_state_dict" in checkpoint, (
        "checkpoint.pt must contain key 'model_state_dict'"
    )

    model_config = checkpoint["model_config"]

    model = build_mlp_from_config(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if eval_mode:
        model.eval()

    return model, checkpoint