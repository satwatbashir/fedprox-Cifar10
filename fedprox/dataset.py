"""CIFAR-10 dataset utilities for federated learning with feature skew.

This file partitions the dataset, splits each client’s data into training
and validation, and optionally applies client-specific feature skew transforms.
"""

from typing import Optional, Tuple

import random
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split

from fedprox.dataset_preperation import _partition_data


class SkewedDataset(Dataset):
    """
    A wrapper for a dataset that applies a client-specific feature skew transform.
    """

    def __init__(self, subset: Dataset, skew_transform: Optional[transforms.Compose]) -> None:
        self.subset = subset
        self.skew_transform = skew_transform

    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.skew_transform is not None:
            image = self.skew_transform(image)
        return image, label

    def __len__(self):
        return len(self.subset)


def get_client_skew_transform(client_id: int, skew_level: float) -> transforms.Compose:
    """
    Generate a torchvision transform to simulate feature skew for a given client.

    The transformation uses ColorJitter and RandomRotation as examples.
    A higher skew_level produces more pronounced transformations.

    Parameters
    ----------
    client_id : int
        The unique client identifier (used to seed randomness).
    skew_level : float
        A float in [0, 1] where 0 means no skew and 1 means high skew.

    Returns
    -------
    transforms.Compose
        The composed transformation.
    """
    # Seed randomness for reproducibility per client.
    random.seed(client_id)
    brightness = 0.5 * skew_level + random.uniform(0, 0.5 * skew_level)
    contrast = 0.5 * skew_level + random.uniform(0, 0.5 * skew_level)
    rotation = int(10 * skew_level)  # maximum rotation in degrees

    skew_transform = transforms.Compose([
        transforms.ColorJitter(brightness=brightness, contrast=contrast),
        transforms.RandomRotation(degrees=rotation)
    ])
    return skew_transform


def load_datasets(
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create the dataloaders to be fed into the model.

    The function partitions the dataset for the given number of clients and,
    if enabled, wraps each client's partition with a feature skew transformation.

    Parameters
    ----------
    config : DictConfig
        Configuration object that parameterizes the dataset partitioning.
        Expected to contain keys: iid, balance, power_law, and optionally feature_skew.
    num_clients : int
        The number of clients that hold a part of the data.
    val_ratio : float, optional
        The ratio of training data to use for validation (default: 0.1).
    batch_size : int, optional
        The batch size for DataLoaders (default: 32).
    seed : int, optional
        Seed for reproducibility (default: 42).

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, and the
        DataLoader for testing.
    """
    print(f"Dataset partitioning config: {config}")
    datasets, testset = _partition_data(
        num_clients,
        iid=config.iid,
        balance=config.balance,
        power_law=config.power_law,
        seed=seed,
    )

    # Retrieve feature_skew level from the config; default is 0 (no skew)
    feature_skew = config.get("feature_skew", 0.0)

    trainloaders = []
    valloaders = []
    for client_id, dataset in enumerate(datasets):
        # If feature skew is desired (feature_skew > 0), wrap the dataset.
        if feature_skew > 0:
            skew_transform = get_client_skew_transform(client_id, feature_skew)
            dataset = SkewedDataset(dataset, skew_transform)
        
        # Split each client's partition into training and validation.
        len_val = int(len(dataset) * val_ratio)
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
        
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader
