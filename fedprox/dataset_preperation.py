"""Functions for dataset download and processing."""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10



def _download_data() -> Tuple[Dataset, Dataset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Normalization values for CIFAR-10 (approximate)
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


# pylint: disable=too-many-locals

def _partition_data(num_clients, iid=False, power_law=True, balance=False, seed=42):
    """
    Split the training set into iid or non-iid partitions.
    """
    # Download trainset and testset
    trainset, testset = _download_data()

    if balance:
        trainset = _balance_classes(trainset, seed)

    partition_size = int(len(trainset) / num_clients)
    lengths = [partition_size] * num_clients

    if iid:
        from torch.utils.data import random_split
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    else:
        if power_law:
            trainset_sorted = _sort_by_class(trainset)
            datasets = _power_law_split(
                trainset_sorted,
                num_partitions=num_clients,
                num_labels_per_partition=2,
                min_data_per_partition=10,
                mean=0.0,
                sigma=2.0,
            )
        else:
            # Your alternative non-iid partitioning code here.
            raise NotImplementedError("Non-power-law non-iid partitioning not implemented.")

    return datasets, testset

def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled




def _sort_by_class(trainset):
    """
    Sort the dataset by label.
    
    This function converts trainset.targets to a NumPy array, sorts the indices,
    and then creates subsets for each contiguous block of labels.
    Finally, it creates a ConcatDataset and attaches a 'targets' attribute that
    is the concatenation of the targets for each subset.
    """
    # Convert targets (a list) to a NumPy array
    targets_array = np.array(trainset.targets)
    # Get counts for each class
    class_counts = np.bincount(targets_array)
    # Get sorted indices based on target values
    sorted_idxs = targets_array.argsort()

    subsets = []
    subsets_targets = []
    start = 0
    # Iterate over the classes using the cumulative counts
    for count in np.cumsum(class_counts):
        # Get indices for this class block
        indices = sorted_idxs[start:int(count)]
        subsets.append(Subset(trainset, indices))
        subsets_targets.append(targets_array[indices])
        start = int(count)

    # Create a ConcatDataset from the sorted subsets
    sorted_dataset = ConcatDataset(subsets)
    # Attach a 'targets' attribute that is the concatenation of the targets for each subset
    sorted_dataset.targets = np.concatenate(subsets_targets)
    return sorted_dataset


def _power_law_split(
    sorted_trainset, 
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
):
    """
    Partition the dataset following a power-law distribution.
    This follows the implementation similar to Li et al. (2020).

    Parameters:
        sorted_trainset: Dataset
            A dataset sorted by label. Expected to have an attribute 'targets' that is a NumPy array.
        num_partitions: int
            The number of partitions (clients).
        num_labels_per_partition: int, optional
            Number of labels (classes) per partition. (default: 2)
        min_data_per_partition: int, optional
            Minimum number of samples per partition (default: 10)
        mean: float, optional
            Mean for the LogNormal distribution (default: 0.0)
        sigma: float, optional
            Sigma for the LogNormal distribution (default: 2.0)

    Returns:
        A list of Subset datasets, one for each partition.
    """
    # Ensure sorted_trainset has a 'targets' attribute
    if hasattr(sorted_trainset, 'targets'):
        targets = sorted_trainset.targets
    else:
        raise ValueError("The provided sorted_trainset must have a 'targets' attribute.")

    full_idx = list(range(len(targets)))
    class_counts = np.bincount(targets)
    # Compute cumulative sum for labels
    labels_cumsum = np.cumsum(class_counts)
    labels_cumsum = [0] + labels_cumsum[:-1].tolist()

    partitions_idx = []  # List of lists of indices for each partition
    num_classes = len(class_counts)
    hist = np.zeros(num_classes, dtype=np.int32)

    # Assign the minimum number of samples per partition per class
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            start_idx = labels_cumsum[cls] + hist[cls]
            end_idx = start_idx + min_data_per_class
            indices = list(full_idx[start_idx:end_idx])
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # Distribute remaining samples using a LogNormal distribution
    probs = np.random.lognormal(mean, sigma, (num_classes, int(num_partitions / num_classes), num_labels_per_partition))
    remaining_per_class = class_counts - hist
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(probs[cls, u_id // num_classes, cls_idx])
            start_idx = labels_cumsum[cls] + hist[cls]
            end_idx = start_idx + count
            indices = full_idx[start_idx:end_idx]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]
    return partitions

