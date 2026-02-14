"""
Dataset loaders for CIFAR-100 (ID) and OOD datasets
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

CIFAR_100_mean = [0.5071, 0.4867, 0.4408]
CIFAR_100_std = [0.2675, 0.2565, 0.2761]

def get_cifar100_loaders(
    data_dir='./data',
    batch_size=128,
    num_workers=2,
    augment=True,
    val_split=0.1
):
    """
    Get CIFAR-100 train/val/test loaders
    
    Args:
        data_dir: where to download/load data
        batch_size: batch size
        num_workers: dataloader workers
        augment: use data augmentation for training
        val_split: fraction of train set to use for validation
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Normalization values for CIFAR-100
    normalize = transforms.Normalize(
        mean= CIFAR_100_mean,
        std= CIFAR_100_std
    )
    
    # Training transforms (with augmentation)
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    
    # Test transforms (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load datasets
    train_dataset_full = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )
    
    # Split train into train/val
    if val_split > 0:
        num_train = len(train_dataset_full)
        indices = list(range(num_train))
        split = int(np.floor(val_split * num_train))
        
        np.random.seed(42)  # reproducibility
        np.random.shuffle(indices)
        
        train_idx, val_idx = indices[split:], indices[:split]
        
        # Create subsets
        train_dataset = Subset(train_dataset_full, train_idx)
        
        # Validation uses test transform (no augmentation)
        val_dataset_full = datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=test_transform
        )
        val_dataset = Subset(val_dataset_full, val_idx)
    else:
        train_dataset = train_dataset_full
        val_dataset = None
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        val_loader = None
    
    return train_loader, val_loader, test_loader


def get_ood_loaders(
    ood_datasets=['SVHN', 'CIFAR10', 'Textures'], # for now, let's start with just SVHN
    data_dir='./data',
    batch_size=128,
    num_workers=2,
    sampling_ratio = None
):
    """
    Get OOD dataset loaders
    
    Args:
        ood_datasets: list of OOD dataset names
        data_dir: where to download/load data
        batch_size: batch size
        num_workers: dataloader workers
        sampling_ratio: if not None, sample this fraction from each dataset (proportional)
    
    Returns:
        dict of {dataset_name: loader} or single concatenated loader if sampling_ratio is set
    """
    
    # CIFAR-100 normalization (apply to all OOD for consistency)
    normalize = transforms.Normalize(
        mean=CIFAR_100_mean,
        std=CIFAR_100_std
    )

    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Force 32x32 (not just 32)
        transforms.ToTensor(),
        normalize,
    ])
    
    loaders = {}
    
    for dataset_name in ood_datasets:
        if dataset_name == 'SVHN':
            dataset = datasets.SVHN(
                root=data_dir, split='test', download=True, transform=transform
            )
        
        elif dataset_name == 'CIFAR10':
            dataset = datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform
            )
        
        elif dataset_name == 'Textures':
            # DTD (Describable Textures Dataset)
            try:
                dataset = datasets.DTD(
                    root=data_dir, split='test', download=True, transform=transform
                )
            except:
                print(f"Warning: DTD not available, skipping {dataset_name}")
                continue
        
        else:
            print(f"Warning: Unknown dataset {dataset_name}, skipping")
            continue
       
        # Apply sampling if ratio provided
        if sampling_ratio is not None:
            total_size = len(dataset)
            sample_size = int(total_size * sampling_ratio)
            indices = np.random.choice(total_size, size=sample_size, replace=False)
            dataset = Subset(dataset, indices)
            #print(f"{dataset_name}: sampled {sample_size}/{total_size} ({sampling_ratio*100:.1f}%)")
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        loaders[dataset_name] = loader
    
    return loaders


def test_dataloaders():
    """Quick test of dataloaders"""
    print("Testing CIFAR-100 loaders...")
    train_loader, val_loader, test_loader = get_cifar100_loaders(batch_size=4)
    
    # Test train loader
    x, y = next(iter(train_loader))
    assert x.shape == (4, 3, 32, 32), f"Expected (4, 3, 32, 32), got {x.shape}"
    assert y.shape == (4,), f"Expected (4,), got {y.shape}"
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Test OOD loaders
    print("\nTesting OOD loaders...")
    ood_loaders = get_ood_loaders(ood_datasets=['SVHN', 'CIFAR10'], batch_size=4)
    
    for name, loader in ood_loaders.items():
        x, y = next(iter(loader))
        print(f"{name}: {len(loader)} batches, shape {x.shape}")
    
    print("\n✓ All dataloader tests passed!")


if __name__ == "__main__":
    test_dataloaders()
