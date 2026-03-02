import torch
import numpy as np
from typing import List, Tuple
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, Subset


TEST_SIZE = 0.2
RANDOM_STATE = 42
NUM_CLIENTS = 5
BATCH_SIZE = 32


def load_dataset() -> Tuple[TensorDataset, TensorDataset]:
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    return train_dataset, test_dataset


def create_clients(train_dataset: TensorDataset) -> List[Subset]:
    total_samples = len(train_dataset)
    indices = np.arange(total_samples)

    np.random.seed(RANDOM_STATE)
    np.random.shuffle(indices)

    split_size = total_samples // NUM_CLIENTS
    client_datasets = []

    for i in range(NUM_CLIENTS):
        start = i * split_size
        if i == NUM_CLIENTS - 1:
            end = total_samples
        else:
            end = (i + 1) * split_size

        client_indices = indices[start:end]
        client_subset = Subset(train_dataset, client_indices.tolist())
        client_datasets.append(client_subset)

    return client_datasets


def get_dataloader(dataset: torch.utils.data.Dataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,

    )