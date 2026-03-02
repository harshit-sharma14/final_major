import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict

from model import DiseasePredictionModel
from dataset import load_dataset, get_dataloader


EPOCHS = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 32


def _evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)

            predictions = torch.sigmoid(outputs) >= 0.5
            total_correct += (predictions.float() == targets).sum().item()
            total_samples += targets.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return {"loss": average_loss, "accuracy": accuracy}


def train_centralized() -> Dict[str, object]:
    train_dataset, test_dataset = load_dataset()

    train_loader = get_dataloader(train_dataset)
    test_loader = get_dataloader(test_dataset)

    model = DiseasePredictionModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for _ in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    metrics = _evaluate(model, test_loader)

    return {
        "model_state_dict": model.state_dict(),
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
    }