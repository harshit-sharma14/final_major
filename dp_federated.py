import copy
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

from opacus import PrivacyEngine

from model import DiseasePredictionModel
from dataset import load_dataset, create_clients, get_dataloader
from aggregation import fedavg


NUM_CLIENTS = 5
LOCAL_EPOCHS = 5
GLOBAL_ROUNDS = 20
LEARNING_RATE = 0.01

CLIP_NORM = 1.0
NOISE_MULTIPLIER = 1.1
DELTA = 1e-5


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

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_dp_federated() -> Dict[str, object]:
    train_dataset, test_dataset = load_dataset()
    client_subsets = create_clients(train_dataset)
    test_loader = get_dataloader(test_dataset)

    global_model = DiseasePredictionModel()
    criterion = nn.BCEWithLogitsLoss()

    # ✅ Initialize tracking lists
    rounds: List[int] = []
    accuracies: List[float] = []
    losses: List[float] = []
    epsilons: List[float] = []

    for _ in range(GLOBAL_ROUNDS):
        client_state_dicts: List[Dict[str, torch.Tensor]] = []
        client_data_sizes: List[int] = []
        round_epsilons: List[float] = []

        for client_subset in client_subsets:
            local_model = DiseasePredictionModel()
            local_model.load_state_dict(copy.deepcopy(global_model.state_dict()))

            optimizer = optim.SGD(local_model.parameters(), lr=LEARNING_RATE)
            train_loader = get_dataloader(client_subset)

            privacy_engine = PrivacyEngine(accountant="rdp")

            local_model, optimizer, train_loader = privacy_engine.make_private(
                module=local_model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=CLIP_NORM,
            )

            for _ in range(LOCAL_EPOCHS):
                local_model.train()
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = local_model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

            epsilon = privacy_engine.get_epsilon(delta=DELTA)
            round_epsilons.append(epsilon)

            client_state_dicts.append(copy.deepcopy(local_model.state_dict()))
            client_data_sizes.append(len(client_subset))

        aggregated_state = fedavg(client_state_dicts, client_data_sizes)

        clean_state = {
            key.replace("_module.", ""): value
            for key, value in aggregated_state.items()
        }

        global_model.load_state_dict(clean_state)

        # ✅ Evaluate after each round
        metrics = _evaluate(global_model, test_loader)

        round_number = len(rounds) + 1
        rounds.append(round_number)
        accuracies.append(metrics["accuracy"])
        losses.append(metrics["loss"])

        epsilons.append(sum(round_epsilons) / len(round_epsilons))

    return {
        "model_state_dict": global_model.state_dict(),
        "rounds": rounds,
        "accuracies": accuracies,
        "losses": losses,
        "epsilons": epsilons,
    }