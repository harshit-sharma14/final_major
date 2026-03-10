# import os
# import matplotlib.pyplot as plt
# from typing import List


# def _save_plot(filename: str) -> None:
#     os.makedirs("plots", exist_ok=True)
#     plt.savefig(os.path.join("plots", filename), dpi=300, bbox_inches="tight")
#     plt.close()


# def plot_accuracy(rounds: List[int], accuracies: List[float]) -> None:
#     plt.figure()
#     plt.plot(rounds, accuracies)
#     plt.xlabel("Global Rounds")
#     plt.ylabel("Accuracy")
#     plt.title("Accuracy vs Global Rounds")
#     _save_plot("accuracy_vs_rounds.png")


# def plot_loss(rounds: List[int], losses: List[float]) -> None:
#     plt.figure()
#     plt.plot(rounds, losses)
#     plt.xlabel("Global Rounds")
#     plt.ylabel("Loss")
#     plt.title("Loss vs Global Rounds")
#     _save_plot("loss_vs_rounds.png")


# def plot_epsilon(rounds: List[int], epsilons: List[float]) -> None:
#     plt.figure()
#     plt.plot(rounds, epsilons)
#     plt.xlabel("Global Rounds")
#     plt.ylabel("Epsilon")
#     plt.title("Epsilon vs Global Rounds")
#     _save_plot("epsilon_vs_rounds.png")
import os
import matplotlib.pyplot as plt
from typing import List


def _save_plot(filename: str) -> None:
    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_accuracy(rounds: List[int], accuracies: List[float]) -> None:
    plt.figure()
    plt.plot(rounds, accuracies)
    plt.xlabel("Global Rounds")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Global Rounds")
    _save_plot("accuracy_vs_rounds.png")


def plot_loss(rounds: List[int], losses: List[float]) -> None:
    plt.figure()
    plt.plot(rounds, losses)
    plt.xlabel("Global Rounds")
    plt.ylabel("Loss")
    plt.title("Loss vs Global Rounds")
    _save_plot("loss_vs_rounds.png")


def plot_epsilon(rounds: List[int], epsilons: List[float]) -> None:
    plt.figure()
    plt.plot(rounds, epsilons)
    plt.xlabel("Global Rounds")
    plt.ylabel("Epsilon")
    plt.title("Epsilon vs Global Rounds")
    _save_plot("epsilon_vs_rounds.png")

def plot_privacy_utility(epsilons, accuracies):
    plt.figure()
    plt.plot(epsilons, accuracies)
    plt.xlabel("Epsilon (Privacy Budget)")
    plt.ylabel("Final Accuracy")
    plt.title("Privacy–Utility Tradeoff")
    _save_plot("privacy_utility_tradeoff.png")
    
def plot_model_comparison(centralized_acc, federated_acc, dp_acc):
    plt.figure()

    models = ["Centralized", "Federated", "DP-Federated"]
    accuracies = [centralized_acc, federated_acc, dp_acc]

    plt.bar(models, accuracies)

    plt.xlabel("Training Approach")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")

    _save_plot("model_accuracy_comparison.png")