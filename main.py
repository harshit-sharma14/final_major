import torch

from centralized import train_centralized
from federated import train_federated
from dp_federated import train_dp_federated
from evaluation import evaluate_models
from visualization import plot_accuracy, plot_epsilon, plot_loss


def main() -> None:
    torch.manual_seed(42)

    centralized_results = train_centralized()
    federated_results = train_federated()
    dp_federated_results = train_dp_federated()

    evaluate_models(
        centralized_results=centralized_results,
        federated_results=federated_results,
        dp_federated_results=dp_federated_results,
    )

    # 🔥 Actually call the plots
    rounds = dp_federated_results["rounds"]

    plot_accuracy(rounds, dp_federated_results["accuracies"])
    plot_loss(rounds, dp_federated_results["losses"])
    plot_epsilon(rounds, dp_federated_results["epsilons"])


if __name__ == "__main__":
    main()