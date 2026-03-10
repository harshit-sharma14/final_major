# import torch

# from centralized import train_centralized
# from federated import train_federated
# from dp_federated import train_dp_federated
# from evaluation import evaluate_models
# from visualization import plot_accuracy, plot_epsilon, plot_loss


# def main() -> None:
#     torch.manual_seed(42)

#     centralized_results = train_centralized()
#     federated_results = train_federated()
#     dp_federated_results = train_dp_federated()

#     evaluate_models(
#         centralized_results=centralized_results,
#         federated_results=federated_results,
#         dp_federated_results=dp_federated_results,
#     )

#     # 🔥 Actually call the plots
#     rounds = dp_federated_results["rounds"]

#     plot_accuracy(rounds, dp_federated_results["accuracies"])
#     plot_loss(rounds, dp_federated_results["losses"])
#     plot_epsilon(rounds, dp_federated_results["epsilons"])


# if __name__ == "__main__":
#     main()
import csv
import torch

from centralized import train_centralized
from federated import train_federated
from dp_federated import train_dp_federated
from evaluation import evaluate_models
from visualization import (
    plot_accuracy,
    plot_epsilon,
    plot_loss,
    plot_privacy_utility,
    plot_model_comparison
)


def main() -> None:
    torch.manual_seed(42)

    # -------------------------------
    # Train baseline models
    # -------------------------------
    centralized_results = train_centralized()
    federated_results = train_federated()

    # -------------------------------
    # DP Noise Multiplier Study
    # -------------------------------
    noise_values = [0.5, 0.8, 1.0, 1.1, 1.3, 1.5]

    dp_experiment_results = []
    all_dp_results = {}

    for noise in noise_values:
        print(f"\nRunning DP-FL with noise multiplier = {noise}")

        dp_results = train_dp_federated(noise_multiplier=noise)
        all_dp_results[noise] = dp_results

        dp_experiment_results.append(
            {
                "noise": noise,
                "accuracy": dp_results["accuracies"][-1],
                "loss": dp_results["losses"][-1],
                "epsilon": dp_results["epsilons"][-1],
            }
        )

    # -------------------------------
    # Print Noise Study Results
    # -------------------------------
    print("\n========== NOISE MULTIPLIER STUDY ==========")

    for result in dp_experiment_results:
        print(
            f"Noise: {result['noise']} | "
            f"Accuracy: {result['accuracy']:.6f} | "
            f"Loss: {result['loss']:.6f} | "
            f"Epsilon: {result['epsilon']:.6f}"
        )

    # -------------------------------
    # Evaluate reference noise model
    # -------------------------------
    reference_noise = 1.1

    if reference_noise not in all_dp_results:
        raise ValueError(f"Reference noise {reference_noise} not found.")

    dp_reference = all_dp_results[reference_noise]

    evaluate_models(
        centralized_results=centralized_results,
        federated_results=federated_results,
        dp_federated_results=dp_reference,
    )

    # -------------------------------
    # Privacy–Utility Tradeoff Plot
    # -------------------------------
    epsilons = [r["epsilon"] for r in dp_experiment_results]
    accuracies = [r["accuracy"] for r in dp_experiment_results]

    plot_privacy_utility(epsilons, accuracies)

    # -------------------------------
    # Training Curve Plots
    # -------------------------------
    rounds = dp_reference["rounds"]

    plot_accuracy(rounds, dp_reference["accuracies"])
    plot_loss(rounds, dp_reference["losses"])
    plot_epsilon(rounds, dp_reference["epsilons"])
    plot_model_comparison(
    centralized_results["accuracy"],
    federated_results["accuracy"],
    dp_reference["accuracies"][-1]
    )

    # -------------------------------
    # Save Results to CSV
    # -------------------------------
    with open("noise_tradeoff_results.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Noise Multiplier", "Final Accuracy", "Final Loss", "Final Epsilon"]
        )

        for result in dp_experiment_results:
            writer.writerow(
                [
                    result["noise"],
                    result["accuracy"],
                    result["loss"],
                    result["epsilon"],
                ]
            )


if __name__ == "__main__":
    main()