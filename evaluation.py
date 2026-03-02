from typing import Dict, Optional


def evaluate_models(
    centralized_results: Dict[str, float],
    federated_results: Dict[str, float],
    dp_federated_results: Dict[str, object],
) -> None:
    centralized_accuracy = centralized_results["accuracy"]
    centralized_loss = centralized_results["loss"]

    federated_accuracy = federated_results["accuracy"]
    federated_loss = federated_results["loss"]

    dp_accuracies = dp_federated_results.get("accuracies", [])
    dp_losses = dp_federated_results.get("losses", [])

    dp_accuracy = dp_accuracies[-1] if len(dp_accuracies) > 0 else None
    dp_loss = dp_losses[-1] if len(dp_losses) > 0 else None

    dp_epsilons = dp_federated_results.get("epsilons", [])
    final_epsilon = dp_epsilons[-1] if len(dp_epsilons) > 0 else None
    if isinstance(dp_epsilons, list) and len(dp_epsilons) > 0:
        final_epsilon = dp_epsilons[-1]

    print("========== MODEL COMPARISON ==========")

    print("Centralized Training:")
    print(f"  Training Loss   : {centralized_loss:.6f}")
    print(f"  Accuracy        : {centralized_accuracy:.6f}")
    print("  Final Epsilon   : N/A")

    print("--------------------------------------")

    print("Federated Training:")
    print(f"  Training Loss   : {federated_loss:.6f}")
    print(f"  Accuracy        : {federated_accuracy:.6f}")
    print("  Final Epsilon   : N/A")

    print("--------------------------------------")

    print("Federated Training with DP:")
    print(f"  Training Loss   : {dp_loss:.6f}")
    print(f"  Accuracy        : {dp_accuracy:.6f}")
    if final_epsilon is not None:
        print(f"  Final Epsilon   : {final_epsilon:.6f}")
    else:
        print("  Final Epsilon   : N/A")

    print("======================================")