from collections import OrderedDict
from typing import List, Dict


def fedavg(
    client_state_dicts: List[Dict[str, object]],
    client_data_sizes: List[int],
) -> Dict[str, object]:
    if len(client_state_dicts) == 0:
        raise ValueError("client_state_dicts must not be empty")

    if len(client_state_dicts) != len(client_data_sizes):
        raise ValueError("Mismatch between number of models and data sizes")

    total_samples = sum(client_data_sizes)
    if total_samples <= 0:
        raise ValueError("Total number of samples must be positive")

    aggregated_state = OrderedDict()

    for key in client_state_dicts[0].keys():
        weighted_sum = None

        for state_dict, data_size in zip(client_state_dicts, client_data_sizes):
            weight = data_size / total_samples
            param = state_dict[key].detach().clone()

            if weighted_sum is None:
                weighted_sum = param * weight
            else:
                weighted_sum += param * weight

        aggregated_state[key] = weighted_sum

    return aggregated_state