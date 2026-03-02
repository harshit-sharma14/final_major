
import torch
import torch.nn as nn


class DiseasePredictionModel(nn.Module):
    def __init__(self):
        super(DiseasePredictionModel, self).__init__()

        self.layer1 = nn.Linear(30, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)

        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.zeros_(self.layer2.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.output_layer(x)
        return x