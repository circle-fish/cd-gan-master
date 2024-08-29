import torch.nn as nn


class FcDiscriminator(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(FcDiscriminator, self).__init__()
        self.layers = []

        prev_dim = input_size
        for idx, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim

        self.layers.append(nn.Linear(prev_dim, output_size))
        self.layers.append(nn.Sigmoid())

        self.layer_module = nn.ModuleList(self.layers)

    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out.view(-1, 1)
