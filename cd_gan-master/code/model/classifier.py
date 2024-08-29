import torch.nn as nn


class FcClassifier(nn.Module):
    def __init__(self, layers):
        super(FcClassifier, self).__init__()
        self.models = []
        if len(layers) < 2:
            raise ValueError("Layers should be at least 2")

        for index in range(1, len(layers)):
            self.models.append(
                nn.Linear(layers[index - 1], layers[index], bias=True))
        self.models.append(nn.Softmax(dim=1))
        self.models = nn.ModuleList(self.models)

    def forward(self, x):
        out = x
        for layer in self.models:
            out = layer(out)
        return out


class Classifiers(object):
    def __init__(self, classifier_a, classifier_b, classifier_b_ground_truth):
        self.classifier_b = classifier_b
        self.classifier_a = classifier_a
        self.classifier_b_ground_truth = classifier_b_ground_truth
