import torch.nn as nn
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(100, 80),
            nn.Sigmoid(),
            nn.Linear(80, 100),
            nn.Sigmoid(),
            nn.Linear(100, 80),
            nn.Sigmoid(),
            nn.Linear(80, 100),
        )


# %%n
    def forward(self, x):
        # x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output

class test(nn.Module):
    def __init__(self):
        super(test, self).__init__()
        self.linear = nn.Linear(1, 1)
# %%n
    def forward(self, x):
        output = self.linear(x)
        return output
