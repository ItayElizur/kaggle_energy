from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(4, 64)
        self.fc = nn.Linear(64, 1)

    def __call__(self, *input, **kwargs) -> Any:
        return super().__call__(*input, **kwargs)

    def forward(self, input: Tensor):
        x, (_, _) = self.lstm(input)  # Output of LSTM is of the structure (outs, (h_n, c_n)), we ignore the two latter
        x = x.view(-1, 64)
        x = self.fc(x)

        return x


def custom_loss(inputs, targets):
    return torch.sqrt(torch.sum(torch.pow(torch.log(inputs + 1) - torch.log(targets + 1), 2)) / inputs.shape[0])


if __name__ == '__main__':
    model = Model()
    data = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]])
    tensor = torch.from_numpy(data).view(-1, 1, 4).float()

    loss_function = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    print(model(tensor))

    for epoch in range(1000):
        model.zero_grad()

        output = model(tensor)
        output = output.view(-1)

        loss = custom_loss(output, torch.tensor([10.0, 10.0, 10.0]))
        loss.backward()
        optimizer.step()

    print(model(tensor))
