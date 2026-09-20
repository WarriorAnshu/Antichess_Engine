
import torch
import torch.nn as nn
from torch import Tensor

class ResidualBlock(nn.Module):
    """
    A standard residual block with a skip connection to prevent the
    vanishing gradient problem in deep neural networks.
    """
    def __init__(self, channels: int = 256):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class AntichessNet(nn.Module):
    """
    The AlphaZero-style neural network architecture for evaluating Antichess positions.
    Consists of a Shared Core, a Policy Head (move probabilities), and a Value Head (win probability).
    """
    def __init__(self):
        super(AntichessNet, self).__init__()
        self.initial_conv = nn.Conv2d(12, 256, kernel_size=3, stride=1, padding=1)
        self.initial_bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )

        self.policy_conv = nn.Conv2d(256, 66, kernel_size=3, stride=1, padding=1)
        self.policy_bn = nn.BatchNorm2d(66)

        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_linear = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.initial_conv(x)
        x = self.initial_bn(x)
        x = self.relu(x)
        x = self.res_blocks(x)

        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = self.relu(p)
        p = torch.flatten(p, start_dim=1)
        p = p[:, :4216]

        v = self.value_conv(x)
        v = self.value_bn(v)
        v = self.relu(v)
        v = torch.flatten(v, start_dim=1)
        v = self.value_linear(v)
        v = self.sigmoid(v)

        return v, p
