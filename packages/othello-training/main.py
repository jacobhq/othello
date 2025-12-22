import torch
import torch.nn as nn


def main():
    print("Hello from othello-training!")


class AlphaZeroNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input conv: 2→16 channels
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        # Residual blocks (example: 16→32→64→128)
        self.res_blocks = nn.Sequential(
            ResBlock(16, 32),  # (out channels = 32)
            ResBlock(32, 64),
            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128),
        )
        # Policy head
        self.p_conv = nn.Conv2d(128, 16, 1)  # 1×1 conv to 16 planes
        self.p_bn = nn.BatchNorm2d(16)
        self.p_fc = nn.Linear(16 * 8 * 8, 64)  # 64 possible moves
        # Value head
        self.v_conv = nn.Conv2d(128, 16, 1)
        self.v_bn = nn.BatchNorm2d(16)
        self.v_fc1 = nn.Linear(16 * 8 * 8, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)
        # Policy branch
        p = nn.ReLU()(self.p_bn(self.p_conv(x)))
        p = p.view(-1, 16 * 8 * 8)
        p = nn.ReLU()(self.p_fc(p))
        p = self.p_fc.weight.new(1).log_softmax(p)  # LogSoftmax or Softmax
        # Value branch
        v = nn.ReLU()(self.v_bn(self.v_conv(x)))
        v = v.view(-1, 16 * 8 * 8)
        v = nn.ReLU()(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))  # Tanh to [-1,1]
        return p, v


if __name__ == "__main__":
    main()
