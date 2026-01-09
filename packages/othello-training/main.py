import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class OthelloDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            magic, version, n = struct.unpack("<III", f.read(12))
            assert magic == 0x4F54484C  # "OTHL"
            assert version == 1

            self.states = np.zeros((n, 2, 8, 8), dtype=np.float32)
            self.policies = np.zeros((n, 64), dtype=np.float32)
            self.values = np.zeros((n,), dtype=np.float32)

            for i in range(n):
                self.states[i] = (
                    np.frombuffer(f.read(2 * 8 * 8 * 4), dtype=np.int32)
                    .reshape(2, 8, 8)
                    .astype(np.float32)
                )
                self.policies[i] = np.frombuffer(f.read(64 * 4), dtype=np.float32)
                self.values[i] = struct.unpack("<f", f.read(4))[0]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.states[idx]),
            torch.tensor(self.policies[idx]),
            torch.tensor(self.values[idx]),
        )


# The residual tower repeatedly refines the board state representation,
# allowing deep strategic reasoning while preserving earlier features
# through skip connections.
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # First 3×3 convolution:
        # Extracts local spatial features while preserving board resolution.
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Second 3×3 convolution:
        # Further refines features before they are added back to the input.
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Save the input to be added back later (identity / skip connection)
        residual = x

        # First conv + normalisation + non-linearity
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Skip connection: preserve earlier features
        out += residual

        # Final non-linearity
        return F.relu(out)


# OthelloNet is an AlphaZero-style neural network for Othello.
#
# The network takes a board position encoded as two 8×8 planes
# (current player stones and opponent stones) and produces:
#   - A policy over the 64 board positions (move preferences)
#   - A scalar value estimating the expected game outcome
#
# A shared convolutional trunk with a residual tower extracts
# spatial features from the board, which are then refined by
# separate policy and value heads.
#
# The model is designed to be used as an evaluation function
# inside a Monte Carlo Tree Search (MCTS), with move legality,
# pass handling, and exploration logic implemented externally.
class OthelloNet(nn.Module):
    def __init__(self, num_blocks=10):
        super().__init__()

        # Initial convolution
        self.conv1 = nn.Conv2d(2, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        # Residual tower
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(num_blocks)])

        # Policy Head
        self.p_conv = nn.Conv2d(128, 2, kernel_size=1, bias=False)
        self.p_bn = nn.BatchNorm2d(2)
        self.p_fc = nn.Linear(2 * 8 * 8, 64)

        # Value Head
        self.v_conv = nn.Conv2d(128, 1, kernel_size=1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(1 * 8 * 8, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Shared trunk
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        # Policy branch
        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)
        p = F.log_softmax(p, dim=1)

        # Value branch
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))

        return p, v


def export_onnx(model, epoch, device):
    model.eval()

    dummy_input = torch.zeros(1, 2, 8, 8, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        f"othello_net_epoch_{epoch:03d}.onnx",
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )

    model.train()


def train(
    model,
    dataset,
    epochs=10,
    batch_size=256,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        total_loss = 0.0

        for states, target_policies, target_values in loader:
            states = states.to(device)
            target_policies = target_policies.to(device)
            target_values = target_values.to(device)

            optimizer.zero_grad()

            log_policy, value = model(states)

            # Policy loss: cross-entropy with soft targets
            policy_loss = -(target_policies * log_policy).sum(dim=1).mean()

            # Value loss: mean squared error
            value_loss = F.mse_loss(value.squeeze(1), target_values)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: loss={total_loss / len(loader):.4f}")

        export_onnx(model, epoch + 1, device)


if __name__ == "__main__":
    dataset = OthelloDataset(
        "../../crates/othello-self-play/data/selfplay_00500_01000.bin"
    )
    model = OthelloNet(num_blocks=10)

    train(model, dataset, epochs=10, batch_size=256, lr=1e-3, device="cuda")
