import argparse
import json
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time


class OthelloDataset(Dataset):
    def __init__(self, path):
        print(f"Opening dataset: {path}")
        start_time = time.time()

        with open(path, "rb") as f:
            magic, version, n = struct.unpack("<III", f.read(12))
            assert magic == 0x4F54484C  # "OTHL"
            assert version == 1

            print(f"Loading {n:,} samples into memory...")

            self.states = np.zeros((n, 2, 8, 8), dtype=np.float32)
            self.policies = np.zeros((n, 64), dtype=np.float32)
            self.values = np.zeros((n,), dtype=np.float32)

            # Show progress every 10% of samples
            checkpoint = max(1, n // 10)

            for i in range(n):
                self.states[i] = (
                    np.frombuffer(f.read(2 * 8 * 8 * 4), dtype=np.int32)
                    .reshape(2, 8, 8)
                    .astype(np.float32)
                )
                self.policies[i] = np.frombuffer(f.read(64 * 4), dtype=np.float32)
                self.values[i] = struct.unpack("<f", f.read(4))[0]

                if (i + 1) % checkpoint == 0:
                    pct = (i + 1) / n * 100
                    print(f"  Loaded {i + 1:,} / {n:,} samples ({pct:.0f}%)")

        elapsed = time.time() - start_time
        print(f"Dataset loaded in {elapsed:.1f}s ({n / elapsed:.0f} samples/sec)")

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


def export_onnx(model, epoch, device, prefix):
    model.eval()

    # Ensure output directory exists
    import os

    output_path = f"{prefix}_othello_net_epoch_{epoch:03d}.onnx"
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    dummy_input = torch.zeros(1, 2, 8, 8, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        f"{prefix}_othello_net_epoch_{epoch:03d}.onnx",
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
    prefix,
    epochs=10,
    batch_size=256,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"\nStarting training on {device}")
    print(f"  Dataset size: {len(dataset):,} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print()

    model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    num_batches = len(loader)

    # Statistics tracking
    training_stats = {
        "config": {
            "dataset_size": len(dataset),
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "device": device,
            "num_batches_per_epoch": num_batches,
        },
        "epochs": [],
    }

    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        batch_count = 0

        batch_losses = []

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
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            batch_count += 1

            batch_losses.append(
                {
                    "batch": batch_count,
                    "loss": loss.item(),
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                }
            )

            # Show progress every 10% of batches
            if batch_count % max(1, num_batches // 10) == 0:
                avg_loss = total_loss / batch_count
                pct = batch_count / num_batches * 100
                print(
                    f"  Epoch {epoch + 1}/{epochs} - Batch {batch_count}/{num_batches} ({pct:.0f}%) - Loss: {avg_loss:.4f}"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)
        avg_policy_loss = total_policy_loss / len(loader)
        avg_value_loss = total_value_loss / len(loader)

        print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.1f}s")
        print(
            f"  Average loss: {avg_loss:.4f} (policy: {avg_policy_loss:.4f}, value: {avg_value_loss:.4f})"
        )

        # Compute standard error of the mean for the losses
        batch_loss_values = [b["loss"] for b in batch_losses]
        batch_policy_loss_values = [b["policy_loss"] for b in batch_losses]
        batch_value_loss_values = [b["value_loss"] for b in batch_losses]

        loss_std = np.std(batch_loss_values)
        policy_loss_std = np.std(batch_policy_loss_values)
        value_loss_std = np.std(batch_value_loss_values)

        loss_sem = loss_std / np.sqrt(len(batch_loss_values))
        policy_loss_sem = policy_loss_std / np.sqrt(len(batch_policy_loss_values))
        value_loss_sem = value_loss_std / np.sqrt(len(batch_value_loss_values))

        # Record epoch statistics
        epoch_stats = {
            "epoch": epoch + 1,
            "time_seconds": epoch_time,
            "avg_loss": avg_loss,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "loss_std": loss_std,
            "policy_loss_std": policy_loss_std,
            "value_loss_std": value_loss_std,
            "loss_sem": loss_sem,
            "policy_loss_sem": policy_loss_sem,
            "value_loss_sem": value_loss_sem,
            "batches": batch_losses,
        }
        training_stats["epochs"].append(epoch_stats)

        export_onnx(model, epoch + 1, device, prefix)

    # Save training statistics to JSON file
    stats_file = f"{prefix}_training_stats.json"
    with open(stats_file, "w") as f:
        json.dump(training_stats, f, indent=2)
    print(f"Training statistics saved to {stats_file}")

    return training_stats

def export_dummy_model(prefix, device="cpu"):
    """
    Exports a tiny ONNX model that outputs:
      - uniform policy over 64 moves
      - zero value
    Useful for iteration 0 of self-play when no trained model exists.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import os

    class TinyOthelloNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(2, 16, 3, padding=1)
            self.policy_head = nn.Conv2d(16, 1, 1)
            self.value_head = nn.Linear(16 * 8 * 8, 1)

        def forward(self, x):
            h = F.relu(self.conv(x))
            p = self.policy_head(h).view(-1)         # flatten to (64,)
            v = torch.tensor([0.0], device=x.device) # always 0
            return F.log_softmax(p, dim=0), v

    model = TinyOthelloNet().to(device)
    model.eval()

    dummy_input = torch.zeros(1, 2, 8, 8, device=device)
    output_path = f"{prefix}_othello_net_epoch_000.onnx"
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"Dummy model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--res-blocks", type=int, default=10)
    parser.add_argument("--out-prefix", type=str, default="othello_net")
    parser.add_argument("--dummy-model", action="store_true",
                        help="Export a dummy ONNX model for iteration 0")
    args = parser.parse_args()

    if args.dummy_model:
        export_dummy_model(args.out_prefix, device="cpu")
    else:
        dataset = OthelloDataset(args.data)
        model = OthelloNet(num_blocks=args.res_blocks)
        train(
            model,
            dataset,
            prefix=args.out_prefix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

