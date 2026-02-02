import argparse
import json
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
import time


def ddp_print(*args, force=False, **kwargs):
    """
    Print only on rank 0 by default.
    Set force=True to print on all ranks.
    """
    if force or not dist.is_initialized() or dist.get_rank() == 0:
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[rank {rank}]", *args, **kwargs)


def load_single_bin_file(path):
    """Load samples from a single .bin file, returning (states, policies, values) arrays."""
    with open(path, "rb") as f:
        magic, version, n = struct.unpack("<III", f.read(12))
        assert magic == 0x4F54484C, f"Invalid magic number in {path}"
        assert version == 1, f"Unsupported version {version} in {path}"

        states = np.zeros((n, 2, 8, 8), dtype=np.float32)
        policies = np.zeros((n, 64), dtype=np.float32)
        values = np.zeros((n,), dtype=np.float32)

        for i in range(n):
            states[i] = (
                np.frombuffer(f.read(2 * 8 * 8 * 4), dtype=np.int32)
                .reshape(2, 8, 8)
                .astype(np.float32)
            )
            policies[i] = np.frombuffer(f.read(64 * 4), dtype=np.float32)
            values[i] = struct.unpack("<f", f.read(4))[0]

    return states, policies, values, n


def extract_file_sort_key(filepath):
    """
    Extract numeric sort key from a filename like 'prefix_selfplay_60000_64000.bin'.
    Returns a tuple of (start_num, end_num) for proper numeric sorting.
    Falls back to (0, 0) if no numbers are found.
    """
    import re
    import os
    basename = os.path.basename(filepath)
    # Match patterns like _60000_64000 or _100000_104000 at the end before .bin
    match = re.search(r'_(\d+)_(\d+)\.bin$', basename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    # Fallback: try to find any numbers in the filename
    numbers = re.findall(r'\d+', basename)
    if numbers:
        return tuple(int(n) for n in numbers[-2:]) if len(numbers) >= 2 else (int(numbers[-1]), 0)
    return (0, 0)


def find_recent_data_files(data_dir, window_size=3, prefix=None):
    """
    Find the most recent `window_size` iterations worth of .bin files.

    Files are sorted numerically by the game range in the filename
    (e.g., prefix_selfplay_60000_64000.bin) and the last `window_size` files are returned.

    Args:
        data_dir: Directory containing .bin files
        window_size: Number of recent files to use
        prefix: If provided, only include files starting with this prefix
    """
    import glob
    import os

    pattern = os.path.join(data_dir, "**/*.bin")
    all_files = sorted(glob.glob(pattern, recursive=True), key=extract_file_sort_key)

    # Filter by prefix if provided
    if prefix:
        all_files = [f for f in all_files if os.path.basename(f).startswith(prefix)]

    if not all_files:
        raise ValueError(f"No .bin files found in {data_dir}" +
                        (f" with prefix '{prefix}'" if prefix else ""))

    # Take the last `window_size` files
    selected = all_files[-window_size:] if len(all_files) > window_size else all_files

    ddp_print(f"Found {len(all_files)} total .bin files" +
          (f" with prefix '{prefix}'" if prefix else "") +
          f", using last {len(selected)}:")
    for f in selected:
        ddp_print(f"  - {os.path.basename(f)}")

    return selected


class OthelloDataset(Dataset):
    def __init__(self, path, window_size=None, prefix=None):
        """
        Load training data from one or more .bin files.

        Args:
            path: Either a single .bin file, a directory containing .bin files,
                  or a comma-separated list of .bin files.
            window_size: If path is a directory, only use the last `window_size`
                        files (sorted by name). If None, use all files.
            prefix: If path is a directory, only include files starting with this prefix.
        """
        import os

        start_time = time.time()

        # Determine which files to load
        if os.path.isdir(path):
            if window_size is not None:
                files = find_recent_data_files(path, window_size, prefix=prefix)
            else:
                import glob
                all_files = sorted(glob.glob(os.path.join(path, "**/*.bin"), recursive=True), key=extract_file_sort_key)
                if prefix:
                    files = [f for f in all_files if os.path.basename(f).startswith(prefix)]
                else:
                    files = all_files
                ddp_print(f"Loading all {len(files)} .bin files from {path}")
        elif "," in path:
            files = [f.strip() for f in path.split(",")]
            ddp_print(f"Loading {len(files)} specified .bin files")
        else:
            files = [path]
            ddp_print(f"Loading single file: {path}")

        if not files:
            raise ValueError(f"No .bin files found at {path}")

        # First pass: count total samples
        total_samples = 0
        file_sample_counts = []
        for f in files:
            with open(f, "rb") as fp:
                magic, version, n = struct.unpack("<III", fp.read(12))
                assert magic == 0x4F54484C, f"Invalid magic in {f}"
                file_sample_counts.append(n)
                total_samples += n

        ddp_print(f"Total samples to load: {total_samples:,}")

        # Allocate arrays
        self.states = np.zeros((total_samples, 2, 8, 8), dtype=np.float32)
        self.policies = np.zeros((total_samples, 64), dtype=np.float32)
        self.values = np.zeros((total_samples,), dtype=np.float32)

        # Load all files
        offset = 0
        for file_path, n_samples in zip(files, file_sample_counts):
            ddp_print(f"  Loading {os.path.basename(file_path)} ({n_samples:,} samples)...")
            states, policies, values, _ = load_single_bin_file(file_path)
            self.states[offset:offset + n_samples] = states
            self.policies[offset:offset + n_samples] = policies
            self.values[offset:offset + n_samples] = values
            offset += n_samples

        elapsed = time.time() - start_time
        ddp_print(f"Dataset loaded in {elapsed:.1f}s ({total_samples / elapsed:.0f} samples/sec)")

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

def export_initial_model(prefix, num_blocks, device="cpu"):
    """
    Exports a randomly initialized OthelloNet WITHOUT training.
    This is suitable for iteration 0 self-play.
    """
    print("Exporting randomly initialized model (no training)")

    model = OthelloNet(num_blocks=num_blocks)
    model.to(device)
    model.eval()

    dummy_input = torch.zeros(1, 2, 8, 8, device=device)

    output_path = f"{prefix}_othello_net_epoch_000.onnx"
    import os
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
        dynamo=False,
    )

    print(f"Initial model exported to {output_path}")

def train(
    model,
    dataset,
    prefix,
    epochs=10,
    batch_size=256,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    ddp_print(f"\nStarting training on {device}")
    ddp_print(f"  Dataset size: {len(dataset):,} samples")
    ddp_print(f"  Batch size: {batch_size}")
    ddp_print(f"  Epochs: {epochs}")
    ddp_print(f"  Learning rate: {lr}")
    ddp_print()

    model.train()

    if dist.is_initialized():
        sampler = DistributedSampler(dataset, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
        if sampler is not None:
            sampler.set_epoch(epoch)

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
                ddp_print(
                    f"  Epoch {epoch + 1}/{epochs} - Batch {batch_count}/{num_batches} ({pct:.0f}%) - Loss: {avg_loss:.4f}"
                )

        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / len(loader)
        avg_policy_loss = total_policy_loss / len(loader)
        avg_value_loss = total_value_loss / len(loader)

        ddp_print(f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.1f}s")
        ddp_print(
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

        if not dist.is_initialized() or dist.get_rank() == 0:
            export_onnx(
                model.module if hasattr(model, "module") else model,
                epoch + 1,
                device,
                prefix
            )

    # Save PyTorch checkpoint for resuming training
    checkpoint_file = f"{prefix}_checkpoint.pt"
    if not dist.is_initialized() or dist.get_rank() == 0:
        torch.save(
            model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            checkpoint_file
        )
    ddp_print(f"PyTorch checkpoint saved to {checkpoint_file}")

    # Save training statistics to JSON file
    stats_file = f"{prefix}_training_stats.json"
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(stats_file, "w") as f:
            json.dump(training_stats, f, indent=2)
    ddp_print(f"Training statistics saved to {stats_file}")

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
        dynamo=False
    )

    ddp_print(f"Dummy model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=False,
                        help="Path to .bin file, directory of .bin files, or comma-separated list of files")
    parser.add_argument("--window", type=int, default=3,
                        help="Number of recent data files to use (sliding window). Only applies when --data is a directory.")
    parser.add_argument("--data-prefix", type=str, default=None,
                        help="Only load data files starting with this prefix. Only applies when --data is a directory.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--res-blocks", type=int, default=10)
    parser.add_argument("--out-prefix", type=str, default="othello_net")
    parser.add_argument("--dummy-model", action="store_true",
                        help="Export a dummy ONNX model for iteration 0")
    parser.add_argument(
        "--init-model",
        action="store_true",
        help="Export a randomly initialized model without training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint file to resume training from"
    )
    args = parser.parse_args()

    if args.init_model:
        export_initial_model(
            prefix=args.out_prefix,
            num_blocks=args.res_blocks,
            device="cpu"
        )
    else:
        if args.data is None:
            raise ValueError("--data is required for training")

        use_ddp = torch.cuda.device_count() > 1

        if use_ddp:
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        dataset = OthelloDataset(args.data, window_size=args.window, prefix=args.data_prefix)

        model = OthelloNet(num_blocks=args.res_blocks).to(device)

        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                output_device=local_rank
            )


        # Load checkpoint if provided
        if args.checkpoint:
            ddp_print(f"Loading checkpoint from {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint)
            ddp_print("Checkpoint loaded successfully")

        train(
            model,
            dataset,
            prefix=args.out_prefix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )

        if dist.is_initialized():
            dist.destroy_process_group()
