import argparse
import os
import pandas as pd
import json
import matplotlib.pyplot as plt

# Load the dataset, and drop data we don't need
def load_dataset(filepath):
    data = json.load(open(filepath))
    df = pd.json_normalize(data["epochs"]).drop("batches", axis=1)
    config = data["config"]
    return df, config

def load_eval_results(evals_dir, prefix, start, end):
    """Load evaluation JSON files for each iteration in [start, end)."""
    iters = []
    vs_prev_rates = []
    vs_random_rates = []

    for i in range(start, end):
        iters.append(i)

        # vs previous / best model
        vs_prev_path = os.path.join(evals_dir, f"{prefix}_iter{i:03d}_vs_prev.json")
        if os.path.exists(vs_prev_path):
            with open(vs_prev_path) as f:
                vs_prev_rates.append(json.load(f)["win_rate"] * 100.0)
        else:
            vs_prev_rates.append(None)

        # vs random
        vs_random_path = os.path.join(evals_dir, f"{prefix}_iter{i:03d}_vs_random.json")
        if os.path.exists(vs_random_path):
            with open(vs_random_path) as f:
                vs_random_rates.append(json.load(f)["win_rate"] * 100.0)
        else:
            vs_random_rates.append(None)

    return iters, vs_prev_rates, vs_random_rates

def load_training_data(data_dir, prefix, start, end):
    """Load and combine training stats across iterations."""
    all_dfs = []
    cumulative_epochs = 0

    for i in range(start, end):
        filepath = os.path.join(data_dir, f"{prefix}_{i}_training_stats.json")
        df, config = load_dataset(filepath)
        df["global_epoch"] = df["epoch"] + cumulative_epochs
        cumulative_epochs += len(df)
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)

def plot_loss(axes, combined_df):
    """Plot loss, std dev, SEM, and timing charts on a 3x2 grid."""
    axes[0, 0].plot(combined_df["global_epoch"], combined_df["avg_loss"], label="Total Loss")
    axes[0, 0].fill_between(combined_df["global_epoch"],
                            combined_df["avg_loss"] - combined_df["loss_std"],
                            combined_df["avg_loss"] + combined_df["loss_std"],
                            alpha=0.3)
    axes[0, 0].set_ylabel("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(combined_df["global_epoch"], combined_df["avg_policy_loss"], label="Policy Loss", color="orange")
    axes[0, 1].fill_between(combined_df["global_epoch"],
                            combined_df["avg_policy_loss"] - combined_df["policy_loss_std"],
                            combined_df["avg_policy_loss"] + combined_df["policy_loss_std"],
                            alpha=0.3, color="orange")
    axes[0, 1].set_ylabel("Policy Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(combined_df["global_epoch"], combined_df["avg_value_loss"], label="Value Loss", color="green")
    axes[1, 0].fill_between(combined_df["global_epoch"],
                            combined_df["avg_value_loss"] - combined_df["value_loss_std"],
                            combined_df["avg_value_loss"] + combined_df["value_loss_std"],
                            alpha=0.3, color="green")
    axes[1, 0].set_ylabel("Value Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(combined_df["global_epoch"], combined_df["loss_std"], label="Loss Std")
    axes[1, 1].plot(combined_df["global_epoch"], combined_df["policy_loss_std"], label="Policy Loss Std")
    axes[1, 1].plot(combined_df["global_epoch"], combined_df["value_loss_std"], label="Value Loss Std")
    axes[1, 1].set_ylabel("Standard Deviation")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[2, 0].plot(combined_df["global_epoch"], combined_df["loss_sem"], label="Loss SEM")
    axes[2, 0].plot(combined_df["global_epoch"], combined_df["policy_loss_sem"], label="Policy Loss SEM")
    axes[2, 0].plot(combined_df["global_epoch"], combined_df["value_loss_sem"], label="Value Loss SEM")
    axes[2, 0].set_ylabel("Standard Error of Mean")
    axes[2, 0].set_xlabel("Epoch (across all runs)")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    axes[2, 1].plot(combined_df["global_epoch"], combined_df["time_seconds"], color="purple")
    axes[2, 1].set_ylabel("Time (seconds)")
    axes[2, 1].set_xlabel("Epoch (across all runs)")
    axes[2, 1].grid(True)

def plot_eval(axes, eval_iters, vs_prev_rates, vs_random_rates):
    """Plot eval win rate charts on a 1x2 grid."""
    prev_iters = [it for it, r in zip(eval_iters, vs_prev_rates) if r is not None]
    prev_rates = [r for r in vs_prev_rates if r is not None]
    axes[0].plot(prev_iters, prev_rates, marker="o", color="crimson", label="vs Previous")
    axes[0].axhline(y=55, color="gray", linestyle="--", alpha=0.7, label="55% gate")
    axes[0].axhline(y=50, color="lightgray", linestyle=":", alpha=0.7, label="50% baseline")
    axes[0].set_ylabel("Win Rate (%)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_title("vs Previous / Best Model")
    axes[0].legend()
    axes[0].grid(True)

    rand_iters = [it for it, r in zip(eval_iters, vs_random_rates) if r is not None]
    rand_rates = [r for r in vs_random_rates if r is not None]
    axes[1].plot(rand_iters, rand_rates, marker="o", color="teal", label="vs Random")
    axes[1].axhline(y=90, color="gray", linestyle="--", alpha=0.7, label="90% gate")
    axes[1].set_ylabel("Win Rate (%)")
    axes[1].set_xlabel("Iteration")
    axes[1].set_title("vs Random Player")
    axes[1].legend()
    axes[1].grid(True)

def main(start, end, prefix, evals_dir, data_dir, mode):
    show_loss = mode in ("all", "loss")
    show_eval = mode in ("all", "eval")

    # Load data as needed
    combined_df = None
    if show_loss:
        combined_df = load_training_data(data_dir, prefix, start, end)

    eval_iters, vs_prev_rates, vs_random_rates = None, [], []
    has_eval_data = False
    if show_eval:
        eval_iters, vs_prev_rates, vs_random_rates = load_eval_results(evals_dir, prefix, start, end)
        has_eval_data = any(r is not None for r in vs_prev_rates + vs_random_rates)

    # Calculate grid layout
    loss_rows = 3 if show_loss else 0
    eval_rows = 1 if (show_eval and has_eval_data) else 0
    total_rows = loss_rows + eval_rows

    if total_rows == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(total_rows, 2, figsize=(15, 4 * total_rows))

    # Ensure axes is always 2D
    if total_rows == 1:
        axes = axes.reshape(1, -1)

    if show_loss:
        plot_loss(axes[:3], combined_df)

    if show_eval and has_eval_data:
        eval_axes = axes[loss_rows]
        plot_eval(eval_axes, eval_iters, vs_prev_rates, vs_random_rates)

    plt.suptitle(f"Training Metrics for {prefix}", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--evals-dir", type=str, default="../../crates/othello-learn/evals",
                        help="Directory containing eval JSON files")
    parser.add_argument("--data-dir", type=str, default=".",
                        help="Directory containing training stats JSON files")
    parser.add_argument("--mode", type=str, choices=["all", "loss", "eval"], default="all",
                        help="What to graph: 'all' (default), 'loss' (training only), 'eval' (evaluation only)")

    args = parser.parse_args()

    main(args.start, args.end, args.prefix, args.evals_dir, args.data_dir, args.mode)
