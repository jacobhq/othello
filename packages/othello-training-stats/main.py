import argparse
import pandas as pd
import json
import matplotlib.pyplot as plt

# Load the dataset, and drop data we don't need
def load_dataset(filepath):
    data = json.load(open(filepath))
    df = pd.json_normalize(data["epochs"]).drop("batches", axis=1)
    config = data["config"]
    return df, config

def main(start, end, prefix):
    # All the dataframes, and cumulative epochs
    all_dfs = []
    cumulative_epochs = 0

    # go through the range the user told us
    for i in range(start, end):
        df, config = load_dataset(f"{prefix}_{i}_training_stats.json")

        # Build up cumulative epochs (over all iterations)
        df["global_epoch"] = df["epoch"] + cumulative_epochs
        cumulative_epochs += len(df)

        all_dfs.append(df)

    # Make a big new dataframe
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Create subplots for different metrics
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Loss components
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

    # Standard deviations
    axes[1, 1].plot(combined_df["global_epoch"], combined_df["loss_std"], label="Loss Std")
    axes[1, 1].plot(combined_df["global_epoch"], combined_df["policy_loss_std"], label="Policy Loss Std")
    axes[1, 1].plot(combined_df["global_epoch"], combined_df["value_loss_std"], label="Value Loss Std")
    axes[1, 1].set_ylabel("Standard Deviation")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Standard error of mean
    axes[2, 0].plot(combined_df["global_epoch"], combined_df["loss_sem"], label="Loss SEM")
    axes[2, 0].plot(combined_df["global_epoch"], combined_df["policy_loss_sem"], label="Policy Loss SEM")
    axes[2, 0].plot(combined_df["global_epoch"], combined_df["value_loss_sem"], label="Value Loss SEM")
    axes[2, 0].set_ylabel("Standard Error of Mean")
    axes[2, 0].set_xlabel("Epoch (across all runs)")
    axes[2, 0].legend()
    axes[2, 0].grid(True)

    # Time per epoch
    axes[2, 1].plot(combined_df["global_epoch"], combined_df["time_seconds"], color="purple")
    axes[2, 1].set_ylabel("Time (seconds)")
    axes[2, 1].set_xlabel("Epoch (across all runs)")
    axes[2, 1].grid(True)

    plt.suptitle(f"Training Metrics for {prefix}", fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--prefix", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)

    args = parser.parse_args()

    main(args.start, args.end, args.prefix)