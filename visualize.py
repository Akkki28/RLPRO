import matplotlib.pyplot as plt
import numpy as np


def _estimate_noise(values):
    values = np.asarray(values, dtype=float)
    if values.size < 3:
        return 0.0
    diffs = np.diff(values)
    value_std = np.std(values) + 1e-8
    diff_std = np.std(diffs)
    return float(diff_std / value_std)


def _auto_window(noise, min_window=5, max_window=50):
    if max_window < min_window:
        min_window, max_window = max_window, min_window
    normalized = max(0.0, min(1.0, noise / 1.5))
    window = int(round(min_window + normalized * (max_window - min_window)))
    return max(1, window)


def _moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values
    weights = np.ones(window, dtype=float) / window
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, weights, mode="valid")

def plot_total_rewards(
    total_rewards,
    title="Total Reward per Episode",
    smoothing="auto",
    min_window=5,
    max_window=50,
    show_raw=True,
):
    """
    Plots the total rewards per episode.
    Args:
        total_rewards (list or array): List of total rewards per episode.
        title (str): Title for the plot.
        smoothing (str|int|None): "auto" chooses a window based on noise,
            an int sets the window directly, None disables smoothing.
        min_window (int): Minimum smoothing window when using auto.
        max_window (int): Maximum smoothing window when using auto.
        show_raw (bool): Plot raw rewards as a faint line.
    """
    rewards = np.asarray(total_rewards, dtype=float)
    if rewards.size == 0:
        return

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(11, 6), dpi=110)

    noise = _estimate_noise(rewards)
    if smoothing == "auto":
        window = _auto_window(noise, min_window=min_window, max_window=max_window)
    elif isinstance(smoothing, (int, np.integer)):
        window = int(smoothing)
    else:
        window = 1

    if show_raw:
        ax.plot(rewards, color="#0098c2", alpha=0.35, linewidth=1.0, label="Raw")

    if window > 1:
        smoothed = _moving_average(rewards, window)
        ax.plot(
            smoothed,
            color="#0004ff",
            linewidth=2.5,
            label=f"Smoothed (window={window})",
        )
    else:
        ax.plot(rewards, color="#0004ff", linewidth=2.0, label="Total Reward")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title)
    ax.grid(True, which="major", linestyle="--", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    plt.show()
