import matplotlib.pyplot as plt

def plot_total_rewards(total_rewards, title="Total Reward per Episode"):
    """
    Plots the total rewards per episode.
    Args:
        total_rewards (list or array): List of total rewards per episode.
        title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, label="Total Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
