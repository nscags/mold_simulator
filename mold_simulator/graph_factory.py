import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class GraphFactory:
    """
    A utility class for generating plots from simulation metrics.
    """

    @staticmethod
    def plot_entropy(
        entropy_log: list[float],
        save_path: Optional[str] = None
    ) -> None:
        frames = np.arange(len(entropy_log))
        plt.figure(figsize=(8, 5))
        plt.plot(frames, entropy_log, color='orange', label="Entropy")
        plt.xlabel("Frame")
        plt.ylabel("Entropy")
        plt.title("Entropy Over Time")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved entropy plot to: {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_avg_displacement(
        displacement_log: list[float],
        save_path: Optional[str] = None
    ) -> None:
        frames = np.arange(len(displacement_log))
        plt.figure(figsize=(8, 5))
        plt.plot(frames, displacement_log, color='blue', label="Avg Displacement")
        plt.xlabel("Frame")
        plt.ylabel("Average Displacement")
        plt.title("Average Particle Displacement Over Time")
        plt.grid(True)
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved avg displacement plot to: {save_path}")
        else:
            plt.show()

    @staticmethod
    def plot_aggregated_metric(
        mean: np.ndarray,
        std: np.ndarray,
        label: str,
        color: str,
        ylabel: str,
        title: str,
        save_path: Optional[str] = None
    ) -> None:
        frames = np.arange(len(mean))
        plt.figure(figsize=(8, 5))
        plt.plot(frames, mean, label=f"Mean {label}", color=color)
        plt.fill_between(frames, mean - std, mean + std, color=color, alpha=0.3, label="±1 Std Dev")
        plt.xlabel("Frame")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved agg plot to: {save_path}")
        else:
            plt.show()