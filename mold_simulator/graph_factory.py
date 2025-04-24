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
        """
        Plots entropy over time from the simulation.
        """
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
            print(f"[✓] Saved entropy plot to: {save_path}")
        else:
            plt.show()