import numpy as np
from typing import Union


class TrailMap:
    """
    Represents a 2D trail intensity map for particles to sense, deposit to, and modify over time.
    """

    def __init__(
        self,
        width: int,
        height: int,
        decay: float = 0.95,
        diffusion_rate: float = 0.1
    ) -> None:
        """
        Initializes the trail map with specified dimensions and parameters.
        """
        self.width: int = width
        self.height: int = height
        self.decay: float = decay
        self.diffusion_rate: float = diffusion_rate
        self.grid: np.ndarray = np.ones((height, width), dtype=np.float32) * 0.1

    def update(self) -> None:
        """
        Applies decay and diffusion to the trail map.
        This simulates natural evaporation and spreading of trails over time.
        """
        # Decay
        self.grid *= self.decay

        # Diffusion
        self.grid += self.diffusion_rate * (
            np.roll(self.grid, 1, axis=0) + np.roll(self.grid, -1, axis=0) +
            np.roll(self.grid, 1, axis=1) + np.roll(self.grid, -1, axis=1) -
            4 * self.grid
        )

    def deposit(
        self, 
        x: Union[int, float], 
        y: Union[int, float], 
        amount: float
    ) -> None:
        """
        Adds trail intensity at a given coordinate.
        """
        xi = int(x) % self.width
        yi = int(y) % self.height
        self.grid[yi, xi] += amount

    def get_concentration(
        self, 
        x: Union[int, float], 
        y: Union[int, float]
    ) -> float:
        """
        Returns the trail concentration at a given coordinate.
        """
        xi = int(np.clip(x, 0, self.width - 1))
        yi = int(np.clip(y, 0, self.height - 1))
        return self.grid[yi, xi]
