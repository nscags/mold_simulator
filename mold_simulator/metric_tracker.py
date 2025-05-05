import numpy as np
from typing import List, Optional

from .particle import Particle
from .trailmap import TrailMap


class MetricTracker:
    """
    Tracks and aggregates convergence/divergence metrics across multiple simulation trials
    """

    def __init__(self) -> None:
        self.entropy_trials: List[List[float]] = []
        self.displacement_trials: List[List[float]] = []
        self._entropy_log: List[float] = []
        self._avg_displacement_log: List[float] = []
        self._prev_positions: Optional[np.ndarray] = None

    def start_new_trial(self) -> None:
        self._entropy_log = []
        self._avg_displacement_log = []
        self._prev_positions = None

    def log_metrics(self, particles: List[Particle], trail_map: TrailMap) -> None:
        entropy = self.compute_entropy(trail_map.grid)
        self._entropy_log.append(entropy)

        displacement = self.compute_avg_displacement(particles)
        self._avg_displacement_log.append(displacement)

    def finalize_trial(self) -> None:
        self.entropy_trials.append(self._entropy_log.copy())
        self.displacement_trials.append(self._avg_displacement_log.copy())

    def compute_entropy(self, trail: np.ndarray) -> float:
        prob = trail / np.sum(trail)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log(prob)))

    def compute_avg_displacement(self, particles: List[Particle]) -> float:
        current_positions = np.array([[p.x, p.y] for p in particles])

        if self._prev_positions is None:
            self._prev_positions = current_positions
            return 0.0

        deltas = np.linalg.norm(current_positions - self._prev_positions, axis=1)
        self._prev_positions = current_positions
        return float(np.mean(deltas))

    def save_to_csv(self, filename: str = "metrics.csv") -> None:
        """
        Saves only the most recent trial data
        """
        import numpy as np
        data = np.column_stack((self._entropy_log, self._avg_displacement_log))
        header = "entropy,avg_displacement"
        np.savetxt(filename, data, delimiter=",", header=header, comments='')

    def get_aggregate_stats(self) -> dict:
        """
        Returns aggregate statistics across all trials
        """
        entropy_array = np.array(self.entropy_trials)
        disp_array = np.array(self.displacement_trials)

        return {
            "entropy_mean": np.mean(entropy_array, axis=0),
            "entropy_std": np.std(entropy_array, axis=0),
            "displacement_mean": np.mean(disp_array, axis=0),
            "displacement_std": np.std(disp_array, axis=0)
        }
