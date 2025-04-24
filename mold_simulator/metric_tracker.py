import numpy as np
from typing import List, Optional
from particle import Particle
from trailmap import TrailMap


class MetricTracker:
    """
    Tracks convergence/divergence metrics of the particle simulation over time.
    """

    def __init__(self) -> None:
        """
        Initializes empty logs for entropy and displacement metrics.
        """
        self.entropy_log: List[float] = []
        self.avg_displacement_log: List[float] = []
        self.prev_positions: Optional[np.ndarray] = None

    def log_metrics(
        self, 
        particles: List[Particle], 
        trail_map: TrailMap
    ) -> None:
        """
        Logs the current frame's entropy and average particle displacement.
        """
        entropy = self.compute_entropy(trail_map.grid)
        self.entropy_log.append(entropy)

        displacement = self.compute_avg_displacement(particles)
        self.avg_displacement_log.append(displacement)

    def compute_entropy(
        self, 
        trail: np.ndarray
    ) -> float:
        """
        Computes the entropy of the trail map. Lower entropy can indicate convergence.
        """
        prob = trail / np.sum(trail)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log(prob)))

    def compute_avg_displacement(
        self, 
        particles: List[Particle]
    ) -> float:
        """
        Computes the average movement distance of particles compared to the previous frame.
        """
        current_positions = np.array([[p.x, p.y] for p in particles])

        if self.prev_positions is None:
            self.prev_positions = current_positions
            return 0.0

        deltas = np.linalg.norm(current_positions - self.prev_positions, axis=1)
        self.prev_positions = current_positions
        return float(np.mean(deltas))

    def save_to_csv(
        self, 
        filename: str = "metrics.csv"
    ) -> None:
        """
        Saves the logged metrics to a CSV file.
        """
        data = np.column_stack((self.entropy_log, self.avg_displacement_log))
        header = "entropy,avg_displacement"
        np.savetxt(filename, data, delimiter=",", header=header, comments='')

