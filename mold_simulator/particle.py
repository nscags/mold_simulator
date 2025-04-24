import numpy as np
from typing import Union


class Particle:
    """
    Represents an individual agent that navigates based on trail concentration
    and deposits trail as it moves. Inspired by slime mold behavior.
    """

    def __init__(
        self,
        x: Union[int, float],
        y: Union[int, float],
        angle: float,
        speed: float,
        sensor_distance: int = 5,
        sensor_angle: float = np.pi / 4,
        turn_angle: float = 0.3
    ) -> None:
        """
        Initializes the particle with position, orientation, and sensing parameters.
        """
        self.x = x
        self.y = y
        self.angle = angle
        self.speed = speed
        self.sensor_distance = sensor_distance
        self.sensor_angle = sensor_angle
        self.turn_angle = turn_angle

    def sense(
        self,
        trailmap: np.ndarray,
        width: int,
        height: int,
        offset: float
    ) -> float:
        """
        Samples the trail concentration in the direction of the current angle plus an offset.
        """
        dx = int(np.cos(self.angle + offset) * self.sensor_distance)
        dy = int(np.sin(self.angle + offset) * self.sensor_distance)
        sx = int(np.clip(self.x + dx, 0, width - 1))
        sy = int(np.clip(self.y + dy, 0, height - 1))
        return trailmap[sy, sx]

    def update(
        self, 
        trailmap: np.ndarray, 
        width: int, 
        height: int
    ) -> None:
        """
        Updates the particle's orientation and position based on trail sensing.
        """
        f = self.sense(trailmap, width, height, 0)
        l = self.sense(trailmap, width, height, -self.sensor_angle)
        r = self.sense(trailmap, width, height, self.sensor_angle)

        if f > l and f > r:
            pass  # Continue straight
        elif l > r:
            self.angle -= self.turn_angle
        elif r > l:
            self.angle += self.turn_angle
        else:
            self.angle += np.random.uniform(-self.turn_angle, self.turn_angle)

        # Move the particle and wrap around the edges
        self.x = (self.x + np.cos(self.angle) * self.speed) % width
        self.y = (self.y + np.sin(self.angle) * self.speed) % height

    def deposit(
        self, 
        trailmap: np.ndarray
    ) -> None:
        """
        Deposits trail intensity at the particle's current position.
        """
        xi = int(self.x) % trailmap.shape[1]
        yi = int(self.y) % trailmap.shape[0]
        trailmap[yi, xi] += 5.0
