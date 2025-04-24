import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List

from particle import Particle
from trailmap import TrailMap
from metric_tracker import MetricTracker


class Simulator:
    """
    Orchestrates the particle-based simulation, handling the update cycle,
    trail dynamics, frame collection, and animation rendering.
    """

    def __init__(
        self, 
        particles: List[Particle], 
        trail_map: TrailMap,
        metric_tracker: MetricTracker
    ) -> None:
        """
        Initializes the simulation with particles and a trail map.
        """
        self.particles: List[Particle] = particles
        self.trail_map: TrailMap = trail_map
        self.metric_tracker: MetricTracker = metric_tracker
        self.frames: List[np.ndarray] = []
        self.width: int = trail_map.width
        self.height: int = trail_map.height

    def step(self) -> None:
        """
        Performs a single update step:
        - Updates each particle (movement + sensing)
        - Deposits trails
        - Applies trail decay and diffusion
        """
        for particle in self.particles:
            particle.update(self.trail_map.grid, self.width, self.height)
            particle.deposit(self.trail_map.grid)
        self.trail_map.update()

        if self.metric_tracker:
            self.metric_tracker.log_metrics(self.particles, self.trail_map)

    def run(
        self, 
        n_frames: int
    ) -> None:
        """
        Runs the full simulation for a given number of steps.
        """
        for _ in range(n_frames):
            self.step()
            self.render_frame()

    def render_frame(self) -> None:
        """
        Captures the current trail state as a frame using a log scale
        to enhance visibility of fine patterns.
        """
        self.frames.append(np.log1p(self.trail_map.grid.copy()))

    def save_gif(
        self, 
        filename: str = "slime_mold_simulation.gif"
    ) -> str:
        """
        Generates and saves an animated GIF from the collected frames.
        """
        fig = plt.figure(figsize=(5, 5))
        im = plt.imshow(self.frames[0], cmap='inferno', origin='lower', animated=True)

        def updatefig(i: int):
            im.set_array(self.frames[i])
            return [im]

        ani = animation.FuncAnimation(
            fig,
            updatefig,
            frames=len(self.frames),
            interval=100,
            blit=True
        )

        ani.save(filename, writer='pillow')
        return filename
