import numpy as np
from time import time

from particle import Particle
from trailmap import TrailMap
from metric_tracker import MetricTracker
from graph_factory import GraphFactory
from simulator import Simulator


def main():
    width, height = 200, 200
    n_particles = 1000
    n_frames = 1000

    trail_map = TrailMap(width, height, decay=0.95, diffusion_rate=0.1)
    metric_tracker = MetricTracker()
    graph_factory = GraphFactory()

    particles = [
        Particle(
            x=np.random.randint(0, width),
            y=np.random.randint(0, height),
            angle=np.random.uniform(0, 2 * np.pi),
            speed=1.0
        )
        for _ in range(n_particles)
    ]

    sim = Simulator(particles, trail_map, metric_tracker)
    sim.run(n_frames)
    sim.save_gif("slime_mold_simulation.gif")
    metric_tracker.save_to_csv("simulation_metrics.csv")

    # TODO: plot cluster analysis (average distance between particles)
    #       
    graph_factory.plot_entropy(metric_tracker.entropy_log, save_path="entropy_plot.png")


if __name__ == "__main__":
    start = time()
    main()
    end = time()

    print(f"Runtime: {end - start}")