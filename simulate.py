import numpy as np
from time import time
import os

from mold_simulator.particle import Particle
from mold_simulator.trailmap import TrailMap
from mold_simulator.metric_tracker import MetricTracker
from mold_simulator.graph_factory import GraphFactory
from mold_simulator.simulator import Simulator


def simulate(
    width=200,
    height=200,
    decay=0.95,
    diffusion_rate=0.1, 
    n_particles=1000,
    n_frames=5000,
    n_trials=10,
):
    metric_tracker = MetricTracker()
    graph_factory = GraphFactory()

    for trial in range(n_trials):
        t_start = time()
        print(f"Initializing Simulator, Trial {trial+1}/{n_trials}...")
        trail_map = TrailMap(width, height, decay=decay, diffusion_rate=diffusion_rate)
        metric_tracker.start_new_trial()

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
        print(f"Simulating {n_particles} particles, {n_frames} frames: ")
        sim.run(n_frames)
        metric_tracker.finalize_trial()

        # per trial metrics if so desired
        print("Writing Trial Metrics...")
        trial_prefix = f"results/trial_{trial}"
        os.makedirs(trial_prefix, exist_ok=True)
        sim.save_gif(f"{trial_prefix}/simulation.mp4")
        print(f"Saved GIF to: results/{trial_prefix}/simulation.mp4")
        metric_tracker.save_to_csv(f"{trial_prefix}/metrics.csv")
        print(f"Saved metrics to: results/{trial_prefix}/metrics.csv")
        graph_factory.plot_entropy(metric_tracker._entropy_log, save_path=f"{trial_prefix}/entropy_plot.png")
        graph_factory.plot_avg_displacement(metric_tracker._avg_displacement_log, save_path=f"{trial_prefix}/avg_displacement_plot.png")
        t_end = time()
        print(f"Trial runtime: {t_end - t_start}\n")

    stats = metric_tracker.get_aggregate_stats()

    graph_factory.plot_aggregated_metric(
        mean=stats["entropy_mean"],
        std=stats["entropy_std"],
        label="Entropy",
        color="orange",
        ylabel="Entropy",
        title="Mean Entropy Over Time Across Trials",
        save_path="results/entropy_aggregated.png"
    )

    graph_factory.plot_aggregated_metric(
        mean=stats["displacement_mean"],
        std=stats["displacement_std"],
        label="Displacement",
        color="blue",
        ylabel="Avg Displacement",
        title="Mean Displacement Over Time Across Trials",
        save_path="results/displacement_aggregated.png"
    )


if __name__ == "__main__":
    start = time()
    simulate()
    end = time()
    print(f"Runtime: {end - start}")