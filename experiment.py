"""Run both model scenarios, save plots/CSVs, and check timestep sensitivity."""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

# Use a non-interactive backend to save charts on servers without a display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model import Config, NAMES, run, metrics


def plot_states(t, states, out, title):
    """Plot the declared stance of each agent as it evolves through time."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i in range(states.shape[1]):
        ax.plot(t, states[:, i], label=NAMES[i], lw=2)
    ax.set(
        xlabel="Model time (arbitrary units)",
        ylabel="Reported stance (-1 to +1)",
        title=title,
        ylim=(-1.05, 1.05),
    )
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    """Read command-line parameters and execute baseline/evidence comparisons."""
    # Expose the primary model coefficients as optional command-line arguments.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents", type=int, choices=[3, 4, 5], default=5)
    ap.add_argument("--epsilon", type=float, default=0.35)
    ap.add_argument("--diffusion", type=float, default=0.8)
    ap.add_argument("--advection", type=float, default=0.35)
    ap.add_argument("--reaction", type=float, default=0.12)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--duration", type=float, default=24)
    ap.add_argument("--output", default="results")
    a = ap.parse_args()

    # Bundle simulation settings and make sure the results directory exists.
    cfg = Config(
        n=a.agents, epsilon=a.epsilon, diffusion=a.diffusion,
        advection=a.advection, reaction=a.reaction,
        dt=a.dt, duration=a.duration,
    )
    dest = Path(a.output)
    dest.mkdir(parents=True, exist_ok=True)

    # Use identical initial states and parameters in both scenarios.
    # Only the evidence injection switch differs between the runs.
    runs = {}
    for name, intervention in [("baseline", False), ("evidence", True)]:
        t, u, terms, W, Q = run(cfg, intervention=intervention)
        runs[name] = (t, u)

        # Save a time-series PNG and CSV containing all agent trajectories.
        plot_states(t, u, dest / f"{name}.png",
                    f"Graph reaction–diffusion: {name}")
        with (dest / f"{name}.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", *NAMES[:cfg.n]])
            writer.writerows(zip(t, *u.T))

        # Display summary statistics for comparison in the terminal.
        print(name, metrics(u))

    # Compute the distance between scenario state vectors at each time.
    t, baseline = runs["baseline"]
    _, evidence = runs["evidence"]
    difference = np.linalg.norm(evidence - baseline, axis=1)

    # Highlight the exact time interval in which evidence was injected.
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(t, difference, label="||evidence - baseline||")
    ax.axvspan(cfg.evidence_start, cfg.evidence_end,
               color="gray", alpha=0.15, label="Evidence injected")
    ax.set(xlabel="Model time",
           ylabel="Euclidean state difference",
           title="Effect of external evidence")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(dest / "intervention.png", dpi=160)
    plt.close(fig)

    # Report the magnitude of the final evidence-induced change.
    print("Final intervention effect:", float(difference[-1]))

    # Halve the timestep: similar results suggest numerical consistency.
    # This check is not a proof of correctness or empirical validity.
    fine = replace(cfg, dt=cfg.dt / 2)
    _, fine_u, *_ = run(fine, True)
    error = np.linalg.norm(evidence[-1] - fine_u[-1])
    print("Time-refinement final-state difference:", float(error))
    print("Outputs:", dest.resolve())


if __name__ == "__main__":
    main()
