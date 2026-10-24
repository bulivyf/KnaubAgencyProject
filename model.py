"""Five-agent graph model inspired by Knaub's advection–diffusion–reaction PDE.

u_i is a declared numerical stance in [-1, +1], NOT the agent's hidden beliefs.
Diffusion tends to equalize neighbors, directed transport transfers a numerical
state quantity, reaction reinforces local stances, and intervention adds evidence.
This is a graph-based analogy, not a physical model of communication energy.
"""
from dataclasses import dataclass
import numpy as np

# Conceptual agent roles: labels only; numerical rules are shared by all agents.
NAMES = ("Observer", "Modeler", "Skeptic", "Validator", "Integrator")


@dataclass(frozen=True)
class Config:
    """All parameters are dimensionless illustrative simulation settings."""
    n: int = 5                 # Number of graph nodes (3, 4, or 5)
    epsilon: float = 0.35      # Small-parameter scaling: epsilon^2 for diffusion
    diffusion: float = 0.8     # D: symmetric neighbor-influence coefficient
    advection: float = 0.35    # A: directed numerical-state transport coefficient
    reaction: float = 0.12     # k: strength of local bistable reaction u - u^3
    dt: float = 0.02           # Requested explicit-Euler timestep
    duration: float = 24.0     # Total simulated time, in arbitrary units
    evidence_start: float = 6.0       # Start of external evidence interval
    evidence_end: float = 10.0        # End of external evidence interval
    evidence_strength: float = 0.14   # Relaxation rate toward a positive stance
    evidence_agent: int = 0           # Zero-based index of the receiving agent


def topology(n):
    """Build an undirected diffusion network W and directed transfer network Q."""
    if n not in (3, 4, 5):
        raise ValueError("Choose 3, 4, or 5 agents")

    # W connects adjacent agents in both directions with equal weights.
    W = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        W[i, i + 1] = W[i + 1, i] = 1.0

    # Five agents also have a weaker Observer <-> Integrator connection.
    if n == 5:
        W[0, 4] = W[4, 0] = 0.35

    # Q describes one-way transport: Q[i,j] flows from i to j.
    Q = np.zeros((n, n), dtype=float)
    for i in range(n - 1):
        Q[i, i + 1] = 1.0
    return W, Q


def initial_state(n):
    """Assign different initial stances evenly spanning -0.8 to +0.8."""
    return np.linspace(-0.8, 0.8, n)


def components(u, t, config, W, Q, intervention=True):
    """Calculate all four independent contributions to du/dt at time t."""
    # Graph diffusion: sum_j W[i,j] * (u[j] - u[i]); conserves sum(u)
    # when W is symmetric. The effective coefficient is epsilon^2 * D.
    diffusion = (
        config.epsilon**2 * config.diffusion
        * (W @ u - W.sum(axis=1) * u)
    )

    # Conservative directed transfer: incoming minus outgoing numerical state.
    # Caution: real information can be copied, so this is not a literal
    # physical energy or knowledge conservation law.
    directed = (
        config.epsilon * config.advection
        * (Q.T @ u - Q.sum(axis=1) * u)
    )

    # Local reaction: +/-1 are locally stable, zero is unstable for k > 0.
    # This term reinforces polarization; it does not test whether a claim is true.
    local = config.reaction * (u - u**3)

    # External evidence affects only the selected node during a specified
    # time window. The forcing declines as the receiving state approaches +1.
    evidence = np.zeros_like(u)
    if intervention and config.evidence_start <= t < config.evidence_end:
        i = config.evidence_agent
        evidence[i] = config.evidence_strength * (1 - u[i])

    # Separate contributions are returned for analysis and debugging.
    return diffusion, directed, local, evidence


def run(config=Config(), intervention=True):
    """Evolve every agent using synchronized explicit-Euler updates."""
    W, Q = topology(config.n)

    # Reject invalid user settings before attempting an integration.
    if not (config.dt > 0 and config.duration > 0
            and config.epsilon >= 0 and config.diffusion >= 0
            and config.advection >= 0 and config.reaction >= 0):
        raise ValueError("Invalid numerical parameters")
    if not 0 <= config.evidence_agent < config.n:
        raise ValueError("Evidence agent index out of range")

    # Use an integer number of steps, adjusting dt slightly if required.
    steps = int(np.ceil(config.duration / config.dt))
    dt = config.duration / steps
    times = np.linspace(0, config.duration, steps + 1)

    # Store all time steps so charts and numerical metrics can be reproduced.
    state = np.empty((steps + 1, config.n))
    state[0] = initial_state(config.n)
    contributions = np.zeros((steps, 4, config.n))

    for step in range(steps):
        # All agents read the SAME previous-time state before updating.
        terms = components(state[step], times[step], config, W, Q, intervention)
        contributions[step] = terms

        # Explicit Euler: new state = old state + dt * total rate of change.
        state[step + 1] = state[step] + dt * sum(terms)

        # Check numerical validity and the declared stance range.
        if (not np.all(np.isfinite(state[step + 1]))
                or np.max(np.abs(state[step + 1])) > 1.000001):
            raise ValueError("State exceeded bounds: reduce dt or coupling strengths")

    return times, state, contributions, W, Q


def metrics(states):
    """Summarize final average stance, disagreement, and net vector movement."""
    return {
        "final_mean": float(states[-1].mean()),
        "final_disagreement": float(np.ptp(states[-1])),
        "total_change": float(np.linalg.norm(states[-1] - states[0])),
    }
