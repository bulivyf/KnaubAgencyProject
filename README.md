# From Differential Equations to Agentic AI
## A guided graph diffusion–transport–reaction experiment (3–5 agents)

**Purpose.** This runnable Python experiment takes the *structure* of a differential equation studied by Karl R. Knaub and gives each component a tangible, testable interpretation in a network of communicating agents. You can adjust diffusion, directional transfer, local reaction, and evidence, then observe how agents' reported **stances** evolve over simulated time. The experiment illustrates mathematical mechanisms; **it does not implement real LLMs, prove that thoughts diffuse like heat, or demonstrate comprehension**.

> **Key question:** Can a reduced mathematical model help predict how a group responds to communication and an external evidence intervention—and distinguish consensus from evidence-driven changes?

## 1. Origin and transformation

Knaub-type continuous-space equation:

$$
\frac{\partial u}{\partial t}=\varepsilon^2\frac{\partial^2u}{\partial x^2}+\varepsilon g(u)\frac{\partial u}{\partial x}+h(u).
$$

Here the state $u(x,t)$ depends on **position** $x$ and **time** $t$. Our agents instead occupy discrete nodes in a graph, so we use $u_i(t)$, where $i$ is an agent index. The resulting system is a **set of coupled ordinary differential equations**:

$$
\boxed{\frac{du_i}{dt}=\underbrace{\varepsilon^2 D\sum_j W_{ij}(u_j-u_i)}_{\mathrm{diffusion}}+\underbrace{\varepsilon A\left(\sum_j Q_{ji}u_j-\sum_j Q_{ij}u_i\right)}_{\mathrm{directed\ transfer}}+\underbrace{k(u_i-u_i^3)}_{\mathrm{local\ reaction}}+\underbrace{e_i(t)}_{\mathrm{evidence}}.}
$$

This is an **inspired graph analogue**, *not* a mathematically equivalent transformation of the original PDE in general:

1. **Spatial curvature → graph differences.** The finite-difference approximation $u_{xx}(x_i)\approx(u_{i-1}-2u_i+u_{i+1})/\Delta x^2$ can be written as a sum of neighboring differences. A weighted graph generalizes those differences to any connected agents: $\sum_j W_{ij}(u_j-u_i)$.
2. **Advection → directed graph flow.** A spatial first derivative associated with directional physical transport motivates an *inflow minus outflow* term using a directed matrix $Q$. It is a separately chosen network transport law, **not** a direct discretization of the $g(u)u_x$ term for arbitrary graphs.
3. **General reaction → bistable cubic.** Choose $h(u)=k(u-u^3)$ so $u=-1$ and $u=+1$ are stable local states for $k>0$, while $u=0$ is unstable. This deliberately **reinforces polarized stances**; it is not an evidence-based inference engine.
4. **New evidence input.** Add $e_i(t)$ as a controlled external intervention (absent as a separate term in the original PDE).
5. **Scale factors.** Preserve $\varepsilon^2$ on diffusion and $\varepsilon$ on transport for analogy and experimentation; there is no established physical reason the same asymptotic scaling governs agent conversations.

**Interpretation:** $u_i=-1$ indicates full opposition to a proposition, $u_i=0$ neutral, and $u_i=+1$ full support. These are **synthetic numerical positions, not calibrated probabilities, actual opinions, attention, knowledge or neural activations**. In this model the five role names are labels on nodes; the roles do not implement different reasoning policies.

## 2. Meet the agents and their connections

Default nodes in index order: **Observer (0)**, **Modeler (1)**, **Skeptic (2)**, **Validator (3)** and **Integrator (4)**. With `--agents 3` or `--agents 4`, the first three or four names are used.

Initial states for five agents: `[-0.8, -0.4, 0.0, +0.4, +0.8]`. The same spacing rule (`np.linspace(-0.8, 0.8, n)`) is used for 3–5 agents. These are deliberate, contrasting starting assumptions, not measurements.

**Undirected influence weights `W`:** a chain 0—1—2—3—4, with adjacent weights **1.0**. With five agents an additional weaker connection 0—4 has weight **0.35**. `W` is symmetric and nonnegative.

**Directed transfers `Q`:** 0 → 1 → 2 → 3 → 4, with weight **1.0** per arrow. With 3 or 4 agents the chain is shorter. There is no reverse directed edge, although diffusion runs both ways along the undirected links.

## 3. Parameters: What does a specific value mean?

| Parameter | Default | Meaning in this simulation | What happens when increased? |
|---|---:|---|---|
| `--agents` | `5` | Number of nodes; allowed values 3, 4, 5 | More nodes and relationships, not necessarily higher intelligence |
| `--epsilon` ($\varepsilon$) | `0.35` | Dimensionless multiplier on diffusion and transfer | Diffusion grows **quadratically** in $\varepsilon$; transfer grows **linearly** |
| `--diffusion` ($D$) | `0.8` | Additional strength of bidirectional neighbor influence | Connected stances tend to become more alike (all else equal) |
| `W_ij` | `1.0` adjacent; `0.35` extra | Relative weight of *undirected* relationship $i\leftrightarrow j$ | A larger edge weight strengthens their mutual influence |
| `--advection` ($A$) | `0.35` | Additional strength of *conservative directed state transfer* | Shifts numerical state along arrows more strongly; may create asymmetry |
| `Q_ij` | `1.0` directed chain | Transfer coefficient from node $i$ to node $j$ | More state flows down that edge in the numerical model |
| `--reaction` ($k$) | `0.12` | Magnitude of cubic local reinforcement $k(u-u^3)$ | Agents tend more strongly toward the signed extremes $-1$ or $+1$ |
| `dt` (`--dt`) | `0.02` | Numerical time increment in explicit Euler | Larger steps are faster but may be less accurate or unstable |
| `duration` (`--duration`) | `24.0` | Total simulated time, in arbitrary units | Runs longer; **not** 24 seconds or 24 messages |
| evidence start/end | `6.0`, `10.0` | Fixed intervention interval `[6,10)` | Evidence applies only in this time window |
| evidence strength | `0.14` | Strength of positive evidence pushing selected agent toward `+1` | Faster adjustment toward +1, not an empirical probability |
| evidence agent | `0` | Node receiving evidence: Observer | Change in `model.py` configuration to target a different node |

**Effective strengths matter more than raw parameters.** For the default $\varepsilon=0.35$, $D=0.8$, and $A=0.35$:

- Effective diffusion factor $\varepsilon^2D = 0.35^2\times 0.8=\mathbf{0.098}$ per unit of graph difference and model time.
- Effective directed-transfer factor $\varepsilon A=0.35\times 0.35=\mathbf{0.1225}$ per unit of graph-transfer expression and model time.
- Local reinforcement factor is $k=\mathbf{0.12}$ multiplied by $(u-u^3)$.
- While evidence is active, its sole nonzero term is $e_0(t)=\mathbf{0.14}(1-u_0)$.

**These numbers cannot be ranked by comparing 0.098, 0.1225 and 0.12 alone.** Each factor multiplies a different expression, which depends on the current states, connectivity and time. For instance, the local reaction for an agent at $u=0.5$ is $0.12(0.5-0.5^3)=\mathbf{0.045}$ per model-time unit. The evidence input for an Observer at $u=-0.8$ during the intervention is $0.14(1-(-0.8))=\mathbf{0.252}$ per model-time unit.

### A one-step calculation (diffusion)

Suppose B has $u_B=0.2$, adjacent to A with $u_A=-0.6$ and C with $u_C=+0.8$, all edge weights 1. Then its neighbor-difference sum is:

$$(-0.6-0.2)+(0.8-0.2)=\mathbf{-0.2}.$$

At the default effective diffusion factor `0.098`, diffusion alone contributes $0.098\times(-0.2)=\mathbf{-0.0196}$ per model-time unit. With `dt=0.02`, B's diffusion-only next-step change would be $0.02\times(-0.0196)=\mathbf{-0.000392}$. The full solver also computes transport, local reaction and evidence before taking the step.

### Important subtlety about information and energy

The directed term conserves the **sum of signed state values** in a closed network when acting alone; the symmetric diffusion term does too. The code tests those properties. This is a mathematical conservation law **inside the selected numerical model**, not a claim that communication conserves information, human belief or physical energy. Unlike mass, real information can be copied without subtracting it from the sender. If the goal changes to *persuasion* or *knowledge transfer*, a directed receiving-agent adjustment may be more suitable than this conservative-flow formulation.

## 4. Precisely what Python does

1. **Read arguments** in `experiment.py`; construct `Config` with the defaults above unless overridden.
2. **Build network matrices** (`topology()` in `model.py`): symmetric undirected `W` and asymmetric directed `Q`.
3. **Initialize agents** (`initial_state()`): equally spaced starting stances from `-0.8` to `+0.8`.
4. **Run two scenarios using identical initial conditions:** `baseline` (no new evidence) and `evidence` (Observer gets positive input from model time 6 inclusive to 10 exclusive).
5. **At every timestep**, calculate each component **from the same old state**:
   - Diffusion: `epsilon**2 * diffusion * (W @ u - W.sum(axis=1)*u)`.
   - Directed transfer: `epsilon * advection * (Q.T @ u - Q.sum(axis=1)*u)`.
   - Local reaction: `reaction * (u - u**3)`.
   - Evidence when active: `evidence_strength * (1-u[0])` for the Observer only.
6. **Advance with explicit Euler:** $\mathbf u^{n+1}=\mathbf u^n+\Delta t\,(\mathbf f_D+\mathbf f_A+\mathbf f_R+\mathbf f_E)$. `dt=0.02` over `duration=24` gives **1,200 updates**.
7. **Check values** after each update: invalid/nonfinite states or excursions outside about `[-1,+1]` generate an error. These are safeguards, **not a formal proof of numerical stability**.
8. **Export** one time-series CSV and one line chart for each scenario, plus a third chart of the difference between scenario trajectories.
9. **Print summary metrics** and compare the evidence run's final state to a rerun at half the timestep as an initial integration-error check.

There are **no live language model API requests**, no generated agent messages, no learning of graph weights from data, and no claims of emergent consciousness or understanding.

## 5. Installation and commands

Requires Python 3.10+ with NumPy and Matplotlib:

```bash
python -m pip install -r requirements.txt
python experiment.py --agents 5 --output results
python -m unittest discover -s tests -v
```

Explore 3 agents and change network influence:

```bash
python experiment.py --agents 3 --diffusion 0.8 --advection 0 --reaction 0.12 --output results_3
```

Isolate terms to see what each one contributes:

```bash
# Diffusion alone: tends toward the initial network-average stance
python experiment.py --diffusion 0.8 --advection 0 --reaction 0 --output results_diffusion

# Transport alone: redirects state along the one-way chain
python experiment.py --diffusion 0 --advection 0.35 --reaction 0 --output results_transport

# Reaction alone: maintains/reinforces opposing stances
python experiment.py --diffusion 0 --advection 0 --reaction 0.12 --output results_reaction

# Weaker interaction effects
python experiment.py --epsilon 0.2 --output results_weak

# Finer numerical resolution
python experiment.py --dt 0.01 --output results_fine
```

**Caution for isolated terms:** The program always runs its positive-evidence comparison, so interpret `baseline.csv` for a clean diffusion-only/transport-only/reaction-only trajectory. `evidence.csv` intentionally includes an external forcing term even when the other processes are disabled.

The executable CLI currently changes only the parameters shown in `--help`. To change `evidence_start`, `evidence_end`, `evidence_strength`, `evidence_agent`, or matrices `W`/`Q`, edit `Config` or `topology()` in `model.py`. For example, `--reaction` does **not** change the form of $h(u)$; it only changes its coefficient.

## 6. How to read the outputs

| Output | How to interpret it |
|---|---|
| `baseline.png` | Each colored line is one agent's stance trajectory without injected evidence. Look for convergence, divergence, or persistent separation. |
| `evidence.png` | Same initial states and graph, but with evidence at Observer between model times 6 and 10. Look for responses during and after the intervention. |
| `baseline.csv` / `evidence.csv` | Raw columns `time, Observer, Modeler, Skeptic, Validator, Integrator` (fewer names for fewer agents). Use these for independent calculations. |
| `intervention.png` | $\|\mathbf u_{\rm evidence}(t)-\mathbf u_{\rm baseline}(t)\|_2$: the **Euclidean distance between whole-network states**, not an accuracy or learning score. Shaded region indicates evidence injection. |
| Console `final_mean` | Arithmetic average of final signed stances. Negative = average opposition; positive = average support. This is **not** probability of truth. |
| Console `final_disagreement` | Maximum minus minimum final agent stance; higher = greater spread of values. Does not measure semantic argument quality. |
| Console `total_change` | Euclidean distance from initial to final network state. It means magnitude of change, **not improvement**. |
| Console `Final intervention effect` | Final Euclidean distance between evidence and baseline states. Measures model sensitivity to the intervention, not causation in real AI agents. |
| Console `Time-refinement final-state difference` | Distance between two evidence simulations with `dt` and `dt/2`, an initial indicator of numerical temporal error. |

### Verified default run (five agents)

The current implementation was executed using `python experiment.py --agents 5 --output results` with its default parameters. Results:

| Metric | Baseline | With evidence | Interpretation |
|---|---:|---:|---|
| Final mean | **-0.366418** | **+0.170581** | External input shifts average stance upward by about **+0.537**. |
| Final disagreement range | **0.384893** | **0.539140** | The evidence run ends **more dispersed**, not more consensual. |
| Total change from initial vector | **1.470798** | **1.083910** | The final evidence state moved **less far overall** from initialization; this is not an indicator of correctness. |

- Final *between-scenario* state-vector difference: **1.221614**.
- Evidence run's final-state difference when halving timestep from `0.02` to `0.01`: **0.000435123**.
- Five unit tests passed (topology, conservation in two isolated models, evidence effect, and improved timestep consistency).

**What we can conclude:** The mathematical agent network is responsive to externally injected evidence; evidence can shift average stance while leaving (or increasing) disagreement. The output varies with graph structure and interaction strengths. These are reproducible observations from this particular synthetic setup.

**What we cannot conclude:** That these agents learned, that the evidence was true, that consensus predicts correctness, that information behaves physically like energy, or that these trajectories forecast what real LLM agents will do. A small difference between two time resolutions does not establish validity of the **agent interpretation**.

## 7. Tangible application: collaborative sensor-fault assessment

A plausible future application is a **multi-agent sensor-validation workflow**. Suppose a drone reports conflicting altitude observations:

- **Observer:** extracts and records sensor samples.
- **Modeler:** fits a systematic-bias correction and predicts sensor readings.
- **Skeptic:** challenges assumptions, such as extrapolation outside measured heights.
- **Validator:** checks held-out observations and uncertainty bands.
- **Integrator:** prepares a final assessment with explicit supporting evidence and unresolved questions.

The present numerical simulation can explore *network design choices* before integrating language models. For example, which communication edges spread a new observation quickly? Under which settings does an initial disagreement disappear, persist, or amplify? Does the system change when evidence is received by the Observer rather than the Validator?

For the system to become operational, replace synthetic stances with reproducible agent-level measurements (e.g., a declared probability or structured assessment with a rubric); collect actual messages and decisions; represent evidence provenance; and compare the predicted updates with observed updates. Prefer a **directed influence/assimilation rule** if communicating information should not subtract it from the sender.

### Practical decision aid (hypotheses to test, not validated advice)

| Observation in the simulation | Possible system design question | Next evidence needed |
|---|---|---|
| Diffusion yields rapid agreement | Are agents becoming too similar without independently checking evidence? | Compare accuracy with/without independent validation |
| High reaction strength preserves opposing groups | Does persistence reflect useful skepticism or unjustified rigidity? | Track which objections survive new ground-truth observations |
| Directional transport changes the network's outcome | Does routing evidence through particular roles matter? | Randomize recipients/topologies in actual agent tasks |
| A single evidence injection changes network mean | How responsive is the system to a verified external finding? | Test verified vs misleading evidence and recovery |
| Mean shifts while disagreement rises | Should the system surface dissent instead of forcing consensus? | Measure correctness, calibration and minority-report value |

## 8. Recommended evaluation roadmap

1. **Numerical validity (already partially tested):** isolate each component; test conservation as appropriate; refine `dt`; add analytic cases and stability bounds.
2. **Parameter sensitivity:** sweep $\varepsilon$, $D$, $A$, $k$, graph topology and the evidence recipient; record final mean, disagreement, evidence sensitivity and settling time. Changing $\varepsilon$ modifies *two* processes simultaneously, so isolate them when interpreting causal contributions.
3. **Ablation:** compare independent agents, graph diffusion only, diffusion+transport, and the complete model. Keep evidence and initial states fixed across comparisons.
4. **External validation:** define a consistent structured stance measure from real agent messages, log each exchange, and compare model-predicted states against held-out observed changes. Include simple averaging and no-interaction baselines.
5. **Outcome validity:** assess ground-truth task accuracy, uncertainty calibration, evidence traceability, cost, number of communication turns and recovery from misleading inputs. Consensus alone is insufficient.
6. **Event-driven comparison:** message-passing systems update at discrete times. Compare this continuous-time approximation with a discrete-turn graph model before assuming a PDE-inspired approach is superior.

## 9. Limitations and scientific interpretation

- The graph equation is a deliberately chosen *analogy*, not a theorem of cognitive physics.
- Five named agent roles share the same cubic update function; they do not implement role-specific independent objectives or LLM reasoning.
- The `[-1,1]` state constraint and cubic reaction are modeling assumptions.
- The conservative directed-transfer law is especially questionable as a literal representation of information sharing, since real messages are copyable and semantically interpreted.
- `epsilon` is dimensionless here, and simulated time is arbitrary; there is no calibration to conversation turns or wall-clock seconds.
- The default trajectories **do not prove metastability** in the rigorous asymptotic sense studied in Knaub's dissertation. Sustained disagreement over a short window alone is insufficient.
- Numerical agreement does not imply empirical validity. The first real research hurdle is **measurement and identification**: can these state variables and rates be estimated reliably from observed agent interactions?

## Files

- `model.py` — config, graph topology, initial states, component derivatives, explicit Euler solver, metrics.
- `experiment.py` — CLI, paired baseline/evidence runs, plots, CSVs, timestep refinement.
- `tests/test_model.py` — numerical and structural checks.
- `requirements.txt` — NumPy/Matplotlib dependencies.
- `results/` — generated graphs and time-series tables.

**Takeaway:** This experiment turns abstract PDE concepts into manipulable mathematical behaviors. Its immediate value is as an **educational and hypothesis-generation tool for multi-agent system design**. Turning it into a credible operational predictor requires actual agent observations, benchmark comparisons and validation against independently defined outcomes.


## Understanding `baseline.png`, `evidence.png`, and `intervention.png`

- **`baseline.png`** graphs the agents' assigned numerical stances over time with no external evidence. Curves are not probabilities or verified beliefs.
- **`evidence.png`** uses the same initial conditions and model but injects a synthetic positive-stance forcing term into the Observer during model time 6–10. It depicts the resulting trajectories, and can be seen as personal agent retention of the evidence (as a simulated entity).
- **`intervention.png`** plots the Euclidean distance between the baseline and evidence state vectors over time and shades the injection window.
- Evidence is modeled as `evidence_strength * (1 - Observer_stance)`. This is a numerical intervention, **not fact checking** or actual LLM inference.

