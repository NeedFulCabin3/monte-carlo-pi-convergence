# Monte Carlo Pi Convergence

> Vectorized Monte Carlo simulation engine tracking empirical pi estimation and theoretical error convergence.

## Overview

Traditional pedagogical demonstrations of Monte Carlo methods often suffer from severe performance degradation caused by explicit Python loops evaluating millions of coordinate points individually. When scaling sample sizes to improve statistical confidence, standard procedural iterations introduce massive interpreter overhead, choking execution pipelines. This tool solves that friction by utilizing vectorized array operations and continuous memory buffers to execute stochastic simulations and convergence analysis efficiently.

## How It Works

The execution pipeline operates through four core stages:
1. **Vectorized Sampling:** Generates uniform 2D random coordinates in $[0, 1)$ across large sample sizes using NumPy's C-optimized Mersenne Twister engine.
2. **Boolean Masking:** Evaluates geometric inclusion within a quarter circle via vectorized array comparison `(x**2 + y**2) <= 1.0` without loops.
3. **Cumulative Aggregation:** Computes running counts of interior points via `np.cumsum()` and derives sequential pi estimates across every sample tier.
4. **Error Decay Analysis:** Calculates absolute deviations against `math.pi` and compares empirical variance against the theoretical $O(1/\sqrt{N})$ decay curve.

## Key Features

* Interactive terminal configuration supporting custom sample sizes and reproducible random seeds.
* Full NumPy vectorization eliminating explicit Python `for` loops over coordinate pairs.
* Dual-axis Matplotlib visualization mapping convergence trajectories and logarithmic error decay.
* Structured error handling and typed function signatures.

## Tech Stack & Core Dependencies

* **Python 3.10+:** Core runtime environment leveraging modern type hinting annotations (`tuple`, `int | None`).
* **NumPy:** Handles high-performance array allocation, uniform random variates, cumulative sums, and vectorized mathematical operations.
* **Matplotlib:** Renders dual-panel figures displaying convergence trends and log-scale error decay rates.
* **Python Standard Library:** Utilizes `time` for performance profiling and `typing` for strict static type definitions.

## Environment & Web-Based Quick Start

### GitHub Codespaces
1. Open the repository in a new GitHub Codespace.
2. Install dependencies from the terminal:
   ```bash
   pip install numpy matplotlib
   ```
3. Execute the simulation:
    ```bash
    python main.py
    ```

### Local Virtual Environment (venv)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install numpy matplotlib
python main.py
```

## Repository Structure

```bash
monte-carlo-pi-convergence/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
├── main.py
└── requirements.txt
```

## Roadmap

**[ ] Type Stubs and Static Analysis:** Integrate mypy and strict type checking configurations into the codebase.

**[ ] Multiprocessing Chunking:** Implement multi-core parallel worker pools for scaling sample sizes beyond ten million points.

**[ ] Structured Logging:** Replace terminal print statements with structured JSON logging for telemetry and execution tracking.