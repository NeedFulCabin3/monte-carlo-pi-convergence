import time
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np


def get_user_inputs() -> Tuple[int, int | None]:
    """Prompts and validates simulation parameters from terminal input."""
    print("=== Monte Carlo Pi Estimator Setup ===")
    
    # 1. Prompt for Sample Count
    while True:
        try:
            raw_samples = input("Enter number of samples (default 1000000): ").strip()
            if not raw_samples:
                num_samples = 1_000_000
                break
            num_samples = int(raw_samples)
            if num_samples <= 0:
                print("Please enter a positive integer greater than 0.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # 2. Prompt for Random Seed
    while True:
        raw_seed = input("Enter random seed (or press Enter for default 42): ").strip()
        if not raw_seed:
            seed = 42
            break
        if raw_seed.lower() in ("none", "null"):
            seed = None
            break
        try:
            seed = int(raw_seed)
            break
        except ValueError:
            print("Invalid seed. Enter an integer or leave blank.")

    return num_samples, seed


def estimate_pi_monte_carlo(
    num_samples: int,
    seed: int | None = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimates Pi using vectorized Monte Carlo simulation."""
    if seed is not None:
        np.random.seed(seed)

    # 1. Uniform random 2D coordinates in [0, 1)
    x = np.random.uniform(0.0, 1.0, size=num_samples)
    y = np.random.uniform(0.0, 1.0, size=num_samples)

    # 2. Boolean mask: True if r^2 <= 1.0
    inside_circle = (x**2 + y**2) <= 1.0

    # 3. Vectorized cumulative count and pi estimates
    cumulative_inside = np.cumsum(inside_circle)
    samples_count = np.arange(1, num_samples + 1)
    pi_estimates = 4.0 * cumulative_inside / samples_count

    # 4. Error calculation
    abs_errors = np.abs(pi_estimates - np.pi)

    return samples_count, pi_estimates, abs_errors


def plot_convergence(
    samples_count: np.ndarray,
    pi_estimates: np.ndarray,
    abs_errors: np.ndarray,
) -> None:
    """Plots convergence and error decay without LaTeX string parser issues."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top Plot: Pi Estimate
    truth_label = f"Truth Standard (pi = {np.pi:.6f})"
    ax1.plot(
        samples_count,
        pi_estimates,
        label="Monte Carlo Estimate",
        color="#1f77b4",
        linewidth=1.2,
    )
    ax1.axhline(
        y=np.pi,
        color="#d62728",
        linestyle="--",
        linewidth=1.5,
        label=truth_label,
    )
    ax1.set_ylabel("Estimated Pi Value", fontsize=11)
    ax1.set_title("Monte Carlo Pi Estimation Convergence", fontsize=13, pad=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Bottom Plot: Error Decay
    ax2.loglog(
        samples_count,
        abs_errors,
        label="Absolute Error |Estimate - Pi|",
        color="#2ca02c",
        linewidth=1.2,
    )

    theoretical_decay = 1.0 / np.sqrt(samples_count)
    ax2.loglog(
        samples_count,
        theoretical_decay,
        label="Theoretical Error Rate O(1/sqrt(N))",
        color="#7f7f7f",
        linestyle=":",
        linewidth=1.5,
    )

    ax2.set_xlabel("Number of Samples (N)", fontsize=11)
    ax2.set_ylabel("Absolute Error (Log Scale)", fontsize=11)
    ax2.set_title("Error Decay Rate Analysis", fontsize=13, pad=10)
    ax2.grid(True, which="both", linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def main() -> None:
    # Get configuration interactively from terminal
    num_samples, seed = get_user_inputs()

    print(f"\nStarting simulation with {num_samples:,} samples...")
    start_time = time.perf_counter()

    try:
        samples_count, pi_estimates, abs_errors = estimate_pi_monte_carlo(
            num_samples=num_samples, seed=seed
        )
    except Exception as err:
        print(f"Simulation error: {err}")
        return

    elapsed_time = time.perf_counter() - start_time

    print("\n--- Simulation Results ---")
    print(f"Execution Time     : {elapsed_time:.4f} seconds")
    print(f"Final Estimate     : {pi_estimates[-1]:.7f}")
    print(f"Truth Standard     : {np.pi:.7f}")
    print(f"Final Absolute Error: {abs_errors[-1]:.7f}\n")

    plot_convergence(samples_count, pi_estimates, abs_errors)


if __name__ == "__main__":
    main()