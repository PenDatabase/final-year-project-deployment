"""Forecast request load and select a least-cost EC2 configuration.

The input resource table must contain these columns:
    name, capacity_rps, cost_per_hour

The model follows the notebook's forecasting setup: a StandardScaler, a
168-hour lookback, and a one-week recursive forecast. The load balancer cost
is charged once when the selected configuration contains more than one
instance.

Methods compared (all on the same required capacity, catalogue and LB cost):
    greedy    - cost-per-capacity constructive heuristic (original baseline)
    genetic   - the same evolutionary loop and repair as the memetic algorithm,
                but WITHOUT local search (isolates the value of local search)
    exact     - dynamic-programming optimum (the true least-cost configuration,
                used as the reference for the optimality gap)
    memetic   - proposed method (evolutionary search + repair + local search)

Every method is timed. For the two evolutionary methods the time and
generation at which the final best cost was first reached are also recorded.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


LOOKBACK_HOURS = 24 * 7
FORECAST_HOURS = 24 * 7


@dataclass(frozen=True)
class ResourceType:
    name: str
    capacity_rps: float
    cost_per_hour: float


@dataclass
class Configuration:
    counts: list[int]
    capacity_rps: float
    hourly_cost: float
    instance_cost: float
    load_balancer_cost: float
    # Timing information (filled in by the methods / timed_run)
    runtime_seconds: float = 0.0
    generations_to_best: int | None = None
    time_to_best_seconds: float | None = None

    @property
    def total_instances(self) -> int:
        return sum(self.counts)


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_requests(path: str | Path) -> np.ndarray:
    data = pd.read_csv(path)
    if "Requests" not in data.columns:
        raise ValueError("Request data must contain a 'Requests' column.")
    values = pd.to_numeric(data["Requests"], errors="raise").to_numpy(dtype=float)
    if len(values) <= LOOKBACK_HOURS:
        raise ValueError("At least 169 hourly observations are required.")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Requests must contain finite, non-negative numbers.")
    return values


def load_resources(path: str | Path) -> list[ResourceType]:
    data = pd.read_csv(path)
    required = {"name", "capacity_rps", "cost_per_hour"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Resource table is missing columns: {sorted(missing)}")
    resources = [
        ResourceType(str(row.name), float(row.capacity_rps), float(row.cost_per_hour))
        for row in data.itertuples(index=False)
    ]
    if not resources:
        raise ValueError("Resource table contains no instance types.")
    if any(r.capacity_rps <= 0 or r.cost_per_hour < 0 for r in resources):
        raise ValueError("Capacity must be positive and cost cannot be negative.")
    return resources


# --------------------------------------------------------------------------
# Forecasting (unchanged)
# --------------------------------------------------------------------------
def build_model(
    values: np.ndarray,
    epochs: int = 5,
    progress_callback: Callable[[str, str], None] | None = None,
):
    from sklearn.preprocessing import StandardScaler

    try:
        from tensorflow.keras.callbacks import Callback
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.models import Sequential
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for LSTM forecasting.") from exc

    if len(values) < LOOKBACK_HOURS * 2:
        raise ValueError("At least 336 observations are required to train the model.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    train_size = min(24 * 7 * 4, len(scaled) - LOOKBACK_HOURS)
    train = scaled[:train_size]
    x, y = create_rnn_dataset(train, LOOKBACK_HOURS)
    x = x.reshape((x.shape[0], 1, x.shape[1]))

    model = Sequential([LSTM(256, input_shape=(1, LOOKBACK_HOURS)), Dense(1)])
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mse"])

    class TrainingProgress(Callback):
        def on_epoch_end(self, epoch, logs=None):
            if progress_callback:
                progress_callback("training", f"Training epoch {epoch + 1} of {epochs}")

    model.fit(x, y, epochs=epochs, batch_size=1, verbose=0, callbacks=[TrainingProgress()])
    return model, scaler, scaled


def create_rnn_dataset(data: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for end in range(lookback, len(data)):
        x.append(data[end - lookback:end, 0])
        y.append(data[end, 0])
    return np.asarray(x), np.asarray(y)


def forecast_week(model: Any, scaler: Any, scaled_values: np.ndarray) -> np.ndarray:
    current = scaled_values[-LOOKBACK_HOURS:, 0].copy()
    predictions = []
    for _ in range(FORECAST_HOURS):
        features = current[-LOOKBACK_HOURS:].reshape(1, 1, LOOKBACK_HOURS)
        prediction = float(model.predict(features, verbose=0)[0, 0])
        predictions.append(prediction)
        current = np.append(current, prediction)
    return scaler.inverse_transform(np.asarray(predictions).reshape(-1, 1)).ravel().clip(0)


# --------------------------------------------------------------------------
# Configuration helpers
# --------------------------------------------------------------------------
def make_configuration(
    counts: list[int], resources: list[ResourceType], load_balancer_cost: float
) -> Configuration:
    instance_cost = sum(c * r.cost_per_hour for c, r in zip(counts, resources))
    capacity = sum(c * r.capacity_rps for c, r in zip(counts, resources))
    lb_cost = load_balancer_cost if sum(counts) > 1 else 0.0
    return Configuration(counts, capacity, instance_cost + lb_cost, instance_cost, lb_cost)


def timed_run(fn: Callable[[], Configuration], repeats: int = 1) -> Configuration:
    """Run ``fn`` ``repeats`` times and store the mean wall-clock time.

    The configuration returned is from the last run. For deterministic
    (seeded) methods every run returns the same configuration, so repeating
    only improves the stability of the timing measurement.
    """
    repeats = max(1, repeats)
    durations = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        durations.append(time.perf_counter() - start)
    result.runtime_seconds = sum(durations) / len(durations)
    return result


# --------------------------------------------------------------------------
# Baseline 1: greedy cost-per-capacity heuristic (original)
# --------------------------------------------------------------------------
def greedy_baseline(required_rps: float, resources: list[ResourceType], load_balancer_cost: float) -> Configuration:
    counts = [0] * len(resources)
    capacity = 0.0
    order = sorted(range(len(resources)), key=lambda i: resources[i].cost_per_hour / resources[i].capacity_rps)
    while capacity < required_rps:
        index = order[0]
        counts[index] += 1
        capacity += resources[index].capacity_rps
    return make_configuration(counts, resources, load_balancer_cost)


# --------------------------------------------------------------------------
# Baseline 3: exact optimum by dynamic programming
# --------------------------------------------------------------------------
def exact_optimal(required_rps: float, resources: list[ResourceType], load_balancer_cost: float) -> Configuration:
    """Least-cost configuration, found exactly.

    This is an unbounded covering-knapsack dynamic programme over integer
    capacity units (1 RPS). The load-balancer charge applies only when more
    than one instance is used, so the state also tracks whether the
    configuration holds 1 or 2-or-more instances.

    Capacities are floored to whole RPS, which keeps every returned
    configuration feasible. The result is exactly optimal when capacities are
    whole numbers, as in the EC2 table used in this study.
    Memory/time are O(required_rps x number_of_types).
    """
    target = max(1, math.ceil(required_rps))
    caps = [max(1, int(math.floor(r.capacity_rps))) for r in resources]
    inf = float("inf")
    # best[k][c]: min instance cost to reach capacity >= c with k instances
    # (k = 1, or 2 meaning "two or more")
    best = [[inf] * (target + 1) for _ in range(3)]
    parent: list[list[tuple[int, int, int] | None]] = [[None] * (target + 1) for _ in range(3)]
    best[0][0] = 0.0
    for c in range(1, target + 1):
        for k in (1, 2):
            sources = (0,) if k == 1 else (1, 2)
            for i, resource in enumerate(resources):
                prev_c = max(0, c - caps[i])
                for pk in sources:
                    prev = best[pk][prev_c]
                    if prev == inf:
                        continue
                    value = prev + resource.cost_per_hour
                    if value < best[k][c]:
                        best[k][c] = value
                        parent[k][c] = (i, pk, prev_c)

    single = best[1][target]
    multi = best[2][target] + load_balancer_cost
    k = 1 if single <= multi else 2
    if best[k][target] == inf:
        raise RuntimeError("No feasible configuration found.")
    counts = [0] * len(resources)
    c = target
    while k != 0:
        i, pk, prev_c = parent[k][c]
        counts[i] += 1
        k, c = pk, prev_c
    return make_configuration(counts, resources, load_balancer_cost)


# --------------------------------------------------------------------------
# Repair and local search (shared by the evolutionary methods)
# --------------------------------------------------------------------------
def repair(counts: list[int], required_rps: float, resources: list[ResourceType], load_balancer_cost: float) -> list[int]:
    counts = [max(0, int(value)) for value in counts]
    while sum(c * r.capacity_rps for c, r in zip(counts, resources)) < required_rps:
        candidates = []
        for index, resource in enumerate(resources):
            next_counts = counts.copy()
            next_counts[index] += 1
            candidate = make_configuration(next_counts, resources, load_balancer_cost)
            candidates.append((candidate.hourly_cost, candidate.capacity_rps, next_counts))
        counts = min(candidates, key=lambda item: (item[0], item[1]))[2]
    return counts


def local_search(counts: list[int], required_rps: float, resources: list[ResourceType], load_balancer_cost: float) -> list[int]:
    current = make_configuration(counts, resources, load_balancer_cost)
    improved = True
    while improved:
        improved = False
        for index, count in enumerate(current.counts):
            if count == 0:
                continue
            candidate_counts = current.counts.copy()
            candidate_counts[index] -= 1
            candidate = make_configuration(candidate_counts, resources, load_balancer_cost)
            if candidate.capacity_rps >= required_rps and candidate.hourly_cost < current.hourly_cost:
                current = candidate
                improved = True
                break
        if improved:
            continue
        for source, source_count in enumerate(current.counts):
            if source_count == 0:
                continue
            for target in range(len(resources)):
                if source == target:
                    continue
                candidate_counts = current.counts.copy()
                candidate_counts[source] -= 1
                candidate_counts[target] += 1
                candidate = make_configuration(candidate_counts, resources, load_balancer_cost)
                if candidate.capacity_rps >= required_rps and candidate.hourly_cost < current.hourly_cost:
                    current = candidate
                    improved = True
                    break
            if improved:
                break
    return current.counts


# --------------------------------------------------------------------------
# Evolutionary search: memetic (with local search) and genetic (without)
# --------------------------------------------------------------------------
def _evolutionary_search(
    required_rps: float,
    resources: list[ResourceType],
    load_balancer_cost: float,
    population_size: int,
    generations: int,
    seed: int,
    use_local_search: bool,
) -> Configuration:
    start = time.perf_counter()
    rng = random.Random(seed)
    upper_bound = max(1, int(np.ceil(required_rps / min(r.capacity_rps for r in resources))) + 1)

    def refine(counts: list[int]) -> list[int]:
        counts = repair(counts, required_rps, resources, load_balancer_cost)
        if use_local_search:
            counts = local_search(counts, required_rps, resources, load_balancer_cost)
        return counts

    def score(counts: list[int]) -> float:
        return make_configuration(counts, resources, load_balancer_cost).hourly_cost

    population = [
        refine([rng.randint(0, upper_bound) for _ in resources]) for _ in range(population_size)
    ]

    best_cost = min(score(c) for c in population)
    generations_to_best = 0
    time_to_best = time.perf_counter() - start

    for generation in range(1, generations + 1):
        population.sort(key=score)
        next_population = population[: max(2, population_size // 10)]
        while len(next_population) < population_size:
            parent_a, parent_b = rng.sample(population[: max(2, population_size // 2)], 2)
            cut = rng.randrange(1, len(resources)) if len(resources) > 1 else 1
            child = parent_a[:cut] + parent_b[cut:]
            if rng.random() < 0.25:
                child[rng.randrange(len(resources))] += rng.choice([-1, 1])
            next_population.append(refine(child))
        population = next_population

        generation_best = min(score(c) for c in population)
        if generation_best < best_cost - 1e-12:
            best_cost = generation_best
            generations_to_best = generation
            time_to_best = time.perf_counter() - start

    best = min(population, key=score)
    configuration = make_configuration(best, resources, load_balancer_cost)
    configuration.runtime_seconds = time.perf_counter() - start
    configuration.generations_to_best = generations_to_best
    configuration.time_to_best_seconds = time_to_best
    return configuration


def memetic_optimize(
    required_rps: float,
    resources: list[ResourceType],
    load_balancer_cost: float,
    population_size: int = 80,
    generations: int = 100,
    seed: int = 7,
) -> Configuration:
    """Proposed method: evolutionary search + repair + local search."""
    return _evolutionary_search(
        required_rps, resources, load_balancer_cost,
        population_size, generations, seed, use_local_search=True,
    )


def genetic_baseline(
    required_rps: float,
    resources: list[ResourceType],
    load_balancer_cost: float,
    population_size: int = 80,
    generations: int = 100,
    seed: int = 7,
) -> Configuration:
    """Baseline 2: identical to the memetic algorithm but with no local search."""
    return _evolutionary_search(
        required_rps, resources, load_balancer_cost,
        population_size, generations, seed, use_local_search=False,
    )


# --------------------------------------------------------------------------
# Running and reporting every method
# --------------------------------------------------------------------------
METHODS: dict[str, Callable[..., Configuration]] = {
    "greedy": greedy_baseline,
    "genetic": genetic_baseline,
    "exact": exact_optimal,
    "memetic": memetic_optimize,
}


def run_all_methods(
    required_rps: float,
    resources: list[ResourceType],
    load_balancer_cost: float,
    repeats: int = 1,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Configuration]:
    results = {}
    for name, method in METHODS.items():
        if progress_callback:
            progress_callback("optimizing", f"Running {name} method")
        results[name] = timed_run(
            lambda m=method: m(required_rps, resources, load_balancer_cost), repeats
        )
    return results


def describe(configuration: Configuration, resources: list[ResourceType]) -> str:
    selected = [
        f"{count} x {resource.name}"
        for count, resource in zip(configuration.counts, resources)
        if count
    ]
    return (
        f"{' + '.join(selected)} | capacity={configuration.capacity_rps:.2f} RPS | "
        f"instances=${configuration.instance_cost:.4f}/hr | "
        f"load_balancer=${configuration.load_balancer_cost:.4f}/hr | "
        f"total=${configuration.hourly_cost:.4f}/hr"
    )


def configuration_report(
    configuration: Configuration,
    resources: list[ResourceType],
    optimal_cost: float | None = None,
) -> dict[str, Any]:
    report = {
        "description": describe(configuration, resources),
        "counts": configuration.counts,
        "capacity_rps": configuration.capacity_rps,
        "hourly_cost": configuration.hourly_cost,
        "instance_cost": configuration.instance_cost,
        "load_balancer_cost": configuration.load_balancer_cost,
        "total_instances": configuration.total_instances,
        "resources": [
            {"name": resource.name, "count": count}
            for count, resource in zip(configuration.counts, resources)
            if count
        ],
        "runtime_seconds": configuration.runtime_seconds,
        "generations_to_best": configuration.generations_to_best,
        "time_to_best_seconds": configuration.time_to_best_seconds,
    }
    if optimal_cost is not None and optimal_cost > 0:
        report["gap_to_optimal_pct"] = max(0.0, (configuration.hourly_cost - optimal_cost) / optimal_cost * 100)
    else:
        report["gap_to_optimal_pct"] = 0.0 if optimal_cost == 0 else None
    return report


def comparison_report(
    required_rps: float,
    resources: list[ResourceType],
    load_balancer_cost: float,
    repeats: int = 1,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, dict[str, Any]]:
    results = run_all_methods(required_rps, resources, load_balancer_cost, repeats, progress_callback)
    optimal_cost = results["exact"].hourly_cost
    return {name: configuration_report(cfg, resources, optimal_cost) for name, cfg in results.items()}


def run_pipeline_report(
    request_path: str,
    resource_path: str,
    load_balancer_cost: float,
    epochs: int,
    progress_callback: Callable[[str, str], None] | None = None,
    repeats: int = 1,
) -> dict[str, Any]:
    if progress_callback:
        progress_callback("validating", "Reading request history and instance catalogue")
    values = load_requests(request_path)
    resources = load_resources(resource_path)

    if progress_callback:
        progress_callback("training", "Preparing the forecast model")
    t0 = time.perf_counter()
    model, scaler, scaled = build_model(values, epochs=epochs, progress_callback=progress_callback)
    training_seconds = time.perf_counter() - t0

    if progress_callback:
        progress_callback("forecasting", "Projecting the next 168 hours")
    t0 = time.perf_counter()
    forecast = forecast_week(model, scaler, scaled)
    forecast_seconds = time.perf_counter() - t0

    peak_hourly = float(forecast.max())
    required_rps = peak_hourly / 3600.0

    if progress_callback:
        progress_callback("optimizing", "Comparing resource configurations")
    methods = comparison_report(required_rps, resources, load_balancer_cost, repeats, progress_callback)

    return {
        "observations": len(values),
        "forecast_hours": FORECAST_HOURS,
        "forecast_peak": peak_hourly,
        "required_rps": required_rps,
        # Kept for backward compatibility with the existing web application
        "baseline": methods["greedy"],
        "result": methods["memetic"],
        # New: every method, each with its runtime
        "methods": methods,
        "timings": {
            "training_seconds": training_seconds,
            "forecast_seconds": forecast_seconds,
            "optimization_seconds": sum(m["runtime_seconds"] for m in methods.values()),
        },
    }


# --------------------------------------------------------------------------
# Optimization-only benchmark over several required-capacity targets
# --------------------------------------------------------------------------
def run_benchmark(
    targets: list[float],
    resource_path: str,
    load_balancer_cost: float,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    resources = load_resources(resource_path)
    rows = []
    for target in targets:
        methods = comparison_report(target, resources, load_balancer_cost, repeats)
        for name, report in methods.items():
            rows.append({"required_rps": target, "method": name, **report})
    return rows


def print_benchmark(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'RPS':>8} {'method':<8} {'cost $/h':>9} {'gap %':>7} "
        f"{'runtime ms':>11} {'to-best ms':>11} {'gen':>4}  configuration"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        to_best = "-" if r["time_to_best_seconds"] is None else f"{r['time_to_best_seconds'] * 1000:.1f}"
        gen = "-" if r["generations_to_best"] is None else str(r["generations_to_best"])
        gap = "-" if r["gap_to_optimal_pct"] is None else f"{r['gap_to_optimal_pct']:.1f}"
        config = " + ".join(f"{x['count']}x{x['name']}" for x in r["resources"])
        print(
            f"{r['required_rps']:>8.0f} {r['method']:<8} {r['hourly_cost']:>9.4f} {gap:>7} "
            f"{r['runtime_seconds'] * 1000:>11.2f} {to_best:>11} {gen:>4}  {config}"
        )
    print()


def run_pipeline(request_path: str, resource_path: str, load_balancer_cost: float, epochs: int, repeats: int = 1) -> None:
    report = run_pipeline_report(request_path, resource_path, load_balancer_cost, epochs, repeats=repeats)
    print(f"Forecast peak: {report['forecast_peak']:.2f} requests/hour")
    print(f"Required capacity: {report['required_rps']:.4f} requests/second")
    t = report["timings"]
    print(f"LSTM training: {t['training_seconds']:.1f}s | forecast: {t['forecast_seconds']:.1f}s")
    for name, m in report["methods"].items():
        extra = ""
        if m["time_to_best_seconds"] is not None:
            extra = f" (best found at generation {m['generations_to_best']}, {m['time_to_best_seconds'] * 1000:.1f} ms)"
        print(f"{name:>8}: {m['description']} | runtime={m['runtime_seconds'] * 1000:.2f} ms{extra}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", default="requests_every_hour.csv")
    parser.add_argument("--resources", default="instance_types.csv")
    parser.add_argument("--load-balancer-cost", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeat each optimisation method this many times and report the mean runtime.")
    parser.add_argument("--targets", type=float, nargs="+",
                        help="Skip forecasting and compare all methods on these required capacities (RPS).")
    args = parser.parse_args()
    if args.targets:
        print_benchmark(run_benchmark(args.targets, args.resources, args.load_balancer_cost, max(args.repeats, 5)))
    else:
        run_pipeline(args.requests, args.resources, args.load_balancer_cost, args.epochs, args.repeats)
