"""Forecast request load and select a least-cost EC2 configuration.

The input resource table must contain these columns:
    name, capacity_rps, cost_per_hour

The model follows the notebook's forecasting setup: a StandardScaler, a
168-hour lookback, and a one-week recursive forecast. The load balancer cost
is charged once when the selected configuration contains more than one
instance.
"""

from __future__ import annotations

import argparse
import random
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

    @property
    def total_instances(self) -> int:
        return sum(self.counts)


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


def make_configuration(
    counts: list[int], resources: list[ResourceType], load_balancer_cost: float
) -> Configuration:
    instance_cost = sum(c * r.cost_per_hour for c, r in zip(counts, resources))
    capacity = sum(c * r.capacity_rps for c, r in zip(counts, resources))
    lb_cost = load_balancer_cost if sum(counts) > 1 else 0.0
    return Configuration(counts, capacity, instance_cost + lb_cost, instance_cost, lb_cost)


def greedy_baseline(required_rps: float, resources: list[ResourceType], load_balancer_cost: float) -> Configuration:
    counts = [0] * len(resources)
    capacity = 0.0
    order = sorted(range(len(resources)), key=lambda i: resources[i].cost_per_hour / resources[i].capacity_rps)
    while capacity < required_rps:
        index = order[0]
        counts[index] += 1
        capacity += resources[index].capacity_rps
    return make_configuration(counts, resources, load_balancer_cost)


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


def memetic_optimize(
    required_rps: float,
    resources: list[ResourceType],
    load_balancer_cost: float,
    population_size: int = 80,
    generations: int = 100,
    seed: int = 7,
) -> Configuration:
    rng = random.Random(seed)
    upper_bound = max(1, int(np.ceil(required_rps / min(r.capacity_rps for r in resources))) + 1)
    population = []
    for _ in range(population_size):
        counts = [rng.randint(0, upper_bound) for _ in resources]
        counts = repair(counts, required_rps, resources, load_balancer_cost)
        population.append(local_search(counts, required_rps, resources, load_balancer_cost))

    def score(counts: list[int]) -> float:
        return make_configuration(counts, resources, load_balancer_cost).hourly_cost

    for _ in range(generations):
        population.sort(key=score)
        next_population = population[: max(2, population_size // 10)]
        while len(next_population) < population_size:
            parent_a, parent_b = rng.sample(population[: max(2, population_size // 2)], 2)
            cut = rng.randrange(1, len(resources)) if len(resources) > 1 else 1
            child = parent_a[:cut] + parent_b[cut:]
            if rng.random() < 0.25:
                child[rng.randrange(len(resources))] += rng.choice([-1, 1])
            child = repair(child, required_rps, resources, load_balancer_cost)
            next_population.append(local_search(child, required_rps, resources, load_balancer_cost))
        population = next_population

    best = min(population, key=score)
    return make_configuration(best, resources, load_balancer_cost)


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


def configuration_report(configuration: Configuration, resources: list[ResourceType]) -> dict[str, Any]:
    return {
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
    }


def run_pipeline_report(
    request_path: str,
    resource_path: str,
    load_balancer_cost: float,
    epochs: int,
    progress_callback: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    if progress_callback:
        progress_callback("validating", "Reading request history and instance catalogue")
    values = load_requests(request_path)
    resources = load_resources(resource_path)
    if progress_callback:
        progress_callback("training", "Preparing the forecast model")
    model, scaler, scaled = build_model(values, epochs=epochs, progress_callback=progress_callback)
    if progress_callback:
        progress_callback("forecasting", "Projecting the next 168 hours")
    forecast = forecast_week(model, scaler, scaled)
    peak_hourly = float(forecast.max())
    required_rps = peak_hourly / 3600.0
    if progress_callback:
        progress_callback("optimizing", "Comparing resource configurations")
    baseline = greedy_baseline(required_rps, resources, load_balancer_cost)
    result = memetic_optimize(required_rps, resources, load_balancer_cost)
    return {
        "observations": len(values),
        "forecast_hours": FORECAST_HOURS,
        "forecast_peak": peak_hourly,
        "required_rps": required_rps,
        "baseline": configuration_report(baseline, resources),
        "result": configuration_report(result, resources),
    }


def run_pipeline(request_path: str, resource_path: str, load_balancer_cost: float, epochs: int) -> None:
    report = run_pipeline_report(request_path, resource_path, load_balancer_cost, epochs)
    print(f"Forecast peak: {report['forecast_peak']:.2f} requests/hour")
    print(f"Required capacity: {report['required_rps']:.4f} requests/second")
    print(f"Greedy baseline: {report['baseline']['description']}")
    print(f"Memetic result: {report['result']['description']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", default="requests_every_hour.csv")
    parser.add_argument("--resources", default="instance_types.csv")
    parser.add_argument("--load-balancer-cost", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    run_pipeline(args.requests, args.resources, args.load_balancer_cost, args.epochs)