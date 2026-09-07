"""Train, compress, benchmark, and save all Week 5 model variants.

The script is intentionally self-contained and reproducible:
- public scikit-learn Digits dataset (no external download),
- fixed random seed and fixed stratified test split,
- one CPU thread for comparable latency measurements,
- identical held-out test set for every model variant.

Primary outputs are written under ``models/`` and ``results/`` relative to the
repository root so the project runs on Windows, Linux, or macOS without any
machine-specific paths.
"""

from __future__ import annotations

import copy
import json
import platform
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
TEST_SIZE = 0.25
PRUNING_AMOUNT = 0.50
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class EdgeMLP(nn.Module):
    """Deliberately over-parameterized MLP used to make compression measurable."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(64, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.flatten(x))


def load_data() -> tuple[DataLoader, DataLoader, np.ndarray]:
    """Return deterministic train/test loaders and raw test images."""
    digits = load_digits()
    images = digits.images.astype("float32") / 16.0
    labels = digits.target.astype("int64")

    x_train, x_test, y_train, y_test = train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    # PyTorch image shape: (N, C, H, W), even though this MLP flattens it.
    x_train = x_train[:, None, :, :]
    x_test = x_test[:, None, :, :]

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    return train_loader, test_loader, x_test


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    epochs: int = 18,
    learning_rate: float = 1e-3,
) -> nn.Module:
    """Train a model using Adam and cross-entropy loss."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
    return model


@torch.inference_mode()
def evaluate_accuracy(model: nn.Module, test_loader: DataLoader) -> float:
    """Compute classification accuracy on the fixed held-out test set."""
    model.eval()
    correct = 0
    total = 0
    for features, labels in test_loader:
        predictions = model(features).argmax(dim=1)
        correct += int((predictions == labels).sum().item())
        total += labels.numel()
    return correct / total


@torch.inference_mode()
def benchmark_latency(
    model: nn.Module,
    x_test: np.ndarray,
    *,
    warmup: int = 50,
    runs: int = 500,
) -> tuple[float, float]:
    """Measure median and p95 single-image CPU latency in milliseconds."""
    model.eval()
    sample = torch.from_numpy(x_test[:1])

    # Warmup avoids counting one-time kernel/setup overhead in the benchmark.
    for _ in range(warmup):
        model(sample)

    timings_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        model(sample)
        end = time.perf_counter_ns()
        timings_ms.append((end - start) / 1e6)

    return (
        float(statistics.median(timings_ms)),
        float(np.percentile(timings_ms, 95)),
    )


def save_torchscript(model: nn.Module, path: Path, x_test: np.ndarray) -> float:
    """Save a TorchScript artifact and return its size in MiB."""
    model.eval()
    sample = torch.from_numpy(x_test[:1])
    scripted = torch.jit.trace(model, sample)
    scripted.save(str(path))
    return path.stat().st_size / 1024**2


def logical_parameter_count(model: nn.Module) -> int:
    """Count model parameters before packing/compression changes storage format."""
    return sum(parameter.numel() for parameter in model.parameters())


def apply_l1_pruning(model: nn.Module, amount: float) -> None:
    """Apply L1 unstructured pruning to every Linear layer in-place."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)


def make_pruning_permanent(model: nn.Module) -> None:
    """Remove pruning reparameterization while keeping zeroed weights."""
    for module in model.modules():
        if isinstance(module, nn.Linear) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")


def calculate_linear_sparsity(model: nn.Module) -> float:
    """Return the fraction of zero weights across all Linear layers."""
    zero_weights = 0
    total_weights = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.detach()
            zero_weights += int((weights == 0).sum().item())
            total_weights += weights.numel()
    return zero_weights / total_weights if total_weights else 0.0


def save_tradeoff_plots(metrics: pd.DataFrame) -> None:
    """Save the two required accuracy-vs-efficiency comparison plots."""
    plt.figure(figsize=(7, 5))
    plt.scatter(metrics["median_latency_ms"], metrics["accuracy"] * 100, s=70)
    for _, row in metrics.iterrows():
        plt.annotate(
            row["variant"],
            (row["median_latency_ms"], row["accuracy"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("Median inference latency (ms/image)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Latency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_vs_latency.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(metrics["model_size_mb"], metrics["accuracy"] * 100, s=70)
    for _, row in metrics.iterrows():
        plt.annotate(
            row["variant"],
            (row["model_size_mb"], row["accuracy"] * 100),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    plt.xlabel("Model size on disk (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Model Size")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "accuracy_vs_size.png", dpi=180)
    plt.close()


def main() -> None:
    """Run baseline training, pruning, quantization, and CPU benchmarking."""
    train_loader, test_loader, x_test = load_data()

    # 1) Full-precision baseline.
    baseline = train_model(EdgeMLP(), train_loader)
    baseline_accuracy = evaluate_accuracy(baseline, test_loader)
    baseline_parameters = logical_parameter_count(baseline)
    torch.save(baseline.state_dict(), MODELS_DIR / "baseline_fp32_state_dict.pth")
    baseline_size = save_torchscript(baseline, MODELS_DIR / "baseline_fp32.pt", x_test)
    baseline_median, baseline_p95 = benchmark_latency(baseline, x_test)

    # 2) 50% L1 pruning, recovery fine-tuning, then re-enforce final sparsity.
    pruned = copy.deepcopy(baseline)
    apply_l1_pruning(pruned, PRUNING_AMOUNT)
    pruned_accuracy_before_recovery = evaluate_accuracy(pruned, test_loader)
    make_pruning_permanent(pruned)

    train_model(pruned, train_loader, epochs=4, learning_rate=2e-4)
    apply_l1_pruning(pruned, PRUNING_AMOUNT)
    make_pruning_permanent(pruned)

    pruned_accuracy = evaluate_accuracy(pruned, test_loader)
    pruning_sparsity = calculate_linear_sparsity(pruned)
    torch.save(pruned.state_dict(), MODELS_DIR / "pruned_fp32_state_dict.pth")
    pruned_size = save_torchscript(pruned, MODELS_DIR / "pruned_fp32.pt", x_test)
    pruned_median, pruned_p95 = benchmark_latency(pruned, x_test)

    # 3) Dynamic INT8 quantization of Linear-layer weights.
    quantized = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(baseline).eval(), {nn.Linear}, dtype=torch.qint8
    )
    quantized_accuracy = evaluate_accuracy(quantized, test_loader)
    quantized_size = save_torchscript(
        quantized, MODELS_DIR / "quantized_int8.pt", x_test
    )
    quantized_median, quantized_p95 = benchmark_latency(quantized, x_test)

    # 4) Combined pruning + dynamic INT8 quantization.
    combined = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(pruned).eval(), {nn.Linear}, dtype=torch.qint8
    )
    combined_accuracy = evaluate_accuracy(combined, test_loader)
    combined_size = save_torchscript(
        combined, MODELS_DIR / "pruned_quantized_int8.pt", x_test
    )
    combined_median, combined_p95 = benchmark_latency(combined, x_test)

    rows = [
        {
            "variant": "Baseline FP32",
            "accuracy": baseline_accuracy,
            "model_size_mb": baseline_size,
            "parameter_count": baseline_parameters,
            "median_latency_ms": baseline_median,
            "p95_latency_ms": baseline_p95,
            "sparsity": 0.0,
            "precision": "FP32",
            "runtime": "PyTorch/TorchScript CPU",
        },
        {
            "variant": "Pruned FP32",
            "accuracy": pruned_accuracy,
            "model_size_mb": pruned_size,
            "parameter_count": baseline_parameters,
            "median_latency_ms": pruned_median,
            "p95_latency_ms": pruned_p95,
            "sparsity": pruning_sparsity,
            "precision": "FP32",
            "runtime": "PyTorch/TorchScript CPU",
        },
        {
            "variant": "Quantized INT8",
            "accuracy": quantized_accuracy,
            "model_size_mb": quantized_size,
            "parameter_count": baseline_parameters,
            "median_latency_ms": quantized_median,
            "p95_latency_ms": quantized_p95,
            "sparsity": 0.0,
            "precision": "INT8 weights / dynamic activations",
            "runtime": "PyTorch/TorchScript CPU",
        },
        {
            "variant": "Pruned + Quantized INT8",
            "accuracy": combined_accuracy,
            "model_size_mb": combined_size,
            "parameter_count": baseline_parameters,
            "median_latency_ms": combined_median,
            "p95_latency_ms": combined_p95,
            "sparsity": pruning_sparsity,
            "precision": "INT8 weights / dynamic activations",
            "runtime": "PyTorch/TorchScript CPU",
        },
    ]

    metrics = pd.DataFrame(rows)
    metrics["size_reduction_pct"] = (
        1 - metrics["model_size_mb"] / baseline_size
    ) * 100
    metrics["latency_change_pct"] = (
        1 - metrics["median_latency_ms"] / baseline_median
    ) * 100
    metrics["accuracy_drop_pp"] = (
        baseline_accuracy - metrics["accuracy"]
    ) * 100

    metrics.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    (RESULTS_DIR / "metrics_summary.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "dataset": "sklearn digits",
                "test_samples": len(test_loader.dataset),
                "pruned_accuracy_before_recovery": pruned_accuracy_before_recovery,
                "final_pruning_sparsity": pruning_sparsity,
                "metrics": metrics.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    save_tradeoff_plots(metrics)

    runtime_info = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_threads": torch.get_num_threads(),
        "note": (
            "Benchmarks were executed in a CPU-only environment. Dedicated "
            "Raspberry Pi/Jetson power telemetry was not available."
        ),
    }
    (RESULTS_DIR / "runtime_info.json").write_text(
        json.dumps(runtime_info, indent=2), encoding="utf-8"
    )

    print(metrics.to_string(index=False))
    print(f"Pruned accuracy before recovery: {pruned_accuracy_before_recovery:.4f}")
    print(f"Final pruning sparsity: {pruning_sparsity:.4f}")
    print(f"Outputs saved under: {ROOT}")


if __name__ == "__main__":
    main()
