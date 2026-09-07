"""Export the exact FP32 checkpoints to ONNX and benchmark ONNX Runtime.

This is the strict edge-runtime pass requested by the Week 5 brief. Run it
after ``src/pipeline.py`` has produced the FP32 state_dict checkpoints.

Required packages:
    pip install onnx onnxruntime

Outputs:
- models/baseline_fp32.onnx
- models/pruned_fp32.onnx
- models/quantized_int8.onnx
- models/pruned_quantized_int8.onnx
- results/onnx_metrics_summary.csv
- results/onnx_metrics_summary.json
- results/onnx_accuracy_vs_latency.png
- results/onnx_accuracy_vs_size.png
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.25
ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

BASELINE_ONNX = MODELS_DIR / "baseline_fp32.onnx"
PRUNED_ONNX = MODELS_DIR / "pruned_fp32.onnx"
QUANTIZED_ONNX = MODELS_DIR / "quantized_int8.onnx"
PRUNED_QUANTIZED_ONNX = MODELS_DIR / "pruned_quantized_int8.onnx"


class EdgeMLP(nn.Module):
    """Architecture shared with the training pipeline."""

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


def load_fixed_test_set() -> tuple[np.ndarray, np.ndarray]:
    """Recreate exactly the same held-out split used by the main pipeline."""
    digits = load_digits()
    images = digits.images.astype("float32") / 16.0
    labels = digits.target.astype("int64")

    _, x_test, _, y_test = train_test_split(
        images,
        labels,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=labels,
    )
    return x_test[:, None, :, :].astype("float32"), y_test


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    """Load one exact PyTorch checkpoint and export it to ONNX."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path.name}. Run 'python src/pipeline.py' first."
        )

    model = EdgeMLP()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    sample = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        str(output_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def make_int8_model(source_path: Path, output_path: Path) -> None:
    """Apply ONNX Runtime dynamic INT8 weight quantization."""
    quantize_dynamic(
        model_input=str(source_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )


def create_session(model_path: Path) -> ort.InferenceSession:
    """Create a single-threaded CPU session for fair local comparison."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def evaluate_accuracy(
    session: ort.InferenceSession, x_test: np.ndarray, y_test: np.ndarray
) -> float:
    """Compute classification accuracy through ONNX Runtime."""
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x_test})[0]
    predictions = logits.argmax(axis=1)
    return float((predictions == y_test).mean())


def benchmark_latency(
    session: ort.InferenceSession,
    x_test: np.ndarray,
    *,
    warmup: int = 50,
    runs: int = 500,
) -> tuple[float, float]:
    """Measure median and p95 single-image ONNX Runtime latency."""
    input_name = session.get_inputs()[0].name
    sample = x_test[:1]

    for _ in range(warmup):
        session.run(None, {input_name: sample})

    timings_ms: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        session.run(None, {input_name: sample})
        end = time.perf_counter_ns()
        timings_ms.append((end - start) / 1e6)

    return (
        float(statistics.median(timings_ms)),
        float(np.percentile(timings_ms, 95)),
    )


def plot_tradeoffs(metrics: pd.DataFrame) -> None:
    """Save ONNX-runtime accuracy-vs-latency and accuracy-vs-size plots."""
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
    plt.xlabel("Median ONNX Runtime latency (ms/image)")
    plt.ylabel("Accuracy (%)")
    plt.title("ONNX Runtime: Accuracy vs Latency")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "onnx_accuracy_vs_latency.png", dpi=180)
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
    plt.xlabel("ONNX model size on disk (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title("ONNX Runtime: Accuracy vs Model Size")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "onnx_accuracy_vs_size.png", dpi=180)
    plt.close()


def main() -> None:
    """Export FP32 checkpoints, quantize them, and benchmark all four variants."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    export_checkpoint(
        MODELS_DIR / "baseline_fp32_state_dict.pth", BASELINE_ONNX
    )
    export_checkpoint(
        MODELS_DIR / "pruned_fp32_state_dict.pth", PRUNED_ONNX
    )
    make_int8_model(BASELINE_ONNX, QUANTIZED_ONNX)
    make_int8_model(PRUNED_ONNX, PRUNED_QUANTIZED_ONNX)

    x_test, y_test = load_fixed_test_set()
    parameter_count = sum(p.numel() for p in EdgeMLP().parameters())

    variants = [
        ("Baseline FP32", BASELINE_ONNX, "FP32"),
        ("Pruned FP32", PRUNED_ONNX, "FP32"),
        ("Quantized INT8", QUANTIZED_ONNX, "INT8 weights"),
        ("Pruned + Quantized INT8", PRUNED_QUANTIZED_ONNX, "INT8 weights"),
    ]

    rows: list[dict[str, object]] = []
    for name, path, precision in variants:
        session = create_session(path)
        accuracy = evaluate_accuracy(session, x_test, y_test)
        median_ms, p95_ms = benchmark_latency(session, x_test)
        rows.append(
            {
                "variant": name,
                "accuracy": accuracy,
                "model_size_mb": path.stat().st_size / 1024**2,
                "parameter_count": parameter_count,
                "median_latency_ms": median_ms,
                "p95_latency_ms": p95_ms,
                "precision": precision,
                "runtime": "ONNX Runtime CPUExecutionProvider",
            }
        )

    metrics = pd.DataFrame(rows)
    baseline_size = float(metrics.iloc[0]["model_size_mb"])
    baseline_latency = float(metrics.iloc[0]["median_latency_ms"])
    baseline_accuracy = float(metrics.iloc[0]["accuracy"])

    metrics["size_reduction_pct"] = (
        1 - metrics["model_size_mb"] / baseline_size
    ) * 100
    metrics["latency_change_pct"] = (
        1 - metrics["median_latency_ms"] / baseline_latency
    ) * 100
    metrics["accuracy_drop_pp"] = (
        baseline_accuracy - metrics["accuracy"]
    ) * 100

    metrics.to_csv(RESULTS_DIR / "onnx_metrics_summary.csv", index=False)
    (RESULTS_DIR / "onnx_metrics_summary.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "dataset": "sklearn digits",
                "test_samples": len(y_test),
                "runtime": "ONNX Runtime CPUExecutionProvider",
                "metrics": metrics.to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_tradeoffs(metrics)

    print(metrics.to_string(index=False))
    print(f"ONNX Runtime results saved under: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
