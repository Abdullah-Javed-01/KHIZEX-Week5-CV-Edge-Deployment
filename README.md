# KHIZEX Week 5 — Computer Vision Edge Deployment

**Author:** Abdullah Javed

## Project goal
Compress a trained image-classification model with **pruning** and **quantization**, then compare accuracy, model size, and real CPU inference latency. This project follows the Week 5 brief while using a CPU-only simulation because dedicated edge hardware is not available.

## Dataset
Public `sklearn` Digits dataset: 1,797 grayscale 8×8 images, 10 classes. Fixed stratified 75/25 train/test split, seed 42.

## Edge scenario and constraints (defined before compression)
Target: a low-power classroom/embedded digit-recognition camera on a Raspberry-Pi-class CPU.

- model on disk: **≤ 2.5 MB**
- median single-image latency: **≤ 5 ms** in the CPU simulation
- classification accuracy: **≥ 95%**
- power envelope: **~5 W budget** (scenario estimate; not directly measured in this container)

The baseline is intentionally over-parameterized so that compression has a meaningful deployment role.

## Methods
- **Baseline:** FP32 MLP image classifier.
- **Pruning:** 50% L1 unstructured pruning of Linear-layer weights, followed by a short recovery fine-tune and re-application of 50% sparsity.
- **Quantization:** dynamic INT8 quantization of Linear layers using PyTorch.
- **Combined:** pruned model + dynamic INT8 quantization.
- **Runtime benchmark:** same CPU, one Torch thread, fixed test set, 50 warmups + 500 single-image timed runs; median and p95 reported.

> **Important runtime note:** The supplied environment does not include the Python `onnx` / `onnxruntime` packages and internet access is disabled, so I could not truthfully claim an ONNX Runtime benchmark here. The executed numbers below are from PyTorch/TorchScript CPU. `src/export_onnx.py` is a runnable ONNX export and ONNX Runtime benchmark script for the exact FP32 checkpoints produced by the main pipeline. This is the only part that still needs a local rerun if the evaluator strictly requires ONNX/TFLite/TensorRT runtime numbers.

## Results

| variant                 | accuracy   |   model_size_mb |   parameter_count |   median_latency_ms |   p95_latency_ms |   size_reduction_pct |   accuracy_drop_pp |
|:------------------------|:-----------|----------------:|------------------:|--------------------:|-----------------:|---------------------:|-------------------:|
| Baseline FP32           | 98.67%     |           8.566 |           2241546 |               0.388 |            0.605 |                0     |              0     |
| Pruned FP32             | 98.22%     |           8.566 |           2241546 |               0.39  |            0.572 |               -0.004 |              0.444 |
| Quantized INT8          | 98.67%     |           2.169 |           2241546 |               0.161 |            0.332 |               74.675 |              0     |
| Pruned + Quantized INT8 | 98.22%     |           2.17  |           2241546 |               0.163 |            0.38  |               74.669 |              0.444 |

Pruned accuracy **before recovery:** 97.78%  
Final measured pruning sparsity: **50.00%**

## Recommendation
Using the stated CPU-simulation constraints, the selected deployment candidate is **Quantized INT8**. See `ANALYSIS.md` for the full trade-off reasoning.

## Project structure
```text
khizex_week5_cv_edge/
├── data/
│   └── README.md
├── models/
│   ├── baseline_fp32.pt
│   ├── baseline_fp32_state_dict.pth
│   ├── pruned_fp32.pt
│   ├── pruned_fp32_state_dict.pth
│   ├── quantized_int8.pt
│   └── pruned_quantized_int8.pt
├── notebooks/
│   └── README.md
├── results/
│   ├── metrics_summary.csv
│   ├── metrics_summary.json
│   ├── runtime_info.json
│   ├── accuracy_vs_latency.png
│   └── accuracy_vs_size.png
├── src/
│   ├── pipeline.py
│   └── export_onnx.py
├── ANALYSIS.md
├── README.md
└── requirements.txt
```

## Setup
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
python src/pipeline.py
```

For the strict ONNX-runtime final pass, install `onnx` and `onnxruntime`, run `python src/pipeline.py` to generate the exact FP32 checkpoints, then run `python src/export_onnx.py`. The script exports baseline/pruned ONNX models, creates INT8 variants, benchmarks all four with ONNX Runtime, and writes ONNX-specific metrics and plots.
