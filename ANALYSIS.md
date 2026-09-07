# Week 5 Analysis — Model Compression for Edge Inference

## 1. Deployment problem

The goal is not only to maximize classification accuracy; the model must also fit a constrained edge deployment. The simulated target is a Raspberry-Pi-class CPU used for low-power digit recognition. Before compression, I set three measurable acceptance criteria: no more than 2.5 MB on disk, median single-image inference latency no more than 5 ms on the benchmark CPU, and at least 95% held-out accuracy. I also use a roughly 5 W power budget as the scenario envelope; actual electrical power could not be measured in the execution environment, so this is an estimate rather than a measured value.

The baseline is a deliberately over-parameterized FP32 MLP operating on 8×8 grayscale images. It achieved **98.67%** accuracy, occupied **8.566 MB**, contained **2,241,546 parameters**, and had a median latency of **0.391 ms/image** on the same CPU used for every comparison. The baseline therefore meets the accuracy and latency requirements but fails the **≤2.5 MB** model-size target.

## 2. What pruning changed

Pruning removes low-importance parameters rather than shrinking every layer uniformly. I used L1 unstructured pruning on the Linear-layer weights at a 50% target. Immediately after pruning, accuracy was **97.78%**. I then performed a short recovery fine-tune and re-applied 50% sparsity so the final saved pruned model remained actually sparse. The final measured sparsity was **50.00%**, and final accuracy was **98.22%**.

The pruned FP32 model remained about **8.566 MB** and had a median latency of **0.378 ms/image**. The important practical result is that unstructured pruning does **not automatically make dense CPU inference substantially smaller or faster**. Standard dense kernels still process tensors containing zeros, and ordinary dense serialization still stores those tensor positions. Pruning therefore creates sparsity that a sparsity-aware runtime could exploit; sparsity alone is not a guarantee of storage or latency savings.

## 3. What quantization changed

Quantization reduces numerical precision. The FP32 baseline stores weights using 32-bit floating-point values, while dynamic INT8 quantization stores Linear-layer weights in 8-bit integer form and quantizes activations dynamically during inference. This directly reduces storage and can improve CPU inference when optimized integer kernels are available.

The quantized model achieved **98.67%** accuracy, used **2.169 MB**, and had a median latency of **0.158 ms/image**. Relative to the baseline, that is a **74.67% reduction in model size**, a **59.46% reduction in median latency**, and **0.00 percentage-point accuracy loss**.

## 4. Combined compression

I also applied dynamic INT8 quantization to the pruned model. The combined variant achieved **98.22%** accuracy, occupied **2.170 MB**, and had a median latency of **0.156 ms/image**. It therefore meets the stated deployment constraints, but compared with quantization alone it gives up about **0.44 percentage points of accuracy** without a meaningful storage advantage.

This result shows that pruning and quantization do not necessarily stack into a better deployment candidate. In this setup, quantization provides the decisive storage and latency benefit, while unstructured pruning mainly provides sparsity.

## 5. Accuracy–efficiency trade-off

| Variant | Accuracy | Model size | Median latency | p95 latency | Size reduction | Accuracy drop |
|---|---:|---:|---:|---:|---:|---:|
| Baseline FP32 | 98.67% | 8.566 MB | 0.391 ms | 0.518 ms | 0.00% | 0.00 pp |
| Pruned FP32 | 98.22% | 8.566 MB | 0.378 ms | 0.517 ms | ~0.00% | 0.44 pp |
| Quantized INT8 | 98.67% | 2.169 MB | 0.158 ms | 0.241 ms | 74.67% | 0.00 pp |
| Pruned + Quantized INT8 | 98.22% | 2.170 MB | 0.156 ms | 0.234 ms | 74.67% | 0.44 pp |

The comparison demonstrates why model compression should be evaluated using measured deployment metrics rather than theoretical FLOP reductions alone. Different techniques affect different bottlenecks: pruning creates sparsity, while quantization directly reduces numerical precision and storage requirements. The actual speed benefit depends on the execution runtime and hardware kernels.

## 6. Deployment readiness

Against the predefined CPU-simulation constraints:

- **Baseline FP32:** passes accuracy and latency, fails model-size target.
- **Pruned FP32:** passes accuracy and latency, fails model-size target.
- **Quantized INT8:** passes accuracy, latency, and model-size targets.
- **Pruned + Quantized INT8:** passes accuracy, latency, and model-size targets.

The power target remains a scenario estimate because dedicated Raspberry Pi/Jetson telemetry was not available. A production deployment should repeat power and thermal measurements on the actual board.

## 7. Final recommendation

I recommend **Quantized INT8** for deployment. It reduces the model from about **8.566 MB to 2.169 MB**, cuts median latency from about **0.391 ms to 0.158 ms**, and preserves the baseline **98.67% accuracy**. The combined pruned + INT8 model is slightly faster in this benchmark, but the difference is very small and comes with a measurable accuracy loss, so quantization alone is the better overall trade-off.

The assignment also calls for benchmarking through an edge runtime such as ONNX Runtime, TensorFlow Lite, or TensorRT. The current measured numbers are from PyTorch/TorchScript CPU because `onnx` and `onnxruntime` were not installed in the execution environment. I did not fabricate those numbers. The repository includes `src/export_onnx.py`, which exports the exact baseline and pruned checkpoints, creates INT8 ONNX variants, evaluates all four models on the same fixed test split, and records ONNX Runtime latency/accuracy/size metrics when run in an environment with those packages installed.
