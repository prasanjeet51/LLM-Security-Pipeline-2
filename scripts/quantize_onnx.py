"""INT8 dynamic quantization for Stage A ONNX model.

Reduces model size ~4x and speeds up CPU inference by 1.5-2x
with no GPU required and negligible accuracy loss.

Usage:
    py -3.10 scripts/quantize_onnx.py
"""

import time

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

FP32_PATH = "models/stage_a_onnx/model.onnx"
INT8_PATH = "models/stage_a_onnx/model_int8.onnx"
N_RUNS = 100
SEQ_LEN = 128


def benchmark(path: str) -> float:
    """Run N_RUNS inference passes and return average latency in ms."""
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    dummy = {
        "input_ids": np.ones((1, SEQ_LEN), dtype=np.int64),
        "attention_mask": np.ones((1, SEQ_LEN), dtype=np.int64),
    }
    # Warm-up
    for _ in range(5):
        sess.run(None, dummy)
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        sess.run(None, dummy)
    return (time.perf_counter() - t0) / N_RUNS * 1000


def main() -> None:
    """Quantize fp32 ONNX to INT8 and print speedup ratio."""
    print(f"Quantizing {FP32_PATH} -> {INT8_PATH} ...")
    quantize_dynamic(
        FP32_PATH,
        INT8_PATH,
        weight_type=QuantType.QInt8,
    )
    print("Quantization complete.")

    fp32_ms = benchmark(FP32_PATH)
    int8_ms = benchmark(INT8_PATH)
    ratio = fp32_ms / int8_ms

    print(f"fp32 : {fp32_ms:.1f} ms avg ({N_RUNS} runs)")
    print(f"int8 : {int8_ms:.1f} ms avg ({N_RUNS} runs)")
    print(f"Speedup ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
