# rfdetr-jetson-tensorrt
Deployment and optimization of RF-DETR for real-time person detection on an NVIDIA Jetson Orin Nano Super, using TensorRT.


## Setup
```bash
uv sync
uv run python export_rfdetr_to_onnx.py   # generates models/ (.onnx not committed)
```

## Benchmark
```bash
sudo jetson_clocks    # lock clocks, unlocked runs are ~50% slower and noisy
uv run python onnx_runtime_inference.py
```

## Results
RFDETRSmall, live Basler frames, Jetson Orin Nano, jetson_clocks locked, 100 runs, inference only:

| Backend | Latency | FPS |
|---|---|---|
| ONNX Runtime CUDA (FP32) | 60.6 ms ± 1.0 | 16.5 |
| ONNX Runtime CPU (FP32) | 654 ms ± 22 | 1.5 |

## Notes
- Never import torch or rfdetr in the inference/benchmark process. 
  Torch's pip CUDA 12.9 libs break the Jetson ORT build's CUDA init. 
  Torch-free preprocessing lives in `vendored_rfdetr/`.