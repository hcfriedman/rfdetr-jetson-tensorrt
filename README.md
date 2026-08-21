# rfdetr-jetson-tensorrt
Deployment and optimization of RF-DETR for real-time person detection on an NVIDIA Jetson Orin Nano Super, using TensorRT.


## Setup
```bash
uv sync
# expose JetPack's tensorrt bindings to the venv (redo after recreating .venv)
echo "/usr/lib/python3.10/dist-packages" > .venv/lib/python3.10/site-packages/jetpack.pth   
uv run python export_rfdetr_to_onnx.py # generates models/ (.onnx not committed)

# build TensorRT engines (on-device; .engine files are hardware-specific, not committed)
/usr/src/tensorrt/bin/trtexec --onnx=models/rfdetr_small.onnx/rfdetr-small.onnx --saveEngine=models/rfdetr_small_fp32.engine
/usr/src/tensorrt/bin/trtexec --onnx=models/rfdetr_small.onnx/rfdetr-small.onnx --saveEngine=models/rfdetr_small_fp16.engine --fp16
```

## Benchmark
```bash
sudo nvpmodel -m 2 # MAXN_SUPER power mode (25W is the default mode)
sudo jetson_clocks # lock clocks, unlocked runs are ~50% slower and noisy
uv run python onnx_runtime_inference.py
uv run python tensorrt_inference.py # set ENGINE_PATH for fp32/fp16
```

## Results
RFDETRSmall, live Basler frames, Jetson Orin Nano Super, jetson_clocks locked, 100 runs, headless, inference only:

| Backend | Live Latency | vs baseline |
|---|---|---|
| ONNX Runtime CPU (FP32) | 521 ms ± 19 (1.9 FPS) | - |
| ONNX Runtime CUDA (FP32) | 56.1 ms ± 1.6 (17.8 FPS) | 1x |
| TensorRT (FP32) | 31.7 ms ± 0.5 (31.5 FPS) | 1.8x |
| TensorRT (FP16) | 13.0 ms ± 0.1 (76.9 FPS) | 4.3x |

## Notes
- Never import torch or rfdetr in the inference/benchmark process. 
  Torch's pip CUDA 12.9 libs break the Jetson ORT build's CUDA init. 
  Torch-free preprocessing lives in `vendored_rfdetr/`.
- TRT input buffers are filled via raw-pointer memcpy; 
  arrays must be C-contiguous (see ascontiguousarray in the frame loop); 
  non-contiguous views produce silent garbage detections.