from vendored_rfdetr.onnx_helpers import _create_onnx_session, _preprocess_pil_to_nchw
import threading
from acquisition import LatestFrame, loop_grab_and_update_latest_frame
import onnxruntime as ort
import numpy as np
from PIL import Image
from collections.abc import Callable
from typing import Any, NamedTuple
import time
import gc

WARMUP_RUNS = 20
MEASURE_RUNS = 100
ONNX_PATH = r"models/rfdetr_small.onnx/rfdetr-small.onnx"
PROVIDER_CONFIGS = ( ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"])

# class taken from https://github.com/roboflow/rf-detr: docs/cookbooks/inference-latency-benchmark.ipynb
class BenchmarkResult(NamedTuple):
    """Single benchmark measurement."""

    label: str
    mean_ms: float
    std_ms: float

    @property
    def fps(self) -> float:
        """Frames per second."""
        return 1000.0 / self.mean_ms


def check_ort_cuda_availibility():
    _ort_providers = ort.get_available_providers()
    print(f"ORT {ort.__version__}, providers: {_ort_providers}")
    if "CUDAExecutionProvider" not in _ort_providers:
        raise RuntimeError(
            f"onnxruntime-gpu with CUDA support required. Available providers: {_ort_providers}. "
            "Fix: reinstall from the CUDA-12 index (see install cell) then restart runtime."
        )


def main():

    check_ort_cuda_availibility()

    # initialize latest frame class, thread for gathering images
    latest_frame = LatestFrame()

    stop = threading.Event()

    live_frame_thread = threading.Thread(target=loop_grab_and_update_latest_frame, args=(latest_frame, stop), daemon=True)

    live_frame_thread.start()
    
    try:
        for providers in PROVIDER_CONFIGS:
            onnx_session = _create_onnx_session(ONNX_PATH, providers=providers)
            active = onnx_session.get_providers()[0]
            if active != providers[0]:
                raise RuntimeError(
                    f"Requested provider {providers[0]!r} not active — ORT fell back to {active!r}. "
                )
            
            inputs = onnx_session.get_inputs()
            input_name = inputs[0].name
            
            current_frame_number = 0
            timings = []
            for usable_frame in range(WARMUP_RUNS + MEASURE_RUNS):
                current_frame, current_frame_number = latest_frame.get_new_latest_frame(current_frame_number)
                input_meta = onnx_session.get_inputs()[0]
                _, channels, height, width = input_meta.shape
                inference_formatted_current_frame = _preprocess_pil_to_nchw(Image.fromarray(current_frame), height, width, channels)
                t0 = time.perf_counter()
                onnx_session.run(None, {input_name: inference_formatted_current_frame})
                if usable_frame >= WARMUP_RUNS:
                    timings.append((time.perf_counter() - t0) * 1000.0)

            # clean up due to tight memory budget
            del onnx_session
            gc.collect()

            timings_array = np.array(timings)
            benchmark_result = BenchmarkResult(
                label=F"ONNX ({active.replace('ExecutionProvider', '')})",
                mean_ms=float(timings_array.mean()),
                std_ms=float(timings_array.std()),
            )
            print(f"{benchmark_result.label:<12} {benchmark_result.mean_ms:7.2f} ms +/- {benchmark_result.std_ms:5.2f}")

    finally:
        stop.set()
        live_frame_thread.join()


if __name__ == "__main__":
    main()