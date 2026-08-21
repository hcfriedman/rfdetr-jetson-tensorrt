from vendored_rfdetr.onnx_helpers import _create_onnx_session, _preprocess_pil_to_nchw, _decode_raw_outputs
import threading
from acquisition import LatestFrame, loop_grab_and_update_latest_frame
import onnxruntime as ort
import numpy as np
from PIL import Image
from collections.abc import Callable
from typing import Any, NamedTuple
import time
import gc
import supervision as sv
import cv2

WARMUP_RUNS = 20
MEASURE_RUNS = 100
CONFIDENCE_THRESHOLD = 0.5
ONNX_PATH = r"models/rfdetr_small.onnx/rfdetr-small.onnx"
PROVIDER_CONFIGS = ( ["CUDAExecutionProvider", "CPUExecutionProvider"], ["CPUExecutionProvider"])
VIEW_LIVE_DETECTION = False
CV2_LIVE_FEED_WAITKEY = 1 # show a continuous image stream
QUIT_KEY = 'q'

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

    # start thread for grabbing frames
    live_frame_thread.start()

    # initialize annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    
    try:
        for providers in PROVIDER_CONFIGS:
            # create onnx session
            onnx_session = _create_onnx_session(ONNX_PATH, providers=providers)
            active = onnx_session.get_providers()[0]
            if active != providers[0]:
                raise RuntimeError(
                    f"Requested provider {providers[0]!r} not active — ORT fell back to {active!r}. "
                )
            
            # define inputs
            inputs = onnx_session.get_inputs()
            input_name = inputs[0].name
            output_names = [o.name for o in onnx_session.get_outputs()]
            
             # loop over incoming frames run inference
            timings = []
            current_frame_number = 0
            for usable_frame in range(WARMUP_RUNS + MEASURE_RUNS):
                # grab and preprocess the latest frame
                current_frame, current_frame_number = latest_frame.get_new_latest_frame(current_frame_number)
                input_meta = onnx_session.get_inputs()[0]
                _, channels, height, width = input_meta.shape
                inference_formatted_current_frame = _preprocess_pil_to_nchw(Image.fromarray(current_frame), height, width, channels)

                # time and run inference
                t0 = time.perf_counter()
                raw_outputs = onnx_session.run(None, {input_name: inference_formatted_current_frame})
                if usable_frame >= WARMUP_RUNS:
                    timings.append((time.perf_counter() - t0) * 1000.0)

                    # decode detections from inference
                    detections = _decode_raw_outputs(
                    raw_outputs=raw_outputs,
                    output_names=output_names,
                    original_size=(current_frame.shape[1], current_frame.shape[0]),
                    threshold=CONFIDENCE_THRESHOLD,
                    num_select=None
                    )

                    # display live detections if set
                    if VIEW_LIVE_DETECTION:
                        # swap channel order due to OpenCV channel ordering, and annotate boxes and classes
                        bgr_image = current_frame[:,:,::-1].copy()
                        display_image = box_annotator.annotate(bgr_image, detections)
                        display_image = label_annotator.annotate(display_image,
                                                    detections, 
                                                    labels= [f"{c} {s:.2f}" for c, s in zip(detections.class_id, detections.confidence)])
                        
                        # display the live image, break session if QUIT_KEY is pressed
                        cv2.imshow("Live Detections", display_image)
                        if cv2.waitKey(CV2_LIVE_FEED_WAITKEY) &0xFF == ord(QUIT_KEY):
                            cv2.destroyAllWindows()
                            break


            # clean up due to potentially tight memory budget
            del onnx_session
            if VIEW_LIVE_DETECTION:
                cv2.destroyAllWindows()
            gc.collect()

            # compute and print timing stats
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