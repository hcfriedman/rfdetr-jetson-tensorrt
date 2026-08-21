import tensorrt as trt
from cuda.bindings import runtime as cudart
import numpy as np
from onnx_runtime_inference import BenchmarkResult
from vendored_rfdetr.onnx_helpers import _decode_raw_outputs, _preprocess_pil_to_nchw
import threading
from acquisition import loop_grab_and_update_latest_frame, LatestFrame
import supervision as sv
import time
from PIL import Image
import cv2

WARMUP_RUNS = 20
MEASURE_RUNS = 100
CONFIDENCE_THRESHOLD = 0.5
ENGINE_PATH = "models/rfdetr_small_fp16.engine"
VIEW_LIVE_DETECTION = False
CV2_LIVE_FEED_WAITKEY = 1 # show a continuous image stream
QUIT_KEY = 'q'

# _check, and the TensorRTEngine's functions are adapted from 
# https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html
# https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html#python-api-docs
def _check(err_result):
    err = err_result[0] if isinstance(err_result, tuple) else err_result # handle tuple with true err in first spot
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")

class TensorRTEngine:
    def __init__(self, engine_path):
        # load in engine
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        # logger is required input for runtime instance
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # deserialize engine bytes and create execution context
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        self.context = self.engine.create_execution_context()

        # get input name, shape, and output names
        self.input_name = None
        self.input_shape = None
        self.output_names = []
        for tensor_index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(tensor_index)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = self.engine.get_tensor_shape(name)
            elif self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        self.context.set_input_shape(self.input_name, self.input_shape)

        # allocate memory sized to input
        input_data_type = self.engine.get_tensor_dtype(self.input_name)
        np_input_data_type = trt.nptype(input_data_type)
        input_num_bytes = int(np.prod(self.input_shape)) * np.dtype(np_input_data_type).itemsize
        err, self.d_input = cudart.cudaMalloc(input_num_bytes); _check(err)

        # size output shape and data type
        output_shapes = [tuple(self.context.get_tensor_shape(output_name)) for output_name in self.output_names]
        self.host_outputs = []
        for output_name, output_shape in zip(self.output_names, output_shapes):
            output_data_type = self.engine.get_tensor_dtype(output_name)
            np_output_data_type = trt.nptype(output_data_type)
            self.host_outputs.append(np.empty(output_shape, dtype=np_output_data_type))

        # allocate memory to output
        self.d_outputs = []
        for host_output in self.host_outputs:
            err, d_output = cudart.cudaMalloc(host_output.nbytes); _check(err)
            self.d_outputs.append(d_output)

        err, self.stream = cudart.cudaStreamCreate(); _check(err)

        # set addresses for input and output
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        for output_name, d_output in zip(self.output_names, self.d_outputs):
            self.context.set_tensor_address(output_name, int(d_output))

    def run_inference(self, input_array: np.ndarray):
        # copy input to device
        _check(cudart.cudaMemcpyAsync(      
            self.d_input, input_array.ctypes.data, input_array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream,
        ))

        # execute stream
        self.context.execute_async_v3(self.stream)

        # copy outputs to self.host_outputs
        for host_output, d_output in zip(self.host_outputs, self.d_outputs):
            _check(cudart.cudaMemcpyAsync(
                host_output.ctypes.data, d_output, host_output.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream,
            ))

        # block until outputs are copied back to self.host_outputs
        _check(cudart.cudaStreamSynchronize(self.stream))

    def free_and_cleanup(self):
        # cleanup memory and destroy stream
        cudart.cudaFree(self.d_input)
        [cudart.cudaFree(d_output) for d_output in self.d_outputs]
        cudart.cudaStreamDestroy(self.stream)

def main():
    # initialize latest frame class and loop for grabbing latest frame
    latest_frame = LatestFrame()
    stop = threading.Event()
    live_frame_thread = threading.Thread(target=loop_grab_and_update_latest_frame, args=(latest_frame, stop), daemon=True)

    # start thread
    live_frame_thread.start()

    try:
        # initialize annotators
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        # setup engine
        engine  = TensorRTEngine(ENGINE_PATH)
        
        # loop over incoming frames run inference
        timings = []
        current_frame_number = 0
        for usable_frame in range(WARMUP_RUNS + MEASURE_RUNS):
            # grab and preprocess the latest frame
            current_frame, current_frame_number = latest_frame.get_new_latest_frame(current_frame_number)
            preprocessed_frame = _preprocess_pil_to_nchw(Image.fromarray(current_frame), engine.input_shape[2], engine.input_shape[3], engine.input_shape[1])
            preprocessed_frame = np.ascontiguousarray(preprocessed_frame)

            # initialize timing and run inference
            t0 = time.perf_counter()
            engine.run_inference(preprocessed_frame)
            if usable_frame >= WARMUP_RUNS:
                timings.append((time.perf_counter() - t0) * 1000.0)

                # decode detections from inference
                detections = _decode_raw_outputs(
                    raw_outputs=engine.host_outputs,
                    output_names=engine.output_names,
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
        
        engine.free_and_cleanup()

        # crunch and print timing data
        timings_array = np.array(timings)
        benchmark_result = BenchmarkResult(
            label=f"TensorRT at {ENGINE_PATH}",
            mean_ms=float(timings_array.mean()),
            std_ms=float(timings_array.std()),
        )
        print(f"{benchmark_result.label:<12} {benchmark_result.mean_ms:7.2f} ms +/- {benchmark_result.std_ms:5.2f}")
    
    # cleanup frame grabbing thread
    finally:
        stop.set()
        live_frame_thread.join()

if __name__ == "__main__":
    main()