from pypylon import pylon
import cv2
import time
import threading

RETRIEVAL_TIMEOUT_MS = 5000 # timout time in miliseconds
EXPOSURE_TIME = 100000 # exposure time in micrseconds
PIXEL_FORMAT = "RGB8"
IMAGE_WINDOW_NAME = "Live Grab"
INVALID_FRAME_MESSAGE = "INVALID FRAME"
INVALID_FRAME_WAIT = 2 # seconds
CV2_LIVE_FEED_WAITKEY = 1 # show a continuous image stream
QUIT_KEY = 'q'

class LatestFrame:
    def __init__(self):
        self.frame_number = 0
        self.frame = None
        self.cond = threading.Condition()

    def update_frame(self, frame):
        with self.cond:
            self.frame = frame
            self.frame_number += 1
            self.cond.notify_all()

    def get_new_latest_frame(self, last_frame_number, timeout=1.0):
        with self.cond:
            if not self.cond.wait_for(lambda: self.frame_number != last_frame_number, timeout):
                raise TimeoutError("no new frame from camera")
            return self.frame, self.frame_number
    

def loop_grab_and_update_latest_frame(latest_frame: LatestFrame, stop: threading.Event):

    # connect to the first found camera. works for a one camera set-up
    with pylon.InstantCamera(pylon.FirstFound) as camera:

        # open camera
        camera.Open()
        print(f"camera is: {camera}")

        # set pixel format value
        camera.PixelFormat.Value = PIXEL_FORMAT

        # set exposure of camera
        camera.ExposureTime.SetValue(EXPOSURE_TIME)
        print(f"camera exposure is {camera.ExposureTime.GetValue()}us")

        # begin grabbing images
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        print(f"camera is grabbing? {camera.IsGrabbing()}")

        # grab loop
        try:
            while not stop.is_set() and camera.IsGrabbing():

                # grab the latest frame
                try: 
                    with camera.RetrieveResult(RETRIEVAL_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException) as grab_result:

                        # check validity of frame
                        if grab_result.GrabSucceeded():

                            # convert and display grab result; break and cleanup if QUIT_KEY is pressed
                            image = grab_result.Array.copy()
                            latest_frame.update_frame(image)
                    
                        else:
                            print(INVALID_FRAME_MESSAGE)
                            time.sleep(INVALID_FRAME_WAIT)

                except pylon.TimeoutException as e:
                    print(f"error retrieving frame: {e}")   
        
        # cleanup camera
        finally:
            print("cleaning up")
            camera.StopGrabbing()

def main():

    # initialize latest frame class and loop for grabbing latest frame
    latest_frame = LatestFrame()
    stop = threading.Event()
    live_frame_thread = threading.Thread(target=loop_grab_and_update_latest_frame, args=(latest_frame, stop), daemon=True)

    # start thread
    live_frame_thread.start()

    current_frame_number = 0
    try:
        # loop to get an updated frame and display it
        while True:
            current_frame, current_frame_number = latest_frame.get_new_latest_frame(current_frame_number)
            cv2.imshow(IMAGE_WINDOW_NAME, current_frame[:,:,::-1]) # reverse channel order due to opencv b,g,r channel ordering
            if cv2.waitKey(CV2_LIVE_FEED_WAITKEY) & 0xFF == ord(QUIT_KEY):
                break
    
    # clean up thread and display window
    finally:
        stop.set()
        live_frame_thread.join()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()