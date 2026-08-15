from pypylon import pylon
import cv2
import time

RETRIEVAL_TIMEOUT_MS = 5000 # timout time in miliseconds
EXPOSURE_TIME = 100000 # exposure time in micrseconds
IMAGE_WINDOW_NAME = "Live Grab"
INVALID_FRAME_MESSAGE = "INVALID FRAME"
INVALID_FRAME_WAIT = 2 # seconds
CV2_LIVE_FEED_WAITKEY = 1 # show a continuous image stream
QUIT_KEY = 'q'

def main():

    # connect to the first found camera. works for a one camera set-up
    with pylon.InstantCamera(pylon.FirstFound) as camera:

        # open camera
        camera.Open()
        print(f"camera is: {camera}")

        # set exposure of camera
        camera.ExposureTime.SetValue(EXPOSURE_TIME)
        print(f"camera exposure is {camera.ExposureTime.GetValue()}")

        # begin grabbing images
        camera.StartGrabbing(pylon.GrabStrategy_LatestImages)
        print(f"camera is grabbing? {camera.IsGrabbing()}")

        # grab loop
        try:
            while camera.IsGrabbing():

                # grab the latest frame
                try: 
                    with camera.RetrieveResult(RETRIEVAL_TIMEOUT_MS, pylon.TimeoutHandling_ThrowException) as grab_result:

                        # check validity of frame
                        if grab_result.GrabSucceeded():

                            # convert and display grab result; break and cleanup if QUIT_KEY is pressed
                            image = grab_result.Array
                            cv2.imshow(IMAGE_WINDOW_NAME, image)
                            if cv2.waitKey(CV2_LIVE_FEED_WAITKEY) & 0xFF == ord(QUIT_KEY):
                                break
                    
                        else:
                            print(INVALID_FRAME_MESSAGE)
                            time.sleep(INVALID_FRAME_WAIT)

                except pylon.TimeoutException as e:
                    print(f"error retrieving frame: {e}")   
        
        # cleanup
        finally:
            print("cleaning up")
            camera.StopGrabbing()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()