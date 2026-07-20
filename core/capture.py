import threading
import time
import cv2
from config.settings import VIDEO_SOURCE, FRAME_SKIP, MODEL_WIDTH, MODEL_HEIGHT

class RTSPStreamReader:
    """
    Threaded video capture class designed to prevent OpenCV frame buffer accumulation (lag).
    Always yields the absolute latest frame from the RTSP stream or test video.
    """
    def __init__(self):
        self.stream = cv2.VideoCapture(VIDEO_SOURCE)
        if not self.stream.isOpened():
            raise ValueError(f"[-] Unable to connect to video source: {VIDEO_SOURCE}")

        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        self.frame_counter = 0
        self.was_looped = False

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True  # Clean exit when main program stops
        self.thread.start()
        return self

    def _update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            if not grabbed:
                # Loop video if it is a local test file, otherwise stop
                if isinstance(VIDEO_SOURCE, str) and not VIDEO_SOURCE.startswith("rtsp://"):
                    self.stream.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self.read_lock:
                        self.was_looped = True
                    continue
                else:
                    self.started = False
                    break

            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.frame_counter += 1

    def read(self):
        """
        Returns the latest grabbed frame and decides if the frame should be processed
        based on the downsampling frame skip setting.
        """
        with self.read_lock:
            frame_to_return = self.frame.copy() if self.frame is not None else None
            should_process = (self.frame_counter % FRAME_SKIP == 0)

            looped = self.was_looped
            self.was_looped = False

        # Downsample resolution early to save CPU on transfer and processing
        if frame_to_return is not None:
            frame_to_return = cv2.resize(frame_to_return, (MODEL_WIDTH, MODEL_HEIGHT))

        return should_process, frame_to_return, looped

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()