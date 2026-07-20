from config.settings import CONFIDENCE_THRESHOLD

class ObjectTracker:
    """
    Wraps Ultralytics' internal ByteTrack integration to persist IDs
    over temporal camera frame sequences.
    """
    def __init__(self, model):
        self.model = model

    def track_frame(self, frame):
        """
        Runs tracking on the frame. Returns tracking results with persistent IDs.
        """
        # Persist = True enables tracking across frames, using ByteTrack
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE_THRESHOLD,
            classes=[0],
            verbose=False
        )
        if isinstance(results, list):
            return results[0] if len(results) > 0 else None
        return results