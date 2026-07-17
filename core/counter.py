import cv2
import os
import datetime
import numpy as np
from config.settings import LOG_FILE_PATH

class FlowCounter:
    """
    Implements double virtual mats logic using original normal zone checking
    with extensive debug print metrics to trace exact matching failures.
    """
    def __init__(self):
        self.in_count = 0
        self.out_count = 0
        # Track the logical side of each active ID. Format: { id: "LEFT" or "RIGHT" }
        self.id_states = {}
        # The X coordinate defining our virtual crossing line (midpoint)
        self.crossing_line_x = 0.45

        # Reduced safety margin (approx. 1% of screen width) to accept late-acquired IDs
        self.buffer = 0.01  # 1% of image width

        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    def _get_tracking_point(self, keypoints):
        """Extracts stable tracking point from hips or falls back to bbox center."""
        if keypoints is not None and len(keypoints.xy) > 0:
            xy = keypoints.xy[0]
            if len(xy) > 12:
                left_hip = xy[11]
                right_hip = xy[12]
                if left_hip[0] > 0 and right_hip[0] > 0:
                    x = float((left_hip[0] + right_hip[0]) / 2)
                    y = float((left_hip[1] + right_hip[1]) / 2)
                    return x, y
        return None

    def _is_inside_zone(self, point, zone_bbox, frame_width, frame_height, zone_name=""):
        """Checks if a point (x, y) lies inside a normalized zone bbox."""
        if point is None or zone_bbox is None:
            return False
        x, y = point
        norm_x = x / frame_width
        norm_y = y / frame_height

        ymin, xmin, ymax, xmax = zone_bbox
        is_in = (xmin <= norm_x <= xmax) and (ymin <= norm_y <= ymax)

        # DEBUG POINT POSITION
        # print(f"  [DEBUG MATCH {zone_name}] Point: ({norm_x:.2f}, {norm_y:.2f}) vs Zone Bounds: X[{xmin:.2f}-{xmax:.2f}] Y[{ymin:.2f}-{ymax:.2f}] -> Result: {is_in}")
        return is_in

    def _convert_poly_to_bbox(self, poly, width, height):
        """Helper to convert pixel polygons to normalized [ymin, xmin, ymax, xmax]"""
        if poly is None or len(poly) == 0:
            return None
        xmin_px = np.min(poly[:, 0])
        xmax_px = np.max(poly[:, 0])
        ymin_px = np.min(poly[:, 1])
        ymax_px = np.max(poly[:, 1])
        return [ymin_px / height, xmin_px / width, ymax_px / height, xmax_px / width]

    def _write_log(self, message):
        """Writes to file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"[!] Failed to write log: {e}")

    def update(self, tracking_results, width, height, active_line):
        current_active_ids = set()

        # Dynamically calculate the crossing boundary X coordinate based on the aligned line
        if active_line is not None and len(active_line) >= 2:
            self.crossing_line_x = float((active_line[0][0] + active_line[1][0]) / 2.0) / width
        else:
            self.crossing_line_x = 0.45 # Fallback midpoint if calibration not ready

        # Check if tracking_results is a list (typical of Ultralytics batch inference)
        # and grab the first element if so.
        results = tracking_results[0] if isinstance(tracking_results, list) else tracking_results

        # Ensure boxes and IDs exist in the frame results
        if results is not None and results.boxes is not None and results.boxes.id is not None:
            # Extract tracking IDs, normalized bounding boxes (xyxyn)
            track_ids = results.boxes.id.int().cpu().tolist()
            boxes_norm = results.boxes.xyxyn.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                current_active_ids.add(track_id)

                # Extract normalized x coordinates (center of bounding box)
                bbox = boxes_norm[i]
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                norm_x = float((x1 + x2) / 2.0)

                # DEBUG: Uncomment to see the exact positions of IDs in real-time
                print(f"[DEBUG ID #{track_id}] X Position: {norm_x:.3f} | Target Line: {self.crossing_line_x:.3f}")

                # Define hysteresis thresholds
                right_threshold = self.crossing_line_x + self.buffer
                left_threshold = self.crossing_line_x - self.buffer

                # Step 1: Handle first appearance of an ID
                if track_id not in self.id_states:
                    if norm_x > self.crossing_line_x:
                        self.id_states[track_id] = "RIGHT"
                    else:
                        # Late detection safe-check: if they appear very close to the line on the left (e.g. up to 0.37)
                        # we assume they came from the right but tracking was delayed, and we init them as RIGHT
                        if norm_x > (self.crossing_line_x - 0.08):
                            self.id_states[track_id] = "RIGHT"
                        else:
                            self.id_states[track_id] = "LEFT"
                    continue

                # Step 2: Track state changes (crossings)
                previous_state = self.id_states[track_id]

                # Case 1: Transition from RIGHT to LEFT -> ENTRY
                if previous_state == "RIGHT" and norm_x < left_threshold:
                    self.in_count += 1
                    self.id_states[track_id] = "LEFT"
                    msg = f"User ID #{track_id} crossed line entering."
                    print(f"[MATCH ENTRY !] -> {msg}")
                    self._write_log(f"COUNT: {msg}")

                # Case 2: Transition from LEFT to RIGHT -> EXIT
                elif previous_state == "LEFT" and norm_x > right_threshold:
                    self.out_count += 1
                    self.id_states[track_id] = "RIGHT"
                    msg = f"User ID #{track_id} crossed line exiting."
                    print(f"[MATCH EXIT !] -> {msg}")
                    self._write_log(f"COUNT: {msg}")

        # CLEANUP
        # Clean up IDs that are no longer tracked to free memory
        all_stored_ids = list(self.id_states.keys())
        for uid in all_stored_ids:
            if uid not in current_active_ids:
                del self.id_states[uid]

        return self.in_count, self.out_count

    def _evaluate_state_sequence(self, track_id, history):
        """
        Evaluates the recent state history of a track to detect crossings,
        incorporating a frame-based cooldown to completely eliminate the ping-pong effect.
        """
        # Check if the ID is currently under cooldown protection
        current_frame = self.frame_index
        last_match_frame = self.cooldown_counters.get(track_id, 0)

        if last_match_frame > 0 and (current_frame - last_match_frame) < 45:
            # ID is locked, ignore any transition to prevent double counting
            return

        # Look for entry pattern: DOOR followed by INSIDE
        if len(history) >= 2 and history[-2] == 'DOOR' and history[-1] == 'INSIDE':
            self.in_count += 1
            self.cooldown_counters[track_id] = current_frame
            msg = f"User ID #{track_id} Entered."
            print(f"[MATCH ENTRY !] -> {msg}")
            self._write_log(f"COUNT: {msg}")
            history.clear()

        # Look for exit pattern: INSIDE followed by DOOR
        elif len(history) >= 2 and history[-2] == 'INSIDE' and history[-1] == 'DOOR':
            self.out_count += 1
            self.cooldown_counters[track_id] = current_frame
            msg = f"User ID #{track_id} Exited."
            print(f"[MATCH EXIT !] -> {msg}")
            self._write_log(f"COUNT: {msg}")
            history.clear()