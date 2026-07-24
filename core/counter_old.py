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
        self.frame_count = 0
        self.crossing_line_x = 0.45

        # Hysteresis buffer: 5% threshold around line to prevent line-flicker
        self.buffer = 0.03

        # Stores master chain states:
        # { chain_id: {"origin_side": str, "last_x": float, "last_frame": int, "counted": bool} }
        self.chains_db = {}

        # Maps current live track_id -> chain_id
        self.track_to_chain = {}

        self.next_chain_id = 1

        # Stores active track states:
        # { track_id: {"start_side": str, "counted": bool} }
        self.tracks_memory = {}

        # Ensure the logs directory exists
        os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

    def reset(self):
        """Resets state memory when video loops."""
        self.chains_db.clear()
        self.track_to_chain.clear()
        self.next_chain_id = 1

        self.tracks_memory.clear()
        self.frame_count = 0
        print("[INFO] Counter state cleared for video loop.")

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
        self.frame_count += 1
        current_active_ids = set()

        # Dynamically calculate the crossing boundary X coordinate based on the aligned line
        if active_line is not None and len(active_line) >= 2:
            self.crossing_line_x = float((active_line[0][0] + active_line[1][0]) / 2.0) / width
        else:
            self.crossing_line_x = 0.45 # Fallback midpoint if calibration not ready

        # Check if tracking_results is a list and grab the first element if so.
        results = tracking_results[0] if isinstance(tracking_results, list) else tracking_results

        # Ensure boxes and IDs exist in the frame results
        if results is not None and results.boxes is not None and results.boxes.id is not None:
            track_ids = results.boxes.id.int().cpu().tolist()
            boxes_norm = results.boxes.xyxyn.cpu().numpy()

            for i, track_id in enumerate(track_ids):
                current_active_ids.add(track_id)

                # Extract normalized x coordinates (center of bounding box)
                bbox = boxes_norm[i]
                norm_x = float((bbox[0] + bbox[2]) / 2.0)

                # Define hysteresis thresholds
                right_threshold = self.crossing_line_x + self.buffer
                left_threshold = self.crossing_line_x - self.buffer

                # DEBUG: X coordinate of
                print(f"ID #{track_id} located at x = {norm_x:.3f} | Crossing Line located at x = {self.crossing_line_x:.3f}")

                # Step 1: Resolve Chain ID for this Track ID
                if track_id not in self.track_to_chain:
                    matched_chain_id = None

                    # Search for recent lost chains close in spatial position (< 5% screen distance, lost within 15 frames)
                    for c_id, c_data in self.chains_db.items():
                        frames_since_last = self.frame_count - c_data["last_frame"]
                        if 1 <= frames_since_last <= 15:
                            spatial_dist = abs(norm_x - c_data["last_x"])
                            if spatial_dist < 0.05:  # Strict 5% width limit
                                matched_chain_id = c_id
                                break

                    if matched_chain_id is not None:
                        # Re-link existing chain (preserves 'counted' state!)
                        self.track_to_chain[track_id] = matched_chain_id
                        chain_id = matched_chain_id
                        print(f"[CHAIN LINK] Track #{track_id} -> Linked to Chain #{chain_id} (Origin: {self.chains_db[chain_id]['origin_side']})")
                    else:
                        # Create a brand new chain
                        chain_id = self.next_chain_id
                        self.next_chain_id += 1
                        origin_side = "RIGHT" if norm_x > self.crossing_line_x else "LEFT"

                        self.chains_db[chain_id] = {
                            "origin_side": origin_side,
                            "last_x": norm_x,
                            "last_frame": self.frame_count,
                            "counted": False
                        }
                        self.track_to_chain[track_id] = chain_id
                        print(f"[NEW CHAIN] Chain #{chain_id} created for Track #{track_id} at norm_X={norm_x:.3f} (Origin: {origin_side})")
                else:
                    chain_id = self.track_to_chain[track_id]

                # Step 2: Update Chain position and check crossing
                chain = self.chains_db[chain_id]
                chain["last_x"] = norm_x
                chain["last_frame"] = self.frame_count

                # Evaluate crossing ONLY IF the chain has not been counted yet
                if not chain["counted"]:
                    origin = chain["origin_side"]

                    # ENTRY: Came from RIGHT, now clearly on LEFT
                    if origin == "RIGHT" and norm_x < (self.crossing_line_x - self.buffer):
                        self.in_count += 1
                        chain["counted"] = True  # Permanently mark chain as counted!
                        msg = f"Chain #{chain_id} (Track #{track_id}) completed ENTRY path."
                        print(f"[MATCH ENTRY !] -> {msg} (norm_X={norm_x:.3f}, Line X={self.crossing_line_x:.3f})")
                        self._write_log(f"COUNT: {msg}")

                    # EXIT: Came from LEFT, now clearly on RIGHT
                    elif origin == "LEFT" and norm_x > (self.crossing_line_x + self.buffer):
                        self.out_count += 1
                        chain["counted"] = True  # Permanently mark chain as counted!
                        msg = f"Chain #{chain_id} (Track #{track_id}) completed EXIT path."
                        print(f"[MATCH EXIT !] -> {msg} (norm_X={norm_x:.3f}, Line X={self.crossing_line_x:.3f})")
                        self._write_log(f"COUNT: {msg}")

        # Clean track mappings no longer visible
        dead_tracks = [tid for tid in self.track_to_chain.keys() if tid not in current_active_ids]
        for tid in dead_tracks:
            del self.track_to_chain[tid]

        # Step 3: Cleanup & Ghost Crossing Evaluation
        dead_chains = []
        for cid, cdata in self.chains_db.items():
            time_lost = self.frame_count - cdata["last_frame"]

            # If the chain has disappeared (between 1 and 10 frames) and has not yet been counted :
            if 1 <= time_lost <= 10 and not cdata["counted"]:
                last_x = cdata["last_x"]
                origin = cdata["origin_side"]

                # If the person disappeared very close to the line (ex: lower than 0.04 from the line)
                if abs(last_x - self.crossing_line_x) < 0.04:
                    if origin == "RIGHT" and last_x < self.crossing_line_x + 0.03:
                        self.in_count += 1
                        cdata["counted"] = True
                        msg = f"Chain #{cid} (GHOST ENTRY) cross validated near line at norm_X={last_x:.3f}."
                        print(f"[MATCH ENTRY (GHOST) !] -> {msg}")
                        self._write_log(f"COUNT: {msg}")

                    elif origin == "LEFT" and last_x > self.crossing_line_x - 0.03:
                        self.out_count += 1
                        cdata["counted"] = True
                        msg = f"Chain #{cid} (GHOST EXIT) cross validated near line at norm_X={last_x:.3f}."
                        print(f"[MATCH EXIT (GHOST) !] -> {msg}")
                        self._write_log(f"COUNT: {msg}")

            if time_lost > 30:
                dead_chains.append(cid)

        for cid in dead_chains:
            del self.chains_db[cid]

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