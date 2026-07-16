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
        # Tracks the state history of each active ID. Format: { id: [state1, state2, ...] }
        self.user_history = {}
        # Tracks how many frames an ID has been missing/inactive
        self.absence_frames = {}
        # NEW: Tracks the cooldown frame counter for each ID to prevent ping-pong effect
        self.cooldown_counters = {}
        # NEW: Current frame processing index
        self.frame_index = 0

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

    def update(self, tracking_results, width, height, active_door_poly, active_inside_poly):
        """Updates tracking history with structural logging outputs."""
        self.frame_index += 1
        if tracking_results is None or tracking_results.boxes is None:
            return self.in_count, self.out_count

        # 1. Vérification de la détection globale
        num_detections = len(tracking_results.boxes)
        if num_detections > 0:
            print(f"\n--- [FRAME INFOS] {num_detections} personne(s) détectée(s) par YOLO ---")

        current_active_ids = []
        door_box = self._convert_poly_to_bbox(active_door_poly, width, height)
        inside_box = self._convert_poly_to_bbox(active_inside_poly, width, height)

        for i, box in enumerate(tracking_results.boxes):
            if box.id is None:
                print(f" [!] Personne #{i} détectée MAIS n'a pas encore reçu d'ID du Tracker (ByteTrack).")
                continue

            track_id = int(box.id[0].item())
            current_active_ids.append(track_id)

            # 2. Extraction du point de tracking (Hanches ou Bbox Center)
            keypoints = tracking_results.keypoints[i] if tracking_results.keypoints is not None else None
            point = self._get_tracking_point(keypoints)
            point_type = "Hanches"

            if point is None:
                xyxy = box.xyxy[0]
                point = (float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2))
                point_type = "Centre Bounding Box (Fallback)"

            norm_x, norm_y = point[0] / width, point[1] / height
            print(f" 👤 ID #{track_id} : Suivi via {point_type} à ({norm_x:.3f}, {norm_y:.3f})")

            # 3. Test de collision avec les zones
            in_door = self._is_inside_zone(point, door_box, width, height, "DOOR")
            in_inside = self._is_inside_zone(point, inside_box, width, height, "INSIDE")

            current_state = "NONE"
            if in_door and in_inside:
                current_state = "BOTH"
            elif in_door:
                current_state = "DOOR"
            elif in_inside:
                current_state = "INSIDE"

            if track_id not in self.user_history:
                self.user_history[track_id] = []

            history = self.user_history[track_id]

            # 4. Suivi du changement d'état
            if not history or history[-1] != current_state:
                if current_state != "NONE":
                    history.append(current_state)
                    if len(history) > 5:
                        history.pop(0)
                print(f"    ↳ Change d'état visuel -> Actuel: '{current_state}' | Historique récent: {history}")
            else:
                print(f"    ↳ Reste stable dans l'état: '{current_state}' | Historique récent: {history}")

            # 5. Évaluation de la séquence
            self._evaluate_state_sequence(track_id, history)

        # CLEANUP LOGIC WITH MEMORY BUFFER
        # For every ID currently active in this frame, reset its absence counter
        for uid in current_active_ids:
            self.absence_frames[uid] = 0

        # Check all stored IDs in our history
        all_stored_ids = list(self.user_history.keys())
        for uid in all_stored_ids:
            if uid not in current_active_ids:
                # Increment the absence frame counter for this missing ID
                self.absence_frames[uid] = self.absence_frames.get(uid, 0) + 1

                # If the ID has been missing for more than 30 frames (~1.5 seconds), delete it
                if self.absence_frames[uid] > 30:
                    print(f" 🗑️ ID #{uid} missing for too long. Permanently deleting history.")
                    del self.user_history[uid]
                    if uid in self.absence_frames:
                        del self.absence_frames[uid]
                    # NEW: Clean up the cooldown dictionary to prevent memory leaks
                    if uid in self.cooldown_counters:
                        del self.cooldown_counters[uid]

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
            print(f"🚀 [MATCH ENTRÉE !!!] -> {msg}")
            self._write_log(f"COUNT: {msg}")
            history.clear()

        # Look for exit pattern: INSIDE followed by DOOR
        elif len(history) >= 2 and history[-2] == 'INSIDE' and history[-1] == 'DOOR':
            self.out_count += 1
            self.cooldown_counters[track_id] = current_frame
            msg = f"User ID #{track_id} Exited."
            print(f"🛑 [MATCH SORTIE !!!] -> {msg}")
            self._write_log(f"COUNT: {msg}")
            history.clear()