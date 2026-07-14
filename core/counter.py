from config.settings import ZONE_DOOR, ZONE_INSIDE

class FlowCounter:
    """
    Implements double virtual mats logic and a state machine to track
    customer directions (In / Out) and handle hesitations.
    """
    def __init__(self):
        self.in_count = 0
        self.out_count = 0
        # Tracks the state history of each active ID. Format: { id: [state1, state2, ...] }
        self.user_history = {}

    def _get_tracking_point(self, keypoints):
        """
        Extracts a stable tracking point from pose keypoints.
        We use the average coordinates of the Left Hip (index 11) and Right Hip (index 12).
        If hips are not visible, we fall back to the bounding box center.
        """
        if keypoints is not None and len(keypoints.xy) > 0:
            xy = keypoints.xy[0] # Keypoints for the first person detected
            if len(xy) > 12:
                left_hip = xy[11]
                right_hip = xy[12]
                # Check if points are detected (non-zero)
                if left_hip[0] > 0 and right_hip[0] > 0:
                    x = float((left_hip[0] + right_hip[0]) / 2)
                    y = float((left_hip[1] + right_hip[1]) / 2)
                    return x, y
        return None

    def _is_inside_zone(self, point, zone, frame_width, frame_height):
        """
        Checks if a point (x, y) lies inside a normalized zone [ymin, xmin, ymax, xmax].
        """
        if point is None:
            return False
        x, y = point
        # Convert point to normalized coordinates (0.0 to 1.0)
        norm_x = x / frame_width
        norm_y = y / frame_height

        ymin, xmin, ymax, xmax = zone
        return (xmin <= norm_x <= xmax) and (ymin <= norm_y <= ymax)

    def update(self, tracking_results, width, height):
        """
        Updates tracking history and transitions state machines to count movements.
        """
        if tracking_results is None or tracking_results.boxes is None:
            return self.in_count, self.out_count

        current_active_ids = []

        # Loop through detected persons
        for i, box in enumerate(tracking_results.boxes):
            if box.id is None:
                continue

            track_id = int(box.id[0].item())
            current_active_ids.append(track_id)

            keypoints = tracking_results.keypoints[i] if tracking_results.keypoints is not None else None
            point = self._get_tracking_point(keypoints)

            if point is None:
                # Fallback to bounding box center
                xyxy = box.xyxy[0]
                point = (float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2))

            # Identify current sub-zone presence
            in_door = self._is_inside_zone(point, ZONE_DOOR, width, height)
            in_inside = self._is_inside_zone(point, ZONE_INSIDE, width, height)

            # Define localized state
            current_state = "NONE"
            if in_door and in_inside:
                current_state = "BOTH"
            elif in_door:
                current_state = "DOOR"
            elif in_inside:
                current_state = "INSIDE"

            # Skip updating if state is unchanged to avoid duplicate logs
            if track_id not in self.user_history:
                self.user_history[track_id] = []

            history = self.user_history[track_id]
            if not history or history[-1] != current_state:
                if current_state != "NONE":
                    history.append(current_state)
                    # Limit memory footprint per ID
                    if len(history) > 5:
                        history.pop(0)

            # Evaluate state transition sequences to register counts
            self._evaluate_state_sequence(track_id, history)

        # Clean up stale IDs that left the camera frame
        stale_ids = [uid for uid in list(self.user_history.keys()) if uid not in current_active_ids]
        for uid in stale_ids:
            del self.user_history[uid]

        return self.in_count, self.out_count

    def _evaluate_state_sequence(self, track_id, history):
        """
        Analyzes the historical transition sequence of a unique user ID to trigger counts.
        """
        if len(history) < 2:
            return

        # 🟢 ENTRANCE LOGIC: DOOR -> BOTH -> INSIDE or DOOR -> INSIDE
        if history[-2:] == ["DOOR", "INSIDE"] or (len(history) >= 3 and history[-3:] == ["DOOR", "BOTH", "INSIDE"]):
            self.in_count += 1
            print(f"[+] COUNT: User ID #{track_id} Entered the store.")
            history.clear() # Reset history for this user to avoid double counting

        # 🔴 EXIT LOGIC: INSIDE -> BOTH -> DOOR or INSIDE -> DOOR
        elif history[-2:] == ["INSIDE", "DOOR"] or (len(history) >= 3 and history[-3:] == ["INSIDE", "BOTH", "DOOR"]):
            self.out_count += 1
            print(f"[-] COUNT: User ID #{track_id} Exited the store.")
            history.clear() # Reset history for this user