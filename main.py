import cv2
import numpy as np
import time
#from config.settings import ZONE_DOOR, ZONE_INSIDE, MODEL_WIDTH, MODEL_HEIGHT
from config.settings import CROSSING_LINE, MODEL_WIDTH, MODEL_HEIGHT
from core.capture import RTSPStreamReader
from core.detector import PoseDetector
from core.tracker import ObjectTracker
from core.counter import FlowCounter
from core.calibration import AutoCalibrator

"""
def draw_debug_polygons(frame, door_poly, inside_poly, in_count, out_count):
    Draws custom perspective-aligned polygons and counters on screen.
    overlay = frame.copy()

    # Draw transparent polygons
    cv2.fillPoly(overlay, [door_poly], (0, 165, 255))   # Orange
    cv2.fillPoly(overlay, [inside_poly], (0, 255, 0))   # Green

    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw polygon outlines
    cv2.polylines(frame, [door_poly], True, (0, 165, 255), 2)
    cv2.polylines(frame, [inside_poly], True, (0, 255, 0), 2)

    # Draw labels near the top-left of each polygon bounding area
    cv2.putText(frame, "ZONE: DOOR", tuple(door_poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
    cv2.putText(frame, "ZONE: INSIDE", tuple(inside_poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Global HUD
    cv2.rectangle(frame, (10, 10), (280, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"INCOMING : {in_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"OUTGOING : {out_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
"""

def draw_debug_line(frame, active_line, in_count, out_count):
    """Draws the auto-calibrated crossing line and counters on screen."""
    if active_line is not None and len(active_line) >= 2:
        pt1 = tuple(active_line[0])
        pt2 = tuple(active_line[1])
        # Draw the virtual line (Red, thickness=3)
        cv2.line(frame, pt1, pt2, (0, 0, 255), 3)
        cv2.putText(frame, "CROSSING LINE", (pt1[0] - 50, pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Global HUD
    cv2.rectangle(frame, (10, 10), (280, 80), (0, 0, 0), -1)
    cv2.putText(frame, f"INCOMING : {in_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"OUTGOING : {out_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def main():
    print("[*] Initializing Edge AI Camera Processing Pipeline...")

    # 1. Initialize core system modules
    reader = RTSPStreamReader().start()
    detector = PoseDetector()
    tracker = ObjectTracker(detector.model)
    counter = FlowCounter()
    calibrator = AutoCalibrator()

    # Placeholder for calibrated pixel-space crossing line
    active_line = None

    # Persistent container to prevent UI flickering during frame skips
    annotated_frame = None

    # Target frame interval (e.g., 1 / 30 FPS = ~0.033 seconds) to control video playback speed
    frame_interval = 1.0 / 30.0

    print("[+] System online. Waiting for first valid frame...")
    print("[+] Core system successfully initialized. Starting processing loop...")
    print("[*] Press 'q' on the video window to stop execution.")

    try:
        cv2.namedWindow("Edge AI Portal - Auto-Calibrated Debugger", cv2.WINDOW_AUTOSIZE)
        while reader.started:

            start_time = time.time()

            # Read frame and check for video loop
            should_process, frame, looped = reader.read()

            if frame is None or not should_process:
                continue

            # Reset tracking memory if test video loops back to frame 0
            if looped:
                counter.reset()

            # Retrieve dimensions (expected to match MODEL_WIDTH/MODEL_HEIGHT)
            height, width, _ = frame.shape

            # One-time startup auto-calibration (aligns crossing line if camera moved)
            if active_line is None:
                print("[*] Performing startup automatic zone alignment...")
                active_line = calibrator.align_line(
                    frame, CROSSING_LINE, width, height
                )
                print("[+] Calibration complete! Crossing line mapped successfully.")

            # 2. Run single-pass inference and tracking
            # Execute tracking continuously on available frames to preserve ByteTrack IDs
            tracking_results = tracker.track_frame(frame)

            in_count, out_count = counter.update(
                tracking_results, width, height, active_line
            )

            # Generate overlay containing boxes
            if tracking_results is not None and tracking_results.boxes is not None:
                annotated_frame = tracking_results.plot(boxes=True, labels=True)
            else:
                annotated_frame = frame
            # Draw center point and X-position debug visualizer
            if tracking_results is not None and tracking_results.boxes is not None and tracking_results.boxes.id is not None:
                boxes = tracking_results.boxes.xyxy.cpu().numpy()
                track_ids = tracking_results.boxes.id.int().cpu().tolist()

                for i, track_id in enumerate(track_ids):
                    x1, y1, x2, y2 = boxes[i]
                    cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
                    norm_x = cx / width

                    # Draw center tracking keypoint
                    cv2.circle(annotated_frame, (cx, cy), 4, (0, 255, 255), -1)

                    # Display ID and current X position on top of the bounding box
                    cv2.putText(
                        annotated_frame,
                        f"ID #{track_id} (X:{cx}px | {norm_x:.2f})",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1
                    )

            # 3. Draw crossing line
            draw_debug_line(annotated_frame, active_line, counter.in_count, counter.out_count)

            # 4. Render window frame
            cv2.imshow("Edge AI Portal - Auto-Calibrated Debugger", annotated_frame)

            # Calculate actual processing time and wait for the remaining frame interval
            elapsed = time.time() - start_time
            sleep_time = max(1, int((frame_interval - elapsed) * 1000))

            # Check for close/exit key with dynamically calculated sleep time
            if cv2.waitKey(sleep_time) & 0xFF == ord('q'):
                print("[*] Exit signal received. Shutting down system...")
                break

    except KeyboardInterrupt:
        print("[!] Execution interrupted by terminal command.")

    finally:
        # Clean up threads and open windows
        reader.stop()
        cv2.destroyAllWindows()
        print("[+] System cleanup complete. Pipeline offline.")

if __name__ == "__main__":
    main()