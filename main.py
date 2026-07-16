import cv2
import numpy as np
from config.settings import ZONE_DOOR, ZONE_INSIDE, MODEL_WIDTH, MODEL_HEIGHT
from core.capture import RTSPStreamReader
from core.detector import PoseDetector
from core.tracker import ObjectTracker
from core.counter import FlowCounter
from core.calibration import AutoCalibrator

def draw_debug_polygons(frame, door_poly, inside_poly, in_count, out_count):
    """Draws custom perspective-aligned polygons and counters on screen."""
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

def main():
    print("[*] Initializing Edge AI Camera Processing Pipeline...")

    # 1. Initialize core system modules
    reader = RTSPStreamReader().start()
    detector = PoseDetector()
    tracker = ObjectTracker(detector.model)
    counter = FlowCounter()
    calibrator = AutoCalibrator()

    # Placeholders for calibrated pixel-space polygons
    active_door_poly = None
    active_inside_poly = None

    # Persistent container to prevent UI flickering during frame skips
    annotated_frame = None

    print("[+] System online. Waiting for first valid frame...")
    print("[+] Core system successfully initialized. Starting processing loop...")
    print("[*] Press 'q' on the video window to stop execution.")

    try:
        cv2.namedWindow("Edge AI Portal - Auto-Calibrated Debugger", cv2.WINDOW_AUTOSIZE)
        while reader.started:
            # Read frame from the threaded buffer
            should_process, frame = reader.read()

            if frame is None:
                continue

            # Retrieve dimensions (expected to match MODEL_WIDTH/MODEL_HEIGHT)
            height, width, _ = frame.shape

            # One-time startup auto-calibration (aligns zones if camera moved)
            if active_door_poly is None:
                print("[*] Performing startup automatic zone alignment...")
                active_door_poly, active_inside_poly = calibrator.align_zones(
                    frame, ZONE_DOOR, ZONE_INSIDE, width, height
                )
                print("[+] Calibration complete! Polygonal zones mapped successfully.")

            # Replaces the calibration above (if you want to deactivate calibration)
            # Instead of using calibrator.align_zones, convert directly
            # Normalized coordinates of settings.py with real pixels
            #if active_door_poly is None:
            #    print("[*] Using robust static polygonal zones...")
            #    active_door_poly = np.array([[int(p[0] * width), int(p[1] * height)] for p in ZONE_DOOR], dtype=np.int32)
            #    active_inside_poly = np.array([[int(p[0] * width), int(p[1] * height)] for p in ZONE_INSIDE], dtype=np.int32)
            #    print("[+] Static zones initialized successfully.")

            # 2. Run inference and tracking only on selected frames (frame skipping)
            if should_process:
                # Track the frame. This automatically executes YOLOv8-Pose with ByteTrack
                tracking_results = tracker.track_frame(frame)

                in_count, out_count = counter.update(
                    tracking_results, width, height, active_door_poly, active_inside_poly
                )

                # Generate the overlay containing skeletons and boxes
                if tracking_results is not None and tracking_results.boxes is not None:
                    annotated_frame = tracking_results.plot(boxes=True, kpt_line=True, labels=True)
                else:
                    annotated_frame = frame.copy()

            # If we skip AI inference, fallback to the raw frame to maintain fluid stream motion
            elif annotated_frame is None:
                annotated_frame = frame.copy()
            else:
                # We reuse the previous frame's visual annotations but update the background matrix pixels
                # to prevent time lag while maintaining visible skeletons
                annotated_frame = frame.copy()

            # 3. Draw diagnostic elements (zones overlay + counters HUD) on the annotated display layer
            draw_debug_polygons(annotated_frame, active_door_poly, active_inside_poly, counter.in_count, counter.out_count)

            # 4. Render window frame
            cv2.imshow("Edge AI Portal - Auto-Calibrated Debugger", annotated_frame)

            # Check for close/exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
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