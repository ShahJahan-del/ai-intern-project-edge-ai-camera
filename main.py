import cv2
import numpy as np
from config.settings import ZONE_DOOR, ZONE_INSIDE, MODEL_WIDTH, MODEL_HEIGHT
from core.capture import RTSPStreamReader
from core.detector import PoseDetector
from core.tracker import ObjectTracker
from core.counter import FlowCounter

def draw_debug_zones(frame, width, height, in_count, out_count):
    """
    Draws the virtual mats (DOOR and INSIDE) and current count statistics
    on the screen for manual adjustment and calibration.
    """
    # Convert normalized zone coordinates to pixel values
    def get_pixel_coords(zone):
        ymin, xmin, ymax, xmax = zone
        return (int(xmin * width), int(ymin * height)), (int(xmax * width), int(ymax * height))

    door_start, door_end = get_pixel_coords(ZONE_DOOR)
    inside_start, inside_end = get_pixel_coords(ZONE_INSIDE)

    # Semi-transparent overlay for zones
    overlay = frame.copy()

    # Red/Orange for Doorway, Green for Store Inside
    cv2.rectangle(overlay, door_start, door_end, (0, 165, 255), -1)   # BGR Orange
    cv2.rectangle(overlay, inside_start, inside_end, (0, 255, 0), -1)  # BGR Green

    # Apply overlay with transparency
    alpha = 0.2
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Draw outlines
    cv2.rectangle(frame, door_start, door_end, (0, 165, 255), 2)
    cv2.rectangle(frame, inside_start, inside_end, (0, 255, 0), 2)

    # Draw Zone Labels
    cv2.putText(frame, "ZONE: DOOR", (door_start[0] + 10, door_start[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    cv2.putText(frame, "ZONE: INSIDE", (inside_start[0] + 10, inside_start[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw Global HUD / Counters
    cv2.rectangle(frame, (10, 10), (280, 80), (0, 0, 0), -1) # Dark background
    cv2.putText(frame, f"INCOMING : {in_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"OUTGOING : {out_count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

def main():
    print("[*] Initializing Edge AI Camera Processing Pipeline...")

    # 1. Initialize core system modules
    reader = RTSPStreamReader().start()
    detector = PoseDetector()
    tracker = ObjectTracker(detector.model)
    counter = FlowCounter()

    print("[+] Core system successfully initialized. Starting processing loop...")
    print("[*] Press 'q' on the video window to stop execution.")

    try:
        while reader.started:
            # Read frame from the threaded buffer
            should_process, frame = reader.read()

            if frame is None:
                continue

            # Retrieve dimensions (expected to match MODEL_WIDTH/MODEL_HEIGHT due to capture downsampling)
            height, width, _ = frame.shape

            # 2. Run inference and tracking only on selected frames (frame skipping)
            if should_process:
                # Track the frame. This automatically executes YOLOv8-Pose with ByteTrack
                tracking_results = tracker.track_frame(frame)

                # Update our virtual mats count logic with tracking coordinate outputs
                in_count, out_count = counter.update(tracking_results, width, height)

                # Optional: Render active skeletons/boxes for real-time visual inspection
                if tracking_results is not None and tracking_results.boxes is not None:
                    # Leverage Ultralytics internal plotter for simple debugging
                    frame = tracking_results.plot(boxes=True, kpt_line=True, labels=True)

            # 3. Draw diagnostic elements (zones overlay + counters HUD)
            draw_debug_zones(frame, width, height, counter.in_count, counter.out_count)

            # 4. Render window frame
            cv2.imshow("Edge AI Portal - Production Debugger", frame)

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