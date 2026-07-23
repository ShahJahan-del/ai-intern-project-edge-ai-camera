import cv2
import os

# Define file paths
input_video_path = "data/test_video_5min_compressed.mp4" # Put the exact name of your 9h video here
output_dir = "data"
output_video_path = os.path.join(output_dir, "test_video_roboflow.mp4")

# Define target slice (Example: starting at 1 hour 15 minutes, lasting 30 minutes)
start_hour = 0
start_minute = 1
start_second = 0
duration_minutes = 4

# Calculate target frame boundaries based on timestamps
start_time_seconds = (start_hour * 3600) + (start_minute * 60) + start_second
duration_seconds = duration_minutes * 60

print(f"[*] Opening source video: {input_video_path}...")
cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"[-] Error: Could not open {input_video_path}. Check the filename!")
    exit()

# Extract video metadata properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate frame indexes
start_frame = int(start_time_seconds * fps)
end_frame = int(start_frame + (duration_seconds * fps))

print(f"[+] Video Properties: {width}x{height} @ {fps} FPS")
print(f"[*] Extracting frames from index {start_frame} to {end_frame}...")

# Fast-forward the video reader pointer directly to the start frame index
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Initialize standard video writer engine
os.makedirs(output_dir, exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Standard MP4 codec
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

current_frame = start_frame
print("[*] Processing... This might take a few minutes depending on your disk speed.")

while cap.isOpened() and current_frame <= end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    current_frame += 1

    # Visual progress updates every 2000 frames
    if current_frame % 2000 == 0:
        progress = ((current_frame - start_frame) / (end_frame - start_frame)) * 100
        print(f"    -> Progress: {progress:.1f}% completed")

# Resource cleanup
cap.release()
out.release()
print(f"[+] Success! Slice saved cleanly at: {output_video_path}")