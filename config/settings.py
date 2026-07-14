import os

#################### VIDEO SOURCE & STREAM SELECTION (The Magic Switch) ####################

# For local development and offline testing:
VIDEO_SOURCE = "data/test_video.mp4"

# For production deployment (Uncomment and configure your IP camera RTSP URL):
# VIDEO_SOURCE = "rtsp://admin:my_password@192.168.1.50:554/Streaming/Channels/101"



#################### OPTIMIZATIONS & DOWNSAMPLING ####################

FRAME_SKIP = 3             # Process 1 out of every 3 frames to divide CPU load
MODEL_WIDTH = 640          # Native YOLOv8 width resolution
MODEL_HEIGHT = 640         # Native YOLOv8 height resolution



#################### AI CONFIGURATION (YOLOv8 & OpenVINO) ####################

MODEL_NAME = "models/yolov8n-pose.pt"  # Will be auto-downloaded and exported to OpenVINO
CONFIDENCE_THRESHOLD = 0.5      # Minimum confidence score to detect a person



#################### VIRTUAL MATS GEOMETRY (Normalized coordinates from 0.0 to 1.0) ####################

# [ymin, xmin, ymax, xmax] coordinates relative to the image frame size.
ZONE_DOOR = [0.35, 0.10, 0.55, 0.90]    # Outer threshold zone (The Doorway)
ZONE_INSIDE = [0.55, 0.10, 0.75, 0.90]  # Inner room zone (Inside the store)



#################### PERSISTENCE & LOGGING ####################

DB_PATH = "data/camera.db"
LOG_FILE_PATH = "data/detection_logs.txt"