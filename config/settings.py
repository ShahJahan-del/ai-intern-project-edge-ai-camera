import os

#################### VIDEO SOURCE & STREAM SELECTION ####################

# For local development and offline testing:
VIDEO_SOURCE = "data/test_video_30min.mp4"

# For production deployment (Uncomment and configure your IP camera RTSP URL):
# VIDEO_SOURCE = "rtsp://admin:my_password@192.168.1.50:554/Streaming/Channels/101"



#################### OPTIMIZATIONS & DOWNSAMPLING ####################

FRAME_SKIP = 1         # Process 1 out of every 3 frames to divide CPU load
MODEL_WIDTH = 640          # Native YOLOv8 width resolution
MODEL_HEIGHT = 640         # Native YOLOv8 height resolution


#################### AI CONFIGURATION (YOLOv8 & OpenVINO) ####################

MODEL_NAME = "models/yolov8n.pt"  # Will be auto-downloaded and exported to OpenVINO
CONFIDENCE_THRESHOLD = 0.15      # Minimum confidence score to detect a person



#################### VIRTUAL POLYGONAL ZONES (Normalized coordinates [x, y]) ####################

### Void zone in between ###

# ZONE_DOOR: Covers the background glass door (Center-right of the active hallway)
#ZONE_DOOR = [
#    [0.51, 0.05],  # Top-left
#    [0.57, 0.05],  # Top-right
#    [0.57, 0.45],  # Bottom-right
#    [0.51, 0.45]   # Bottom-left
#]

# ZONE_INSIDE: Covers the staircase/left hallway entrance (Center-left of the active hallway)
#ZONE_INSIDE = [
#    [0.30, 0.05],  # Top-left
#    [0.40, 0.05],  # Top-right
#    [0.40, 0.65],  # Bottom-right
#    [0.30, 0.65]   # Bottom-left
#]

### No void zone in between ###

# ZONE_DOOR: Covers the background glass door (Right part of the hallway, from 0.45 to 0.58)
#ZONE_DOOR = [
#    [0.45, 0.05],  # Top-left
#    [0.58, 0.05],  # Top-right
#    [0.58, 0.95],  # Bottom-right
#    [0.45, 0.95]   # Bottom-left
#]

# ZONE_INSIDE: Covers the staircase/left hallway entrance (Left part, from 0.30 to 0.45)
#ZONE_INSIDE = [
#    [0.30, 0.05],  # Top-left
#    [0.45, 0.05],  # Top-right
#    [0.45, 0.95],  # Bottom-right
#    [0.30, 0.95]   # Bottom-left
#]

#################### VIRTUAL CROSSING LINE (Normalized coordinates [x, y]) ####################

# CROSSING_LINE: A single vertical line at X = 0.45, spanning from top to bottom
CROSSING_LINE = [
    [0.45, 0.05],  # Top point (x, y)
    [0.45, 0.95]   # Bottom point (x, y)
]

#################### PERSISTENCE, LOGGING & CALIBRATION ####################

DB_PATH = "data/camera.db"
LOG_FILE_PATH = "data/detection_logs.txt"
REFERENCE_IMAGE_PATH = "data/reference.jpg"  # Base image containing the ideal setup