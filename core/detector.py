import os
from ultralytics import YOLO
from config.settings import MODEL_NAME, CONFIDENCE_THRESHOLD

class PoseDetector:
    """
    Handles loading, exporting (to OpenVINO format), and executing
    YOLOv8 model for object detection.
    """
    def __init__(self):
        print(f"[+] Loading core AI model: {MODEL_NAME}...")

        """
        # Load standard PyTorch model first
        self.model = YOLO(MODEL_NAME)

        # Define expected OpenVINO directory name
        openvino_model_path = MODEL_NAME.replace(".pt", "_openvino_model")

        # Auto-export to OpenVINO if not already done (highly optimized for CPU)
        if not os.path.exists(openvino_model_path):
            print("[*] Optimizing model for CPU execution (Exporting to OpenVINO FP16)...")
            self.model.export(format="openvino", half=True)

        # Reload the optimized model
        self.model = YOLO(openvino_model_path)
        """

        # Check model folder exists
        if not os.path.exists(MODEL_NAME):
            raise FileNotFoundError(
                f"[!] Error: Model path '{MODEL_NAME}' not found. "
                "Ensure your OpenVINO folder is placed inside the 'models/' directory."
            )

        # Load folder downloaded from Colab
        self.model = YOLO(MODEL_NAME, task="detect")

        print("[+] Model loaded successfully on optimized CPU runtime.")

    def detect(self, frame):
        """
        Runs inference on a single frame. Returns predictions.
        """
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        return results[0] if results else None