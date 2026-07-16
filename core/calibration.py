import cv2
import numpy as np
import os
from config.settings import REFERENCE_IMAGE_PATH

class AutoCalibrator:
    """
    Uses ORB feature matching and Homography calculation to automatically align
    polygonal detection zones if the physical camera has drifted or rotated.
    """
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=1000)
        # Use Brute-Force Matcher with Hamming distance (perfect for ORB)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_img_gray = None

        if os.path.exists(REFERENCE_IMAGE_PATH):
            self.ref_img = cv2.imread(REFERENCE_IMAGE_PATH)
            self.ref_img_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
            self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.ref_img_gray, None)
            print("[+] Loaded calibration reference image.")
        else:
            print("[!] No reference image found. The first frame will be saved as reference.")

    def save_reference(self, frame):
        """Saves the current frame as the golden reference image."""
        os.makedirs(os.path.dirname(REFERENCE_IMAGE_PATH), exist_ok=True)
        cv2.imwrite(REFERENCE_IMAGE_PATH, frame)
        self.ref_img = frame.copy()
        self.ref_img_gray = cv2.cvtColor(self.ref_img, cv2.COLOR_BGR2GRAY)
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.ref_img_gray, None)
        print(f"[+] Saved new startup reference image to '{REFERENCE_IMAGE_PATH}'")

    def align_zones(self, current_frame, zone_door, zone_inside, width, height):
        """
        Calculates homography matrix and transforms normalized polygons
        to match the current frame's perspective.
        """
        # If no reference image exists yet, initialize it and return original zones
        if self.ref_img_gray is None:
            self.save_reference(current_frame)
            return self._denormalize(zone_door, width, height), self._denormalize(zone_inside, width, height)

        # Process current frame
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        curr_kp, curr_des = self.orb.detectAndCompute(current_gray, None)

        if curr_des is None or len(curr_kp) < 10:
            print("[!] Warning: Not enough features detected. Using fallback coordinates.")
            return self._denormalize(zone_door, width, height), self._denormalize(zone_inside, width, height)

        # Match keypoints
        matches = self.bf.match(self.ref_descriptors, curr_des)
        matches = sorted(matches, key=lambda x: x.distance)

        # We need at least 15 solid matches to compute a reliable Homography matrix
        if len(matches) < 15:
            print("[!] Warning: Camera shift too severe or scene too dark. Fallback active.")
            return self._denormalize(zone_door, width, height), self._denormalize(zone_inside, width, height)

        # Extract coordinates of matching points
        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([curr_kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute Homography Matrix using RANSAC to filter outliers
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            print("[!] Error: Homography calculation failed. Using default coordinates.")
            return self._denormalize(zone_door, width, height), self._denormalize(zone_inside, width, height)

        # Warp normalized polygon points
        aligned_door = self._warp_polygon(zone_door, H, width, height)
        aligned_inside = self._warp_polygon(zone_inside, H, width, height)

        return aligned_door, aligned_inside

    def _denormalize(self, zone, width, height):
        """Converts normalized [0.0 - 1.0] polygon coordinates to pixel spaces."""
        return np.array([[int(p[0] * width), int(p[1] * height)] for p in zone], dtype=np.int32)

    def _warp_polygon(self, normalized_polygon, H, width, height):
        """Applies homography matrix transformation to a normalized polygon."""
        # Step 1: Denormalize to reference pixel space
        pixel_poly = self._denormalize(normalized_polygon, width, height)

        # Step 2: Reshape points for OpenCV perspectiveTransform input requirements
        points = np.array(pixel_poly, dtype=np.float32).reshape(-1, 1, 2)

        # Step 3: Compute spatial deformation
        transformed_points = cv2.perspectiveTransform(points, H)

        # Step 4: Cast back to integer coordinates
        return np.array(transformed_points.reshape(-1, 2), dtype=np.int32)