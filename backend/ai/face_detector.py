import pathlib
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

# Ensure MediaPipe model paths exist
MODEL_DIR = pathlib.Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FULL_RANGE_MODEL = MODEL_DIR / "blaze_face_full_range.tflite"
if not FULL_RANGE_MODEL.exists():
    try:
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
        print("[+] Downloading blaze_face_full_range.tflite model...")
        urllib.request.urlretrieve(url, FULL_RANGE_MODEL)
    except Exception as e:
        print(f"[!] Warning: Could not download full range model: {e}")

SHORT_RANGE_MODEL = MODEL_DIR / "blaze_face_short_range.tflite"

FACE_LANDMARKER_MODEL = MODEL_DIR / "face_landmarker.task"
if not FACE_LANDMARKER_MODEL.exists():
    try:
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        print("[+] Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(url, FACE_LANDMARKER_MODEL)
    except Exception as e:
        print(f"[!] Warning: Could not download face_landmarker.task: {e}")

# Initialize Face Detectors
model_to_use = FULL_RANGE_MODEL if FULL_RANGE_MODEL.exists() else SHORT_RANGE_MODEL
full_options = vision.FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=str(model_to_use)),
    min_detection_confidence=0.3
)
FULL_DETECTOR = vision.FaceDetector.create_from_options(full_options)

SHORT_DETECTOR = None
if SHORT_RANGE_MODEL.exists():
    short_options = vision.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(SHORT_RANGE_MODEL)),
        min_detection_confidence=0.15
    )
    SHORT_DETECTOR = vision.FaceDetector.create_from_options(short_options)

# Initialize Face Landmarker
LANDMARKER = None
if FACE_LANDMARKER_MODEL.exists():
    lm_options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL)),
        running_mode=vision.RunningMode.IMAGE
    )
    LANDMARKER = vision.FaceLandmarker.create_from_options(lm_options)

# SIFT Feature Extractor
SIFT = cv2.SIFT_create(nfeatures=128)


def extract_face_embedding(crop, target_size=(64, 64)):
    """Computes normalized feature vector combining 3D MediaPipe Facial Landmarks (1434-d),
    SIFT geometric structure (128-d), and HSV color histogram (256-d).
    """
    crop_resized = cv2.resize(crop, target_size)

    # 1. 3D Facial Landmark Feature Vector (1434-d)
    landmark_vec = None
    if LANDMARKER is not None:
        try:
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            crop_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_rgb))
            lm_res = LANDMARKER.detect(crop_mp)
            if lm_res and lm_res.face_landmarks:
                lms = lm_res.face_landmarks[0]
                coords = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)
                coords_centered = coords - np.mean(coords, axis=0)
                scale = np.max(np.linalg.norm(coords_centered, axis=1)) + 1e-7
                landmark_vec = (coords_centered / scale).flatten()
                landmark_vec /= (np.linalg.norm(landmark_vec) + 1e-7)
        except Exception:
            landmark_vec = None

    if landmark_vec is None:
        landmark_vec = np.zeros((1434,), dtype=np.float32)

    # 2. SIFT Invariant Feature Vector (128-d)
    gray_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = SIFT.detectAndCompute(gray_crop, None)
    if descriptors is not None and len(descriptors) > 0:
        sift_vec = np.mean(descriptors, axis=0)
        sift_vec /= (np.linalg.norm(sift_vec) + 1e-7)
    else:
        sift_vec = np.zeros((128,), dtype=np.float32)

    # 3. HSV Color Histogram Feature Vector (256-d)
    hsv_crop = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_crop], [0, 1], None, [16, 16], [0, 180, 0, 256])
    hist_feat = cv2.normalize(hist, hist).flatten()

    # Combine: 60% 3D Facial Landmark geometry + 25% SIFT structure + 15% HSV color
    combined_feat = np.hstack([landmark_vec * 0.60, sift_vec * 0.25, hist_feat * 0.15])
    combined_feat /= (np.linalg.norm(combined_feat) + 1e-7)
    return combined_feat


def detect_faces_with_central_priority(img, target_size=(64, 64)):
    """Detects faces in an image and identifies the central/main subject face.
    Returns:
        total_faces: int (number of faces detected in the image)
        central_feature: np.ndarray or None (facial landmark feature vector of primary central person)
        all_features: list of np.ndarray (facial landmark feature vectors of all faces detected)
    """
    h_img, w_img = img.shape[:2]
    img_cx, img_cy = w_img / 2.0, h_img / 2.0
    max_dist = np.sqrt(img_cx**2 + img_cy**2) + 1e-7

    # Convert BGR image to MediaPipe Image format
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

    detection_result = FULL_DETECTOR.detect(mp_image)

    # Fallback to short-range detector if no faces found
    if (not detection_result or not detection_result.detections) and SHORT_DETECTOR:
        detection_result = SHORT_DETECTOR.detect(mp_image)

    candidates = []

    if detection_result and detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            # Add 10% padding margin around detected face box
            margin_w, margin_h = int(w * 0.1), int(h * 0.1)
            x_min, y_min = max(0, x - margin_w), max(0, y - margin_h)
            x_max, y_max = min(w_img, x + w + margin_w), min(h_img, y + h + margin_h)

            crop = img[y_min:y_max, x_min:x_max]
            if crop.size > 0:
                feat = extract_face_embedding(crop, target_size)

                # Calculate centrality score (combining face area, detection score, and distance to image center)
                fcx, fcy = x + w / 2.0, y + h / 2.0
                dist = np.sqrt((fcx - img_cx)**2 + (fcy - img_cy)**2)
                norm_dist = dist / max_dist
                score = detection.categories[0].score if detection.categories else 0.5
                area = w * h

                centrality_score = (area * score) / (1.0 + 2.0 * norm_dist)

                candidates.append({
                    "feature": feat,
                    "centrality_score": centrality_score,
                    "bbox": (x, y, w, h)
                })

    if not candidates:
        return 0, None, []

    # Sort candidates by centrality score descending to pick main central person
    candidates.sort(key=lambda c: c["centrality_score"], reverse=True)
    central_feature = candidates[0]["feature"]
    all_features = [c["feature"] for c in candidates]

    return len(candidates), central_feature, all_features


def detect_all_faces_and_features(img, target_size=(64, 64)):
    """Detects ALL faces in an image using MediaPipe Face Detector."""
    num_faces, _, all_features = detect_faces_with_central_priority(img, target_size=target_size)
    return all_features


def detect_faces_and_features(img, existing_features=None, match_threshold=1.0, target_size=(64, 64)):
    """Backward compatible wrapper."""
    return detect_all_faces_and_features(img, target_size=target_size)
