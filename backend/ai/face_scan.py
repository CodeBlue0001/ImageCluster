import pathlib
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from draw_result import draw_landmarks_on_image
except ImportError:
    try:
        from ai.draw_result import draw_landmarks_on_image
    except ImportError:
        from backend.ai.draw_result import draw_landmarks_on_image

try:
    from file_config import get_image
except ImportError:
    try:
        from ai.file_config import get_image
    except ImportError:
        from backend.ai.file_config import get_image

# Solve cv2_imshow import error when running outside Google Colab
try:
    from google.colab.patches import cv2_imshow
except ImportError:
    def cv2_imshow(img):
        """Fallback cv2_imshow for non-Colab (local) Python environments."""
        try:
            import matplotlib.pyplot as plt
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 and img.shape[2] == 3 else img
            plt.imshow(rgb)
            plt.axis('off')
            plt.show()
        except Exception:
            cv2.imshow('Image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

# Resolve model paths dynamically
MODEL_DIR = pathlib.Path(__file__).resolve().parent / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FACE_DETECTOR_MODEL_PATH = MODEL_DIR / "blaze_face_full_range.tflite"
if not FACE_DETECTOR_MODEL_PATH.exists():
    try:
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite"
        print("[+] Downloading blaze_face_full_range.tflite model...")
        urllib.request.urlretrieve(url, FACE_DETECTOR_MODEL_PATH)
        print("[+] Download complete.")
    except Exception as e:
        print(f"[!] Warning: Could not download blaze_face_full_range.tflite: {e}")
        short_range = MODEL_DIR / "blaze_face_short_range.tflite"
        if short_range.exists():
            FACE_DETECTOR_MODEL_PATH = short_range
        else:
            FACE_DETECTOR_MODEL_PATH = MODEL_DIR / "face_detector.task"

FACE_LANDMARKER_MODEL_PATH = MODEL_DIR / "face_landmarker.task"
if not FACE_LANDMARKER_MODEL_PATH.exists():
    try:
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        print("[+] Downloading face_landmarker.task model...")
        urllib.request.urlretrieve(url, FACE_LANDMARKER_MODEL_PATH)
        print("[+] Download complete.")
    except Exception as e:
        print(f"[!] Warning: Could not download face_landmarker.task: {e}")

MODEL_PATH = FACE_DETECTOR_MODEL_PATH

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def visualize(image, detection_result):
    """Draws bounding boxes on the input image and returns it."""
    if not detection_result or not detection_result.detections:
        return image
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        start_point = (bbox.origin_x, bbox.origin_y)
        end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

        if detection.categories:
            category = detection.categories[0]
            category_name = category.category_name if category.category_name else ''
            probability = round(category.score, 2)
            result_text = f'{category_name} ({probability})'.strip()
            text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
    return image


def prepare_data():
    # load the image getting new images one after another
    image_path = get_image()
    if not image_path:
        print("[!] No image available to process.")
        return None
    mp_image = mp.Image.create_from_file(image_path)
    return mp_image


def create_from_option():
    BaseOption = mp.tasks.BaseOptions
    FaceDetectorOption = mp.tasks.vision.FaceDetectorOptions
    VisionrunningMode = mp.tasks.vision.RunningMode

    # creating the face detector instance
    options = FaceDetectorOption(
        base_options=BaseOption(model_asset_path=str(FACE_DETECTOR_MODEL_PATH)),
        running_mode=VisionrunningMode.IMAGE,
        min_detection_confidence=0.3
    )
    detector = vision.FaceDetector.create_from_options(options)

    # calling the image from prepare data
    mp_image = prepare_data()
    if mp_image is None:
        return None, None

    face_detector_result = detector.detect(mp_image)

    # Fallback to short-range detector if no faces found
    if (not face_detector_result or not face_detector_result.detections) and (MODEL_DIR / "blaze_face_short_range.tflite").exists():
        short_options = FaceDetectorOption(
            base_options=BaseOption(model_asset_path=str(MODEL_DIR / "blaze_face_short_range.tflite")),
            running_mode=VisionrunningMode.IMAGE,
            min_detection_confidence=0.15
        )
        short_detector = vision.FaceDetector.create_from_options(short_options)
        face_detector_result = short_detector.detect(mp_image)

    print('face_detection result', face_detector_result)
    image_copy = np.copy(mp_image.numpy_view())
    annotated_image = visualize(image_copy, face_detector_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2_imshow(rgb_annotated_image)
    return mp_image, face_detector_result


# creating the face landmark for face identification
def face_landmark_identification(image_input=None):
    """Detects, displays, and extracts 3D face landmarks (478 keypoints per face) for all faces in an image.
    Returns:
        all_landmarks: list of lists of NormalizedLandmark objects (478 landmarks per detected face)
    """
    mp_image = None
    face_detector_result = None

    if image_input is None:
        mp_image = prepare_data()
    elif isinstance(image_input, tuple):
        mp_image = image_input[0]
        if len(image_input) > 1:
            face_detector_result = image_input[1]
    elif isinstance(image_input, list):
        mp_image = image_input[0] if image_input else None
    elif hasattr(image_input, '_image_ptr'):
        mp_image = image_input
    else:
        mp_image = prepare_data()

    if mp_image is None:
        print("[!] No valid mp.Image available for landmark identification.")
        return []

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    model_path_str = str(FACE_LANDMARKER_MODEL_PATH if FACE_LANDMARKER_MODEL_PATH.exists() else MODEL_PATH)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path_str),
        running_mode=VisionRunningMode.IMAGE
    )

    landmarker = FaceLandmarker.create_from_options(options)

    # 1. Try detecting landmarks directly on full mp_image
    face_landmarker_result = landmarker.detect(mp_image)
    all_landmarks = []

    if face_landmarker_result and face_landmarker_result.face_landmarks:
        all_landmarks = list(face_landmarker_result.face_landmarks)
        print(f"[+] Direct FaceLandmarker found landmarks for {len(all_landmarks)} face(s).")
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), face_landmarker_result)
        cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # 2. If direct landmarker returned 0 landmarks (e.g. large DSLR image), crop faces first
    if not all_landmarks:
        if face_detector_result is None:
            detector_options = mp.tasks.vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(FACE_DETECTOR_MODEL_PATH)),
                running_mode=VisionRunningMode.IMAGE,
                min_detection_confidence=0.3
            )
            detector = vision.FaceDetector.create_from_options(detector_options)
            face_detector_result = detector.detect(mp_image)

        img_np = mp_image.numpy_view()
        h_img, w_img = img_np.shape[:2]

        if face_detector_result and face_detector_result.detections:
            print(f"[+] Displaying face landmarks for {len(face_detector_result.detections)} face(s)...")
            for idx, detection in enumerate(face_detector_result.detections):
                bbox = detection.bounding_box
                x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
                margin_w, margin_h = int(w * 0.1), int(h * 0.1)
                x_min, y_min = max(0, x - margin_w), max(0, y - margin_h)
                x_max, y_max = min(w_img, x + w + margin_w), min(h_img, y + h + margin_h)

                crop = img_np[y_min:y_max, x_min:x_max]
                if crop.size > 0:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if crop.shape[2] == 3 else crop
                    crop_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop_rgb))
                    lm_res = landmarker.detect(crop_mp)
                    if lm_res and lm_res.face_landmarks:
                        all_landmarks.append(lm_res.face_landmarks[0])
                        # Display landmark mesh overlay for this face
                        crop_annotated = draw_landmarks_on_image(crop_mp.numpy_view(), lm_res)
                        print(f"    - Displaying 478 face landmarks for Face #{idx + 1}")
                        cv2_imshow(cv2.cvtColor(crop_annotated, cv2.COLOR_RGB2BGR))

    print(f"[+] Total face landmark sets displayed: {len(all_landmarks)}")
    return all_landmarks


if __name__ == "__main__":
    face = create_from_option()
    landmarks = face_landmark_identification(face)
    print(f"Extracted landmark sets count: {len(landmarks)}")





