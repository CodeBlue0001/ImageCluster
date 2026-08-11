import os
import json
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
PRIMARY_STORAGE = BASE_DIR / "storage" / "primary"
PROCESSED_STORAGE = BASE_DIR / "storage" / "processed"
CLUSTERS_STORAGE = PROCESSED_STORAGE / "clusters"
TRACK_FILE = BASE_DIR / "storage" / "delivered_images.json"

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')


def record_delivered(image_name):
    """Keep record of the delivered image in a file."""
    delivered = get_delivered_records()
    if image_name not in delivered:
        delivered.append(image_name)
        with open(TRACK_FILE, "w", encoding="utf-8") as f:
            json.dump(delivered, f, indent=2)


def get_delivered_records():
    """Read and return list of already delivered image names."""
    if TRACK_FILE.exists():
        with open(TRACK_FILE, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []


def get_image():
    """Returns one by one un-delivered image path from primary storage when called."""
    if not PRIMARY_STORAGE.exists():
        return None

    # Get all images from primary storage
    all_images = [
        f.name for f in PRIMARY_STORAGE.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Get delivered history
    delivered_images = get_delivered_records()

    # Find the next image that hasn't been delivered yet
    for img_name in all_images:
        if img_name not in delivered_images:
            record_delivered(img_name)
            return str(PRIMARY_STORAGE / img_name)

    return None  # All images delivered
