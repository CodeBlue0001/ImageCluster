import json
import shutil
from pathlib import Path

try:
    from ai.file_config import PROCESSED_STORAGE, CLUSTERS_STORAGE
except ImportError:
    from file_config import PROCESSED_STORAGE, CLUSTERS_STORAGE


def save_clustering_results(metadata, labels, group_images=None):
    """Saves the clustering mapping and statistics to a JSON file in storage/processed."""
    PROCESSED_STORAGE.mkdir(parents=True, exist_ok=True)
    if group_images is None:
        group_images = []

    cluster_map = {}
    for (img_name, face_num, img_path), label in zip(metadata, labels):
        folder_name = f"person_{label + 1}" if label != -1 else "unclustered_noise"
        if folder_name not in cluster_map:
            cluster_map[folder_name] = {
                "images": [],
                "faces": []
            }
        if img_name not in cluster_map[folder_name]["images"]:
            cluster_map[folder_name]["images"].append(img_name)
        cluster_map[folder_name]["faces"].append({
            "image_name": img_name,
            "face_number": face_num,
            "image_path": str(img_path)
        })

    unique_labels = set(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)

    result_data = {
        "total_faces_detected": len(metadata),
        "total_person_clusters_discovered": num_clusters,
        "total_group_images": len(group_images),
        "group_images": [str(p) for p in group_images],
        "person_clusters": cluster_map
    }

    json_path = PROCESSED_STORAGE / "clustering_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(f"\n[+] Clustering record saved to JSON: {json_path}")
    return json_path


def organize_images_by_cluster(metadata, labels, group_images=None):
    """Copies source images into separate person subdirectories and group directory."""
    CLUSTERS_STORAGE.mkdir(parents=True, exist_ok=True)
    if group_images is None:
        group_images = []

    # 1. Organize into individual person folders
    cluster_to_images = {}
    for (img_name, face_num, img_path), label in zip(metadata, labels):
        folder_name = f"person_{label + 1}" if label != -1 else "unclustered_noise"
        if folder_name not in cluster_to_images:
            cluster_to_images[folder_name] = set()
        cluster_to_images[folder_name].add(img_path)

    print(f"\n[+] Organizing images into person directories under: {CLUSTERS_STORAGE}")
    for folder_name, img_paths in cluster_to_images.items():
        cluster_dir = CLUSTERS_STORAGE / folder_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        for src_path in img_paths:
            dst_path = cluster_dir / Path(src_path).name
            shutil.copy2(src_path, dst_path)

        print(f"    - Person Folder '{folder_name}': copied {len(img_paths)} image(s)")

    # 2. Organize images containing multiple faces into dedicated 'group' folder
    if group_images:
        group_dir = CLUSTERS_STORAGE / "group"
        group_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[+] Organizing multiple face images into group directory: {group_dir}")
        for src_path in group_images:
            dst_path = group_dir / Path(src_path).name
            shutil.copy2(src_path, dst_path)
        print(f"    - Group Folder 'group': copied {len(group_images)} group image(s)")
