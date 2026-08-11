import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure backend directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from ai.file_config import get_image, PROCESSED_STORAGE
    from ai.face_detector import detect_faces_with_central_priority
    from ai.organizer import save_clustering_results, organize_images_by_cluster
    from ai.visualizer import draw_cluster_map_with_folders, draw_standard_cluster_plot, create_multiple_cluster_maps
except ImportError:
    try:
        from backend.ai.file_config import get_image, PROCESSED_STORAGE
        from backend.ai.face_detector import detect_faces_with_central_priority
        from backend.ai.organizer import save_clustering_results, organize_images_by_cluster
        from backend.ai.visualizer import draw_cluster_map_with_folders, draw_standard_cluster_plot, create_multiple_cluster_maps
    except ImportError:
        from file_config import get_image, PROCESSED_STORAGE
        from face_detector import detect_faces_with_central_priority
        from organizer import save_clustering_results, organize_images_by_cluster
        from visualizer import draw_cluster_map_with_folders, draw_standard_cluster_plot, create_multiple_cluster_maps


def face_clustering():
    """Main pipeline for reading images, extracting 3D facial landmark features for the central person,
    performing DBSCAN clustering, saving JSON records, organizing files into person/group subfolders,
    and rendering multiple cluster maps.
    """
    i = 1
    all_face_features = []
    metadata = []  # Stores (image_name, face_number, img_path)
    group_images = set()  # Set of image paths containing multiple faces (group photos)

    print("=== Starting 3D Landmark Facial Clustering Pipeline ===")

    while True:
        # Fetch next un-processed image path from primary storage
        img_path = get_image()
        if img_path is None:
            print("\n[+] Processing complete: No more un-processed images in storage/primary.")
            break

        img_name = Path(img_path).name
        print(f"[{i}] Reading image: {img_name}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"    Skipping unreadable image file: {img_path}")
            continue

        # Detect faces and extract 3D landmark features for central main person
        num_faces, central_feat, all_feats = detect_faces_with_central_priority(img)
        print(f"    Detected {num_faces} face(s) in {img_name}")

        if num_faces == 0 or central_feat is None:
            print(f"    No faces detected in {img_name}")
            continue

        # Mark multi-face images (>= 2 faces) for group folder
        if num_faces >= 2:
            group_images.add(img_path)
            print(f"    -> Flagged '{img_name}' for 'group' folder (contains {num_faces} faces, prioritizing central person)")

        # Use 3D landmark feature vector of central person for clustering
        all_face_features.append(central_feat)
        metadata.append((img_name, 1, img_path))

        i += 1

    if all_face_features:
        print(f"\n[+] Running DBSCAN clustering across {len(all_face_features)} 3D landmark feature vectors...")
        X = np.array(all_face_features)
        scaled_X = StandardScaler().fit_transform(X)

        # DBSCAN clustering using cosine metric on facial landmark embeddings
        dbscan = DBSCAN(eps=0.05, min_samples=1, metric='cosine')
        labels = dbscan.fit_predict(X)

        unique_labels = set(labels)
        num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        print(f"    - Discovered {num_clusters} distinct person cluster(s)")
        print(f"    - Found {len(group_images)} group image(s) with multiple faces")

        # Dimensionality reduction to 2D PCA for graph visualization
        if len(X) > 1:
            pca = PCA(n_components=2)
            coords_2d = pca.fit_transform(scaled_X)
        else:
            coords_2d = np.zeros((1, 2))

        # 1. Keep record of clustering in JSON
        json_file = save_clustering_results(metadata, labels, group_images=group_images)

        # 2. Organize images into person subfolders and group folder
        organize_images_by_cluster(metadata, labels, group_images=group_images)

        # 3. Draw & save standard scatter plot
        draw_standard_cluster_plot(
            coords_2d, labels, metadata, num_clusters,
            save_path=PROCESSED_STORAGE / "clustering_plot.png"
        )

        # 4. Draw & save cluster map with folder circles
        fig = draw_cluster_map_with_folders(
            coords_2d, labels, metadata, num_clusters,
            save_path=PROCESSED_STORAGE / "cluster_map_folders.png"
        )

        # 5. Create multiple individual cluster maps and multi-cluster grid figure
        multiple_maps = create_multiple_cluster_maps(
            coords_2d, labels, metadata, num_clusters,
            output_dir=PROCESSED_STORAGE / "cluster_maps"
        )

        # 6. Display the cluster result graph
        print("\n[+] Displaying facial landmark clustering result graph...")
        plt.show()

    else:
        print("\n[!] No facial landmark features were extracted from the provided images.")

    print("\n=== 3D Landmark Facial Clustering Pipeline Finished Successfully ===")


if __name__ == "__main__":
    face_clustering()