import pickle
import sqlite3
import numpy as np
from pathlib import Path

# Resolve database path relative to project storage
DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "face_data.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def init_db(db_path=None):
    """Initializes the SQLite database table for face landmarks and names."""
    if db_path is None:
        db_path = DB_PATH
    db = sqlite3.connect(str(db_path))
    cursor = db.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            face_landmarks BLOB,
            label TEXT,
            name TEXT,
            cluster_name TEXT
        )
    """)
    db.commit()
    db.close()


def find_face_data(query_embedding, distance_threshold=0.15, db_path=None):
    """Searches SQLite database for a matching face landmark embedding using cosine distance.

    Args:
        query_embedding: 1D numpy array representing the normalized face landmark feature vector.
        distance_threshold: Maximum cosine distance threshold to consider a match.
        db_path: Path to sqlite database.

    Returns:
        tuple: (matched_name, matched_cluster, min_distance, is_match)
    """
    init_db(db_path)
    if db_path is None:
        db_path = DB_PATH

    query_vec = np.array(query_embedding, dtype=np.float32).flatten()
    query_norm = np.linalg.norm(query_vec)
    if query_norm > 0:
        query_vec = query_vec / query_norm

    db = sqlite3.connect(str(db_path))
    cursor = db.cursor()
    cursor.execute("SELECT face_landmarks, label, name, cluster_name FROM face_data")
    rows = cursor.fetchall()
    db.close()

    if not rows:
        return None, None, 1.0, False

    best_match_name = None
    best_match_cluster = None
    min_dist = float('inf')

    for row in rows:
        try:
            stored_landmarks = pickle.loads(row[0])
            stored_vec = np.array(stored_landmarks, dtype=np.float32).flatten()
            stored_norm = np.linalg.norm(stored_vec)
            if stored_norm > 0:
                stored_vec = stored_vec / stored_norm

            # Cosine distance: 1 - cosine_similarity
            cosine_sim = np.dot(query_vec, stored_vec)
            dist = 1.0 - float(cosine_sim)

            if dist < min_dist:
                min_dist = dist
                best_match_name = row[2]
                best_match_cluster = row[3] if row[3] else row[2]
        except Exception:
            continue

    if min_dist <= distance_threshold and best_match_name:
        return best_match_name, best_match_cluster, min_dist, True
    else:
        return best_match_name, best_match_cluster, min_dist, False


def save_face_data(face_landmarks, label, name, cluster_name=None, db_path=None):
    """Saves serialized face landmark embedding, label, name, and cluster into the SQLite database."""
    init_db(db_path)
    if db_path is None:
        db_path = DB_PATH
    if cluster_name is None:
        cluster_name = name

    landmarks_np = np.array(face_landmarks, dtype=np.float32)
    blob_data = pickle.dumps(landmarks_np)

    db = sqlite3.connect(str(db_path))
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO face_data(face_landmarks, label, name, cluster_name) VALUES (?, ?, ?, ?)",
        (blob_data, str(label), str(name), str(cluster_name))
    )
    db.commit()
    db.close()
    print(f"[+] Saved face profile for '{name}' (Cluster: '{cluster_name}') into database.")