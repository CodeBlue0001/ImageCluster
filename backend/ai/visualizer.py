import numpy as np
import matplotlib.pyplot as plt

try:
    from ai.file_config import PROCESSED_STORAGE
except ImportError:
    from file_config import PROCESSED_STORAGE


def draw_cluster_map_with_folders(coords_2d, labels, metadata, num_clusters, save_path=None):
    """Plots 2D scatter graph of facial features with enclosing folder circles around each cluster."""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)

    unique_labels = set(labels)
    cmap = plt.get_cmap('tab10' if num_clusters <= 10 else 'rainbow')
    colors = {lbl: cmap(idx / max(1, len(unique_labels) - 1)) for idx, lbl in enumerate(sorted(unique_labels))}

    # 1. Draw cluster folder circles around each cluster's data points
    for lbl in sorted(unique_labels):
        if lbl == -1:
            continue  # Skip drawing folder circle for unclustered noise points

        pts = coords_2d[labels == lbl]
        if len(pts) == 0:
            continue

        center = np.mean(pts, axis=0)
        distances = np.linalg.norm(pts - center, axis=1)
        max_dist = np.max(distances) if len(distances) > 0 else 0.1
        radius = max(max_dist * 1.35, 0.25)

        cluster_color = colors[lbl]

        # Draw translucent folder circle patch
        circle = plt.Circle(
            center,
            radius,
            color=cluster_color,
            alpha=0.18,
            linestyle='--',
            linewidth=2
        )
        ax.add_patch(circle)

        # Draw folder label box above circle
        cluster_name = f"📁 person_{lbl + 1}"
        ax.text(
            center[0],
            center[1] + radius + 0.05,
            cluster_name,
            fontsize=10,
            fontweight='bold',
            ha='center',
            va='bottom',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=cluster_color, lw=1.5, alpha=0.9)
        )

    # 2. Plot face data points inside clusters
    for idx, (name, face_num, _) in enumerate(metadata):
        lbl = labels[idx]
        pt_color = colors[lbl]
        ax.scatter(
            coords_2d[idx, 0],
            coords_2d[idx, 1],
            color=pt_color,
            s=120,
            edgecolors='black',
            linewidths=1,
            zorder=3
        )
        label_text = f"{name}"
        ax.annotate(
            label_text,
            (coords_2d[idx, 0], coords_2d[idx, 1]),
            fontsize=7,
            xytext=(4, 4),
            textcoords='offset points',
            zorder=4
        )

    plt.title(
        f"Final Facial Landmark Cluster Map with Cluster Folders\n"
        f"Total Faces: {len(metadata)} | Total Discovered Person Clusters: {num_clusters}",
        fontsize=13,
        fontweight='bold',
        pad=15
    )
    plt.xlabel("PCA Landmark Component 1", fontsize=10)
    plt.ylabel("PCA Landmark Component 2", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"\n[+] Folder Circle Cluster Map saved to: {save_path}")

    return fig


def draw_standard_cluster_plot(coords_2d, labels, metadata, num_clusters, save_path=None):
    """Plots standard 2D PCA scatter graph of facial clusters."""
    fig = plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        coords_2d[:, 0],
        coords_2d[:, 1],
        c=labels,
        cmap='tab10' if num_clusters <= 10 else 'rainbow',
        s=130,
        edgecolors='black',
        alpha=0.85
    )

    for idx, (name, face_num, _) in enumerate(metadata):
        label_text = f"{name}"
        plt.annotate(
            label_text,
            (coords_2d[idx, 0], coords_2d[idx, 1]),
            fontsize=8,
            xytext=(5, 5),
            textcoords='offset points'
        )

    plt.colorbar(scatter, label='Person / Cluster ID')
    plt.title(
        f"Facial Landmark Clustering Graph\n"
        f"Total Faces: {len(metadata)} | Discovered Person Clusters: {num_clusters}"
    )
    plt.xlabel("PCA Face Landmark Component 1")
    plt.ylabel("PCA Face Landmark Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    if save_path:
        PROCESSED_STORAGE.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[+] Standard Cluster Plot saved to: {save_path}")

    return fig


def create_multiple_cluster_maps(coords_2d, labels, metadata, num_clusters, output_dir=None):
    """Creates multiple distinct cluster map plots for each individual person cluster
    as well as a multi-panel overview grid figure.
    """
    if output_dir is None:
        output_dir = PROCESSED_STORAGE
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap('tab10' if num_clusters <= 10 else 'rainbow')
    colors = {lbl: cmap(idx / max(1, len(unique_labels) - 1)) for idx, lbl in enumerate(unique_labels)}

    saved_maps = []

    # 1. Create individual cluster map graph for each distinct person/cluster
    for lbl in unique_labels:
        person_name = f"person_{lbl + 1}" if lbl != -1 else "unclustered_noise"
        indices = [i for i, l in enumerate(labels) if l == lbl]
        if not indices:
            continue

        fig, ax = plt.subplots(figsize=(8, 6))
        pt_color = colors[lbl]

        # Draw background points
        ax.scatter(coords_2d[:, 0], coords_2d[:, 1], color='lightgrey', s=40, alpha=0.5, label='Other Persons')

        # Highlight current person points
        ax.scatter(coords_2d[indices, 0], coords_2d[indices, 1], color=pt_color, s=140, edgecolors='black', linewidths=1.5, zorder=3, label=f"📁 {person_name}")

        for idx in indices:
            name = metadata[idx][0]
            ax.annotate(name, (coords_2d[idx, 0], coords_2d[idx, 1]), fontsize=8, xytext=(4, 4), textcoords='offset points', zorder=4)

        ax.set_title(f"Cluster Map: {person_name} ({len(indices)} images)", fontsize=12, fontweight='bold')
        ax.set_xlabel("PCA Landmark Component 1")
        ax.set_ylabel("PCA Landmark Component 2")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')

        map_path = output_dir / f"{person_name}_cluster_map.png"
        plt.savefig(map_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        saved_maps.append(map_path)

    # 2. Create multi-panel overview grid figure showing all person clusters side-by-side
    valid_labels = [l for l in unique_labels if l != -1]
    n_plots = len(valid_labels)
    if n_plots > 0:
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

        for idx, lbl in enumerate(valid_labels):
            r, c = idx // cols, idx % cols
            ax = axes[r, c]
            person_name = f"person_{lbl + 1}"
            p_indices = [i for i, l in enumerate(labels) if l == lbl]

            ax.scatter(coords_2d[:, 0], coords_2d[:, 1], color='lightgrey', s=30, alpha=0.4)
            ax.scatter(coords_2d[p_indices, 0], coords_2d[p_indices, 1], color=colors[lbl], s=100, edgecolors='black')
            ax.set_title(f"{person_name} ({len(p_indices)} imgs)", fontsize=11, fontweight='bold')
            ax.grid(True, linestyle=':', alpha=0.5)

        # Hide empty subplots
        for idx in range(n_plots, rows * cols):
            r, c = idx // cols, idx % cols
            fig.delaxes(axes[r, c])

        fig.suptitle(f"Multi-Person Facial Landmark Cluster Overview Grid ({num_clusters} Persons)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        grid_path = output_dir / "multi_cluster_maps_grid.png"
        plt.savefig(grid_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        saved_maps.append(grid_path)

    print(f"\n[+] Created {len(saved_maps)} individual & multi-cluster map plots under: {output_dir}")
    return saved_maps
