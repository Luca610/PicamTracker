import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from zero_suppression_encoding import read_file_header, read_and_decode_frame
from collections import deque
from scipy.spatial import cKDTree
from matplotlib.colors import LogNorm

def plot_cluster_2d(x, y, charge, filename="cluster.png", binsize=1.0,
                    title=None, cmap='viridis', dpi=150):
    """
    Plot a 2D histogram of a single cluster (pixel charge map) and save to PNG.

    Parameters
    ----------
    x, y : array-like
        Pixel coordinates of hits in the cluster.
    charge : array-like
        Charge (amplitude) values corresponding to each pixel.
    filename : str
        Path of the output PNG file.
    binsize : float, optional
        Size of each histogram bin in coordinate units (default = 1.0).
        Set smaller to get finer resolution.
    title : str or None
        Optional title for the plot.
    cmap : str
        Matplotlib colormap name for the charge intensity (default 'viridis').
    dpi : int
        Output image resolution.
    """

    x = np.asarray(x)
    y = np.asarray(y)
    charge = np.asarray(charge)

    if len(x) == 0:
        raise ValueError("No hits in cluster — cannot plot an empty cluster.")

    # Determine histogram edges based on data range
    x_min, x_max = x.min() - 0.5*binsize, x.max() + 0.5*binsize
    y_min, y_max = y.min() - 0.5*binsize, y.max() + 0.5*binsize

    # Compute number of bins automatically
    nx_bins = int(np.ceil((x_max - x_min) / binsize))
    ny_bins = int(np.ceil((y_max - y_min) / binsize))

    # Compute 2D charge histogram (weighted by charge)
    H, xedges, yedges = np.histogram2d(
        x, y, bins=(nx_bins, ny_bins),
        range=[[x_min, x_max], [y_min, y_max]],
        weights=charge
    )

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        H.T, origin='lower', cmap=cmap,
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        aspect='equal'
    )

    # Colorbar and labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Charge (a.u.)", fontsize=10)

    ax.set_xlabel("Pixel X coordinate")
    ax.set_ylabel("Pixel Y coordinate")
    ax.grid(True)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Cluster charge map")

    # Tight layout and save
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    plt.savefig(filename, dpi=dpi)
    plt.close(fig)

    print(f"Cluster plot saved as: {filename}")

def reconstruct_clusters_kdtree(x, y, amplitude,
                                 radius=1.0,
                                 connectivity='8',
                                 min_charge=None,
                                 return_indices=False):
    """
    Reconstruct clusters using a KD-tree for fast neighbor queries.

    Parameters
    ----------
    x, y : array-like (N,)
        Hit coordinates (can be float or int).
    amplitude : array-like (N,)
        Hit amplitudes/charges (floats).
    radius : float, optional
        Euclidean radius used to determine if two hits are neighbors.
        Typical grid use:
            - For 8-connectivity on integer grid, use radius = sqrt(2) + eps (often 1.42...)
            - For 4-connectivity on integer grid, use radius = 1.0 + eps but filter by Manhattan distance.
        Default: 1.0
    connectivity : {'8', '4'} or None
        If '8' use Euclidean neighbor acceptance (all neighbors within `radius`).
        If '4' use Manhattan (L1) connectivity: neighbors must satisfy |dx|+|dy| <= 1 (or <= radius),
           implemented by post-filtering the KD-tree results.
        If None: accept all neighbors returned by KD-tree (Euclidean).
    return_indices : bool
        If True, each cluster dict includes 'hits' list of original indices.

    Returns
    -------
    clusters : list of dict
        Each dict has keys:
          'charge' : float (sum of amplitudes)
          'size'   : int   (number of hits)
          'centroid' : (x_centroid, y_centroid) amplitude-weighted
          'hits' (optional) : list of hit indices (present if return_indices=True)

    Notes
    -----
    - Uses an iterative BFS (deque) for cluster growth.
    - KD-tree query_ball_point returns indices of points within `radius` (Euclidean).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    amp = np.asarray(amplitude)
    assert x.shape == y.shape == amp.shape
    N = x.size

    if N == 0:
        return []

    points = np.column_stack((x, y))
    tree = cKDTree(points)

    visited = np.zeros(N, dtype=bool)
    clusters = []

    # For numerical safety add tiny eps when comparing equality
    eps = 1e-12

    for i in range(N):
        if visited[i]:
            continue

        # start BFS/DFS from seed i
        queue = deque([i])
        cluster_idx = []

        while queue:
            idx = queue.popleft()
            if visited[idx]:
                continue
            visited[idx] = True
            cluster_idx.append(idx)

            # KD-tree: fast retrieval of candidate neighbors within Euclidean radius
            nbrs = tree.query_ball_point(points[idx], r=radius + eps)

            # Post-filter neighbors depending on connectivity
            if connectivity == '4':
                # keep only L1 neighbors: |dx| + |dy| <= radius (commonly radius==1)
                keep = []
                px, py = points[idx]
                for j in nbrs:
                    if not visited[j]:
                        dx = abs(points[j,0] - px)
                        dy = abs(points[j,1] - py)
                        if dx + dy <= radius + eps:
                            keep.append(j)
                nbrs = keep
            else:
                # for '8' or None, use Euclidean results; only unvisited neighbors
                nbrs = [j for j in nbrs if not visited[j]]

            # add neighbors to queue for exploration
            for j in nbrs:
                if not visited[j]:
                    queue.append(j)

        # compute cluster properties
        cluster_idx = np.array(cluster_idx, dtype=int)
        cluster_amp = amp[cluster_idx]
        total_charge = float(cluster_amp.sum())
        size = int(cluster_idx.size)

        if total_charge > 0:
            cx = float(np.sum(points[cluster_idx, 0] * cluster_amp) / total_charge)
            cy = float(np.sum(points[cluster_idx, 1] * cluster_amp) / total_charge)
        else:
            # fallback: unweighted geometric centroid if total charge is zero
            cx = float(points[cluster_idx, 0].mean())
            cy = float(points[cluster_idx, 1].mean())

        cluster = {
            'charge': total_charge,
            'size': size,
            'centroid': (cx, cy)
        }
        if return_indices:
            cluster['hits'] = cluster_idx.tolist()
        clusters.append(cluster)

    return clusters

argparse = argparse.ArgumentParser(description="Analyse source data from picamera.")
argparse.add_argument("filename", type=str, default='input.raw', help="Path to the source data file.")
argparse.add_argument("--clustering_threshold", type=int, default=73, help="Minimum amplitude to consider a hit for clustering.")
argparse.add_argument("--plot_clusters", action="store_true")
args = argparse.parse_args()
file_path = args.filename
filename = os.path.basename(file_path)
clustering_threshold = args.clustering_threshold
hist_amp_tot, bin_edges_amp_tot = np.histogram([], bins=256, range=(-0.5, 1023.5))
hist_cluster_size_tot, bin_edges_cluster_size_tot = np.histogram([], bins=50, range=(0.5, 50.5))
hist_cluster_charge_tot, bin_edges_cluster_charge_tot = np.histogram([], bins=256, range=(-0.5, 2047.5))
bins_centroids = (444, 330)  # 50x50 grid
range_centroids = [[-0.5, 1331.5], [-0.5, 989.5]]
hist_centroids_tot, xedges_centroids_tot, yedges_centroids_tot = np.histogram2d([], [], bins=bins_centroids, range=range_centroids)
max_x = 0
max_y = 0
plotted_sizes = []
with open(file_path, "rb") as infile:
    # Read the file header
    try:
        frame_width, frame_height = read_file_header(infile)
    except ValueError as e:
        print(f"Error reading file header: {e}")
        exit(1)
    print(f"Frame dimensions: {frame_width} x {frame_height}")
    # Loop until EOF
    frame_index = 0
    while True:
        if frame_index % 100 == 0:
            print(f"Processing frame {frame_index}", end='\r')
        # Try to read one frame header
        try:
            timestamp, y, x, amplitudes = read_and_decode_frame(infile) #TODO understand why they are swapped
        except EOFError:
            print(f"End of file reached at frame {frame_index}.")
            break
         # Skip empty frames (no hits)
        if len(x) == 0:
            frame_index += 1
            continue
        # Update cumulative amplitude histogram
        max_x = max(max_x, x.max())
        max_y = max(max_y, y.max())
        hist_amp_tot += np.histogram(amplitudes, bins=256, range=(-0.5, 1023.5))[0]
        # Reconstruct clusters
        mask = amplitudes > clustering_threshold
        clusters = reconstruct_clusters_kdtree(x[mask], y[mask], amplitudes[mask], radius=1., connectivity='8', return_indices = True)
        cluster_charges = [c['charge'] for c in clusters]
        cluster_sizes = [c['size'] for c in clusters]
        centroids = [c['centroid'] for c in clusters]
        hits = [c['hits'] for c in clusters]
        # Optionally plot individual clusters
        if args.plot_clusters:
            cluster_size_max = max(cluster_sizes) if cluster_sizes else 0
            if cluster_size_max not in plotted_sizes and cluster_size_max > 1:
                # Find the largest cluster
                largest_cluster_idx = np.argmax(cluster_sizes)
                hit_indices = hits[largest_cluster_idx]
                cluster_x = x[mask][hit_indices]
                cluster_y = y[mask][hit_indices]
                cluster_amplitudes = amplitudes[mask][hit_indices]
                plot_filename = f"clusters/frame_{frame_index:05d}_size_{cluster_size_max}.png"
                plot_title = f"Frame {frame_index}, Cluster size {cluster_size_max}"
                plot_cluster_2d(cluster_x, cluster_y, cluster_amplitudes,
                                filename=plot_filename,
                                binsize=1.0,
                                title=plot_title,
                                cmap='viridis',
                                dpi=150)
                plotted_sizes.append(cluster_size_max)
        # Update cumulative cluster histograms
        hist_cluster_charge_tot += np.histogram(cluster_charges, bins=256, range=(-0.5, 2047.5))[0]
        hist_cluster_size_tot += np.histogram(cluster_sizes, bins=50, range=(0.5, 50.5))[0]
        if centroids:
            centroids_x, centroids_y = zip(*centroids)
            hist_centroids_tot += np.histogram2d(centroids_x, centroids_y, bins=bins_centroids, range=range_centroids)[0]
        frame_index += 1


base_out = os.path.splitext(args.filename)[0]

# 2x2 figure: amplitude, cluster charge, cluster size, centroids heatmap
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax_amp, ax_charge, ax_size, ax_cent = axes.ravel()

# Amplitude histogram
amp_centers = (bin_edges_amp_tot[:-1] + bin_edges_amp_tot[1:]) / 2.0
amp_width = bin_edges_amp_tot[1] - bin_edges_amp_tot[0]
ax_amp.bar(amp_centers, hist_amp_tot, width=amp_width, align='center', edgecolor='k')
ax_amp.set_title('Sr(90) Amplitude Distribution')
ax_amp.set_xlabel('Charge (a.u.)')
ax_amp.set_ylabel('Entries')
ax_amp.set_yscale('log')
ax_amp.grid(True)

# Cluster charge histogram
charge_centers = (bin_edges_cluster_charge_tot[:-1] + bin_edges_cluster_charge_tot[1:]) / 2.0
charge_width = bin_edges_cluster_charge_tot[1] - bin_edges_cluster_charge_tot[0]
ax_charge.bar(charge_centers, hist_cluster_charge_tot, width=charge_width, align='center', edgecolor='k')
ax_charge.set_title('Total Cluster Charge Distribution')
ax_charge.set_xlabel('Cluster charge (a.u.)')
ax_charge.set_ylabel('Entries')
ax_charge.set_yscale('log')
ax_charge.grid(True)

# Cluster size histogram
size_centers = (bin_edges_cluster_size_tot[:-1] + bin_edges_cluster_size_tot[1:]) / 2.0
size_width = bin_edges_cluster_size_tot[1] - bin_edges_cluster_size_tot[0]
ax_size.bar(size_centers, hist_cluster_size_tot, width=size_width, align='center', edgecolor='k')
ax_size.set_title('Cluster Size Distribution')
ax_size.set_xlabel('Cluster size (pixels)')
ax_size.set_ylabel('Entries')
ax_size.set_yscale('log')
ax_size.grid(True)

# Centroids 2D histogram (heatmap)
# hist_centroids_tot is shaped (nx, ny) consistent with xedges_centroids_tot, yedges_centroids_tot
# Determine a sensible LogNorm vmin if any non-zero entries exist
centroid_vmin = None
if np.any(hist_centroids_tot > 0):
    centroid_vmin = float(hist_centroids_tot[hist_centroids_tot > 0].min())

extent = [xedges_centroids_tot[0], xedges_centroids_tot[-1], yedges_centroids_tot[0], yedges_centroids_tot[-1]]
if centroid_vmin is not None:
    im = ax_cent.imshow(hist_centroids_tot.T, origin='lower', extent=extent, aspect='auto', cmap='viridis',
                        norm=LogNorm(vmin=centroid_vmin, vmax=hist_centroids_tot.max()))
else:
    im = ax_cent.imshow(hist_centroids_tot.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
ax_cent.set_title('Cluster Centroid Density')
ax_cent.set_xlabel('Centroid X')
ax_cent.set_ylabel('Centroid Y')
cbar = fig.colorbar(im, ax=ax_cent)
cbar.set_label('Counts')

plt.tight_layout()
plt.savefig(f'{base_out}_summary.png')
plt.close(fig)

print(f"Analysis complete. Results saved to {base_out}_summary.png")
print(f"Maximum X coordinate observed: {max_x}, maximum Y coordinate observed: {max_y}")
