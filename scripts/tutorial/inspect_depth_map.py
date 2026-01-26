"""Quick script to inspect a depth map .npy file."""

import numpy as np
import sys
from pathlib import Path

# Load the depth map
depth_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/interim/depth_maps/1719472538890313_depth.npy")

if not depth_path.exists():
    print(f"File not found: {depth_path}")
    sys.exit(1)

depth_map = np.load(depth_path)

print(f"Depth map shape: {depth_map.shape}")
print(f"Depth map dtype: {depth_map.dtype}")
print(f"Depth range: {depth_map.min():.2f} to {depth_map.max():.2f}")
print(f"Mean depth: {depth_map.mean():.2f}")
print(f"Median depth: {np.median(depth_map):.2f}")

# Optional: Visualize with matplotlib if available
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Raw depth map
    im1 = axes[0].imshow(depth_map, cmap='viridis')
    axes[0].set_title(f"Depth Map ({depth_map.shape})")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], label='Depth')
    
    # Histogram
    axes[1].hist(depth_map.flatten(), bins=50)
    axes[1].set_title("Depth Distribution")
    axes[1].set_xlabel("Depth Value")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    output_path = depth_path.parent / f"{depth_path.stem}_viz.png"
    plt.savefig(output_path, dpi=100)
    print(f"\nVisualization saved to: {output_path}")
    plt.show()
except ImportError:
    print("\nNote: Install matplotlib for visualization: uv add matplotlib")
