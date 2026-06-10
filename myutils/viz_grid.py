import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# Base directory
# base_dir = Path("/home/lukovsky/repos/ACL-SSL/train_outputs_viz_final/acl_intro_san")
# nested_path = "Visual_results_test/vggss/ACL_ViT16_Exp_ACL_v1/epochbest/overlaid"
base_dir = Path("/home/lukovsky/repos/ACL-SSL/train_outputs_viz_final/acl_san_v5_intro")
nested_path = "Visual_results_test/vggss/ACL_ViT16_aclifa_2gpu/epoch16/overlaid"

# Audio types in vertical order
audio_types = ["original", "silence", "noise"]

# Get sample names from the first directory
sample_dir = base_dir / audio_types[0] / nested_path
samples = sorted([f.stem for f in sample_dir.glob("*.jpg")])

# Create figure with equal aspect ratio for spacing
num_rows = len(audio_types)
num_cols = len(samples)
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6), squeeze=False)

# Add title
# fig.suptitle("Audio Type Analysis - Overlaid Results", fontsize=16, fontweight='bold', y=0.98)

# Load and display images
for row_idx, audio_type in enumerate(audio_types):
    for col_idx, sample in enumerate(samples):
        img_path = base_dir / audio_type / nested_path / f"{sample}.jpg"

        ax = axes[row_idx, col_idx]

        if img_path.exists():
            img = mpimg.imread(img_path)
            ax.imshow(img, aspect='equal')
        else:
            ax.text(0.5, 0.5, "Image not found", ha='center', va='center', transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if row_idx == 0:
            # ax.set_xlabel(f'{sample[:6]} #{col_idx+1}', fontsize=12, fontweight='bold')
            ax.set_title(f"{sample[:7]} #{col_idx+1}", fontsize=8, fontweight='bold', color="#222222", pad=4)

        # Set label only for first column
        if col_idx == 0:
            ax.set_ylabel(audio_type.capitalize(), fontsize=12, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

# Adjust spacing to be uniform
plt.subplots_adjust(left=0.1, right=0.95, top=0.93, bottom=0.05, hspace=0.05, wspace=0.05)

plt.savefig(base_dir / "overlaid_grid.png", dpi=300, bbox_inches='tight')
# plt.show()

print("Grid saved to:", base_dir / "overlaid_grid.png")