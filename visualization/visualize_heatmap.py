import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import argparse
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description='Visualize token-level predictions as heatmap')
parser.add_argument('--idx', type=int, default=9, help='Index of the sample to visualize (default: 1)')
parser.add_argument('--max_tokens', type=int, default=120, help='Maximum number of tokens to display (default: None, show all)')
parser.add_argument('--logits_file', type=str, default='alpha_0.8/model_epoch_29.pt_logits.jsonl', 
                    help='Path to the logits JSONL file (default: alpha_0.8/model_epoch_29.pt_logits.jsonl)')
args = parser.parse_args()

target_idx = args.idx
max_tokens = args.max_tokens
logits_file = args.logits_file

# Read the JSONL file and find the target idx
with open(logits_file, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data['idx'] == target_idx:
            pred = data['pred']
            gt = data['gt']
            response_type = data['type']
            break

# Store original length
original_length = len(pred)

# Truncate to max_tokens if specified
if max_tokens is not None:
    pred = pred[:max_tokens]
    gt = gt[:max_tokens]
    print(f"Truncating from {original_length} to {max_tokens} tokens")

# Find first malicious token position (in the truncated data)
first_mal_pos = gt.index(1) if 1 in gt else None

print(f"Target idx: {target_idx}")
print(f"Data type: {response_type}")
print(f"Number of tokens: {len(pred)}")
if first_mal_pos is not None:
    print(f"First malicious token at position: {first_mal_pos}")
else:
    print("No malicious token found in ground truth")

# Convert predictions to numpy array
pred_array = np.array(pred)

# Create figure with appropriate size
fig, ax = plt.subplots(figsize=(10, 3))

# Create custom colormap from green (0) to red (1)
colors = ['#00ff00', '#ffff00', '#ff0000']  # Green -> Yellow -> Red
n_bins = 256
cmap = mcolors.LinearSegmentedColormap.from_list('green_red', colors, N=n_bins)

# Reshape to make it a 2D array for visualization (height=1, width=num_tokens)
pred_2d = pred_array.reshape(1, -1)

# Create heatmap
im = ax.imshow(pred_2d, cmap=cmap, aspect='auto', vmin=0, vmax=1, interpolation='nearest')

# Mark the first malicious token position with a vertical line (only if exists)
if first_mal_pos is not None:
    ax.axvline(x=first_mal_pos, color='blue', linewidth=2, linestyle='--', label=f'First malicious token (pos={first_mal_pos})')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1)
cbar.set_label('Prediction Probability (0=benign, 1=harmful)', fontsize=12)

# Set labels and title
ax.set_xlabel('Token Position', fontsize=12)
ax.set_ylabel('')
ax.set_yticks([])
title_suffix = f' (truncated to {max_tokens})' if max_tokens is not None else ''
ax.set_title(f'Token-level Prediction Heatmap (idx={target_idx}, type={response_type}){title_suffix}', fontsize=14, fontweight='bold')

# Add legend (only if there's a malicious token)
if first_mal_pos is not None:
    ax.legend(loc='upper left', fontsize=10)

# Add grid for x-axis
ax.set_xticks(np.arange(0, len(pred), 50))
ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)

# Generate output filename with idx and max_tokens
output_dir = os.path.dirname(logits_file)
base_name = os.path.basename(logits_file).replace('_logits.jsonl', '')
tokens_suffix = f'_maxT{max_tokens}' if max_tokens is not None else ''
output_file = os.path.join(output_dir, f'{base_name}_idx{target_idx}{tokens_suffix}_heatmap.png')

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nHeatmap saved to: {output_file}")

plt.show()
