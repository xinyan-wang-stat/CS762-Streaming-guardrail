"""
Bar chart: 50 harmful responses — Harmful tokens vs benign tokens proportions.
Max length 1024 tokens. Uses only rows with label==1 (harmful).
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MAX_LENGTH = 1024

# Read data
df = pd.read_csv("data/annotated_output.csv")
harmful = df[df["label"] == 1].head(50).copy()

def word_count(text):
    return len(str(text).split())

# Cap at 1024 tokens (whitespace tokenization)
harmful["total_tokens"] = harmful["response"].apply(word_count).clip(upper=MAX_LENGTH)
harmful["benign_tokens"] = harmful["cut_index"].clip(lower=0)
# Benign cannot exceed total; harmful = total - benign
harmful["benign_tokens"] = harmful[["benign_tokens", "total_tokens"]].min(axis=1)
harmful["harmful_tokens"] = (harmful["total_tokens"] - harmful["benign_tokens"]).clip(lower=0)

# Stacked bar chart (tokens, not proportion)
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(harmful))
width = 0.72

ax.bar(x, harmful["benign_tokens"].values, width, label="Benign tokens", color="#2ecc71", edgecolor="none")
ax.bar(x, harmful["harmful_tokens"].values, width, bottom=harmful["benign_tokens"].values, label="Harmful tokens", color="#e74c3c", edgecolor="none")
# Line: harmful token appearance position (cut_index) per response
ax.plot(x, harmful["cut_index"].values, "o-", color="#3498db", linewidth=1.5, markersize=4, label="Harmful token position (cut_index)")

ax.set_ylabel("Tokens", fontsize=11)
ax.set_xlabel("Harmful response index", fontsize=11)
ax.set_title("50 harmful responses: Harmful vs benign tokens (max 1024 tokens)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([str(i + 1) for i in x], fontsize=8)
ax.set_ylim(0, 40)
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(axis="y", alpha=0.3, linestyle="--")
fig.tight_layout()
plt.savefig("data/harmful_responses_token_proportions.png", dpi=150, bbox_inches="tight")
print("Saved: data/harmful_responses_token_proportions.png")
plt.close()
