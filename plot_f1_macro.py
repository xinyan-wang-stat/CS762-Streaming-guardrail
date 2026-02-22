"""
绘制 streaming-level 和 response-level 的 4 个指标随 alpha 变化曲线
"""
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df_streaming = pd.read_csv("results_streaming_level.csv")
df_response = pd.read_csv("results_response_level.csv")

alpha = df_streaming["alpha"].values

# 4 个指标
metrics = ["accuracy", "precision-macro", "recall-macro", "f1-macro"]
metric_labels = ["Accuracy", "Precision-macro", "Recall-macro", "F1-macro"]

# 2x2 子图，共享 x 轴，扁一点
fig, axes = plt.subplots(2, 2, figsize=(12, 5), sharex=True)

for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 2, idx % 2]
    ax.plot(
        alpha, df_streaming[metric].values,
        marker="o", label="Streaming-level", linewidth=2, markersize=6
    )
    ax.plot(
        alpha, df_response[metric].values,
        marker="s", label="Response-level", linewidth=2, markersize=6
    )
    ax.set_ylabel(label, fontsize=11)
    if idx == 0:
        ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(alpha)
    # 左上角：abcd + 原 subtitle
    ax.text(0.02, 0.98, f"({chr(97 + idx)}) {label}", transform=ax.transAxes,
            fontsize=10, va="top", ha="left")

axes[1, 0].set_xlabel("Alpha", fontsize=11)
axes[1, 1].set_xlabel("Alpha", fontsize=11)

plt.tight_layout()
plt.savefig("f1_macro_vs_alpha.png", dpi=150)
plt.show()
print("图片已保存为 f1_macro_vs_alpha.png")
