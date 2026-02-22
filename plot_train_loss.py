"""
读取 train_log.txt，绘制训练 Loss 和 Acc(token) 曲线
"""
import re
from collections import defaultdict

import matplotlib.pyplot as plt

def parse_train_log(file_path):
    """解析 train_log.txt，按 epoch 聚合，返回 epochs, losses, accs（每 epoch 一个点）"""
    epoch_losses = defaultdict(list)
    epoch_accs = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "UpdateStep" not in line or "Loss:" not in line:
                continue
            # Epoch [1/30], UpdateStep [1/90], LR: 1.25e-05, Loss: 0.0232, Acc(token): 0.4237, α: 0.6
            m_epoch = re.search(r"Epoch\s*\[\s*(\d+)/\d+\]", line)
            m_loss = re.search(r"Loss:\s*([\d.]+)", line)
            m_acc = re.search(r"Acc\(token\):\s*([\d.]+)", line)
            if m_epoch and m_loss and m_acc:
                ep = int(m_epoch.group(1))
                epoch_losses[ep].append(float(m_loss.group(1)))
                epoch_accs[ep].append(float(m_acc.group(1)))
    epochs = sorted(epoch_losses.keys())
    losses = [sum(epoch_losses[ep]) / len(epoch_losses[ep]) for ep in epochs]
    accs = [sum(epoch_accs[ep]) / len(epoch_accs[ep]) for ep in epochs]
    return epochs, losses, accs


def main():
    log_path = "/home/ruqi/public/Kelp-ruqi-test/ckpts/seval_qwen3_8b_alpha_exp_2/alpha_0.6/train_log.txt"
    epochs, losses, accs = parse_train_log(log_path)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, losses, color="C0", linewidth=2.5, label="Loss")
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14, color="C0")
    ax1.tick_params(axis="x", labelsize=12)
    ax1.tick_params(axis="y", labelcolor="C0", labelsize=12)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, color="C1", linewidth=2.5, label="Acc (token)")
    ax2.set_ylabel("Acc (token)", fontsize=14, color="C1")
    ax2.tick_params(axis="y", labelcolor="C1", labelsize=12)

    plt.tight_layout()
    out_path = "/home/ruqi/public/Kelp-ruqi-test/train_loss_acc.png"
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"图片已保存为 {out_path}")


if __name__ == "__main__":
    main()
