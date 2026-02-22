from datasets import load_dataset
import pandas as pd
import os

# ————————————
# 1. 加载数据集
# ————————————
# Hugging Face datasets 会自动从 hub 下载数据
dataset = load_dataset("Alibaba-AAIG/StreamGuardBench", "seval_qwen3_8b")
train = dataset["train"]

print(f"Train size: {len(train)}")

# ————————————
# 2. 转成 pandas 方便操作
# ————————————
df = train.to_pandas()

# 查看 label 分布
print("Label counts:\n", df["label"].value_counts())

# ————————————
# 3. 按 label 抽样
# ————————————
df0 = df[df["label"] == 0].sample(n=50, random_state=42)
df1 = df[df["label"] == 1].sample(n=50, random_state=42)

df_sample = pd.concat([df0, df1]).reset_index(drop=True)

print("Sampled dataset size:", len(df_sample))
print(df_sample["label"].value_counts())

# ————————————
# 4. 保存到本地
# ————————————
output_dir = "../data/"
os.makedirs(output_dir, exist_ok=True)

# 保存成 CSV
csv_path = os.path.join(output_dir, "seval_qwen3_8b_sample.csv")
df_sample.to_csv(csv_path, index=False)

# 保存成 JSON Lines（如果需要）
#jsonl_path = os.path.join(output_dir, "seval_qwen3_8b_sample.jsonl")
#df_sample.to_json(jsonl_path, orient="records", lines=True)

print("Saved sampled data to:")
print("  -", csv_path)
#print("  -", jsonl_path)
