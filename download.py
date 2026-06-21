import os
import pandas as pd
from datasets import load_dataset

SAVE_DIR = r""  # 你想保存的路径

os.environ["HF_ENDPOINT"] = ""

os.makedirs(SAVE_DIR, exist_ok=True)


def sample_count(ds, n):
	return min(len(ds), n)

# 1. 下载 BELLE
print("正在下载 BELLE...")
belle = load_dataset("BelleGroup/train_1M_CN", split="train", cache_dir=os.path.join(SAVE_DIR, "belle"))
belle_sample = belle.shuffle(seed=42).select(range(sample_count(belle, 210)))
belle_df = pd.DataFrame([
	{
		"question": x.get("instruction", x.get("question", "")),
		"ref": x.get("response", x.get("output", x.get("answer", x.get("target", "")))),
		"dataset": "belle"
	}
	for x in belle_sample
])

# 2. 下载 GSM8K
print("正在下载 GSM8K...")
gsm8k_df = pd.DataFrame(columns=["question", "ref", "dataset"])
for gsm8k_config in ["main", "default", "socratic"]:
	try:
		gsm8k = load_dataset("gsm8k", gsm8k_config, split="train", cache_dir=os.path.join(SAVE_DIR, "gsm8k"))
		gsm8k_sample = gsm8k.shuffle(seed=42).select(range(sample_count(gsm8k, 140)))
		gsm8k_df = pd.DataFrame([
			{
				"question": x.get("question", ""),
				"ref": x.get("answer", x.get("final_answer", "")),
				"dataset": "gsm8k"
			}
			for x in gsm8k_sample
		])
		print(f"GSM8K 使用配置：{gsm8k_config}")
		break
	except Exception as e:
		print(f"尝试 GSM8K 配置 {gsm8k_config} 失败：{e}")

if gsm8k_df.empty:
	print("GSM8K 下载失败，本次先只合并 BELLE（及可用的 WritingPrompts）")

# 3. 下载 WritingPrompts
print("正在下载 writingprompts...")
wp_df = pd.DataFrame(columns=["question", "ref", "dataset"])
for wp_name in ["euclaise/writingprompts", "writingprompts"]:
	try:
		wp = load_dataset(wp_name, split="train", cache_dir=os.path.join(SAVE_DIR, "writingprompts"))
		wp_sample = wp.shuffle(seed=42).select(range(sample_count(wp, 112)))
		wp_df = pd.DataFrame([
			{
				"question": x.get("prompt", x.get("question", "")),
				"ref": x.get("response", x.get("story", x.get("answer", ""))),
				"dataset": "writingprompts"
			}
			for x in wp_sample
		])
		print(f"writingprompts 使用数据集：{wp_name}")
		break
	except Exception as e:
		print(f"尝试 {wp_name} 失败：{e}")

if wp_df.empty:
	print("writingprompts 下载失败，本次先只合并 BELLE + GSM8K")

final_csv_path = os.path.join(SAVE_DIR, "merged_dataset.csv")
belle_df = belle_df[(belle_df["question"].astype(str).str.strip() != "") & (belle_df["ref"].astype(str).str.strip() != "")]
gsm8k_df = gsm8k_df[(gsm8k_df["question"].astype(str).str.strip() != "") & (gsm8k_df["ref"].astype(str).str.strip() != "")]
wp_df = wp_df[(wp_df["question"].astype(str).str.strip() != "") & (wp_df["ref"].astype(str).str.strip() != "")]

frames = [belle_df]
if not gsm8k_df.empty:
	frames.append(gsm8k_df)
if not wp_df.empty:
	frames.append(wp_df)

all_data = pd.concat(frames, ignore_index=True)
all_data.to_csv(final_csv_path, index=False, encoding="utf-8-sig")

print("全部下载完成！")
print(f"数据集保存路径：{SAVE_DIR}")
print(f"最终合并CSV：{final_csv_path}")
print(f"总样本数：{len(all_data)}")