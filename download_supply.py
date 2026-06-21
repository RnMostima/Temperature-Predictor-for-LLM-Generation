import os
import pandas as pd
from datasets import load_dataset

# 保存路径
SAVE_DIR = r""

os.environ["HF_ENDPOINT"] = ""
os.makedirs(SAVE_DIR, exist_ok=True)

def sample_dataset(ds, n, seed=42):
    """从数据集中随机抽取 n 条（若不足则全取）"""
    return ds.shuffle(seed=seed).select(range(min(len(ds), n)))

# 1. COIG-CQIA 子集：zhihu（50条）
print("正在下载 COIG-CQIA (zhihu)...")
try:
    ds_zhihu = load_dataset("m-a-p/COIG-CQIA", "zhihu", split="train")
    df_zhihu = pd.DataFrame([
        {
            "question": x.get("instruction", ""),
            "ref": x.get("output", ""),
            "dataset": "coig_zhihu"
        }
        for x in sample_dataset(ds_zhihu, 50)
    ])
    df_zhihu = df_zhihu[(df_zhihu["question"].astype(str).str.strip() != "") & 
                        (df_zhihu["ref"].astype(str).str.strip() != "")]
    print(f"  zhihu: {len(df_zhihu)} 条")
except Exception as e:
    print(f"  zhihu 下载失败: {e}")
    df_zhihu = pd.DataFrame(columns=["question", "ref", "dataset"])

# 2. COIG-CQIA 子集：douban（50条）
print("正在下载 COIG-CQIA (douban)...")
try:
    ds_douban = load_dataset("m-a-p/COIG-CQIA", "douban", split="train")
    df_douban = pd.DataFrame([
        {
            "question": x.get("instruction", ""),
            "ref": x.get("output", ""),
            "dataset": "coig_douban"
        }
        for x in sample_dataset(ds_douban, 50)
    ])
    df_douban = df_douban[(df_douban["question"].astype(str).str.strip() != "") & 
                          (df_douban["ref"].astype(str).str.strip() != "")]
    print(f"  douban: {len(df_douban)} 条")
except Exception as e:
    print(f"  douban 下载失败: {e}")
    df_douban = pd.DataFrame(columns=["question", "ref", "dataset"])

# 3. firefly-train-1.1M（80条）
print("正在下载 firefly-train-1.1M...")
try:
    ds_firefly = load_dataset("YeungNLP/firefly-train-1.1M", split="train")
    df_firefly = pd.DataFrame([
        {
            "question": x.get("input", ""),
            "ref": x.get("target", ""),
            "dataset": "firefly"
        }
        for x in sample_dataset(ds_firefly, 80)
    ])
    df_firefly = df_firefly[(df_firefly["question"].astype(str).str.strip() != "") & 
                            (df_firefly["ref"].astype(str).str.strip() != "")]
    print(f"  firefly: {len(df_firefly)} 条")
except Exception as e:
    print(f"  firefly 下载失败: {e}")
    df_firefly = pd.DataFrame(columns=["question", "ref", "dataset"])

# 4. BELLE（1000条）
print("正在下载 BELLE...")
try:
    ds_belle = load_dataset("BelleGroup/train_1M_CN", split="train")
    df_belle = pd.DataFrame([
        {
            "question": x.get("instruction", ""),
            "ref": x.get("response", ""),
            "dataset": "belle"
        }
        for x in sample_dataset(ds_belle, 1000)
    ])
    df_belle = df_belle[(df_belle["question"].astype(str).str.strip() != "") & 
                        (df_belle["ref"].astype(str).str.strip() != "")]
    print(f"  belle: {len(df_belle)} 条")
except Exception as e:
    print(f"  belle 下载失败: {e}")
    df_belle = pd.DataFrame(columns=["question", "ref", "dataset"])

# 合并并保存
frames = [df_zhihu, df_douban, df_firefly, df_belle]
valid_frames = [df for df in frames if not df.empty]

if valid_frames:
    all_data = pd.concat(valid_frames, ignore_index=True)
    output_csv = os.path.join(SAVE_DIR, "chinese_mixed.csv")
    all_data.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n合并完成！总样本数：{len(all_data)}")
    print(f"保存至：{output_csv}")
    print("\n各数据集分布：")
    print(all_data["dataset"].value_counts())
else:
    print("没有成功下载任何数据集")