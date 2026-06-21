import requests
import json
import pandas as pd
import numpy as np
import re
import argparse
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# DeepSeek 生成配置
DEEPSEEK_CONFIG = {
    "name": "DeepSeek",
    "endpoint": "",
    "apiKey": "",
    "model": "",
    "timeout": 300
}

# 向量模型配置（本地）

# 使用本地 all-MiniLM-L6-v2 模型，确保稳定
LOCAL_EMBEDDING_MODEL_PATH = r""

# 温度搜索配置
COARSE_TEMPS = [0.05, 0.2, 0.5, 0.8, 1.1, 1.5, 2.0]   # 粗筛温度列表
SAMPLES_PER_TEMP = 4      
TOP_K = 3                 
FINE_SPAN = 0.25             
FINE_STEPS = 9                

# 本地数据集配置
LOCAL_DATASET_PATH = r""  # 你的输入数据
OUTPUT_CSV_PATH = "question_best_temp.csv"
SAVE_EVERY = 80
OUTPUT_COLUMNS = ["question", "dataset", "best_temperature"]

# 全局变量
embedding_model = None 

def load_embedding_model():
    """加载本地 sentence-transformers 模型（只加载一次）"""
    global embedding_model
    if embedding_model is None:
        print(f"加载本地向量模型: {LOCAL_EMBEDDING_MODEL_PATH}")
        embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_PATH)
    return embedding_model

def get_embedding(text: str):
    """将文本转为归一化向量（numpy array）"""
    model = load_embedding_model()
    # encode 返回 numpy array, shape=(1, dim)
    emb = model.encode([text], normalize_embeddings=True)  # 已归一化，余弦相似度就是点积
    return emb[0]

def compute_similarity(pred: str, ref: str) -> float:
    """计算生成答案与参考答案的余弦相似度（得分 0~1）"""
    if not pred or not ref:
        return 0.0
    emb_pred = get_embedding(pred)
    emb_ref = get_embedding(ref)
    # 由于已经归一化，余弦相似度 = 点积
    sim = np.dot(emb_pred, emb_ref)
    # 将 [-1,1] 映射到 [0,1] 并裁剪
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))

def ask_deepseek(prompt: str, temperature: float = 0.7) -> str:
    """调用 DeepSeek API 生成回答"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_CONFIG['apiKey']}"
    }
    payload = {
        "model": DEEPSEEK_CONFIG["model"],
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 1024,
        "stream": False
    }
    timeout = DEEPSEEK_CONFIG["timeout"] if DEEPSEEK_CONFIG["timeout"] > 0 else None
    response = requests.post(
        DEEPSEEK_CONFIG["endpoint"],
        headers=headers,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"].strip()

def load_local_data():
    """从本地文件读取问题和参考答案（与原逻辑相同）"""
    data = []
    if LOCAL_DATASET_PATH.endswith('.json'):
        with open(LOCAL_DATASET_PATH, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict) and 'data' in raw:
            items = raw['data']
        else:
            items = [raw]
        for item in items:
            question = item.get("question", "")
            ref = ""
            dataset_name = str(item.get("dataset", "unknown")).strip().lower()
            if "answers" in item and item["answers"]:
                answers = item["answers"]
                if isinstance(answers, dict) and answers.get("text"):
                    ref = answers["text"][0]
                elif isinstance(answers, list) and len(answers) > 0:
                    ref = answers[0]
            elif "answer" in item:
                ref = item["answer"]
            elif "ref" in item:
                ref = item["ref"]
            elif "label" in item:
                ref = str(item["label"])
            if question and ref:
                data.append({"question": question, "ref": ref, "dataset": dataset_name})
    elif LOCAL_DATASET_PATH.endswith('.csv'):
        df = pd.read_csv(LOCAL_DATASET_PATH)
        for _, row in df.iterrows():
            question = str(row.get("question", "")).strip()
            ref = str(row.get("answer", "") or row.get("ref", "")).strip()
            dataset_name = str(row.get("dataset", "unknown")).strip().lower()
            if question and ref:
                data.append({"question": question, "ref": ref, "dataset": dataset_name})
    return data

def get_best_temp(question: str, ref_answer: str) -> float:
    """
    两阶段搜索最优温度：粗筛 -> 细化
    返回最佳温度值
    """
    score_dict = {}
    
    # ---- 阶段1：粗筛 ----
    for t in COARSE_TEMPS:
        scores = []
        for _ in range(SAMPLES_PER_TEMP):
            pred = ask_deepseek(question, temperature=t)
            sim = compute_similarity(pred, ref_answer)
            scores.append(sim)
        score_dict[t] = np.mean(scores)
        print(f"    粗筛 temp={t:.2f} -> 平均相似度={score_dict[t]:.4f}")
    
    # 选出 top_k 温度
    sorted_coarse = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    top_temps = [t for t, _ in sorted_coarse[:TOP_K]]
    
    # ---- 阶段2：细化 ----
    low = max(0.01, min(top_temps) - FINE_SPAN)
    high = min(2.0, max(top_temps) + FINE_SPAN)
    fine_temps = list(np.linspace(low, high, FINE_STEPS))
    for t in fine_temps:
        t = round(t, 4)
        if t in score_dict:
            continue
        scores = []
        for _ in range(SAMPLES_PER_TEMP):
            pred = ask_deepseek(question, temperature=t)
            sim = compute_similarity(pred, ref_answer)
            scores.append(sim)
        score_dict[t] = np.mean(scores)
        print(f"    细化 temp={t:.4f} -> 平均相似度={score_dict[t]:.4f}")
    
    best_t = max(score_dict, key=score_dict.get)
    return best_t

def append_result_row(row, output_path=OUTPUT_CSV_PATH):
    """追加单条结果到 CSV（与原逻辑相同）"""
    normalized = {col: row.get(col) for col in OUTPUT_COLUMNS}
    normalized["question"] = str(normalized.get("question", "")).replace("\r", " ").replace("\n", " ").strip()
    normalized["dataset"] = str(normalized.get("dataset", "")).strip()
    df = pd.DataFrame([normalized], columns=OUTPUT_COLUMNS)
    write_header = not os.path.exists(output_path)
    df.to_csv(output_path, mode="a", index=False, header=write_header, encoding="utf-8-sig")

def load_existing_results(output_path=OUTPUT_CSV_PATH):
    """读取已保存结果，用于续跑（与原逻辑相同）"""
    if not os.path.exists(output_path):
        return []
    try:
        existing_df = pd.read_csv(output_path)
        for col in OUTPUT_COLUMNS:
            if col not in existing_df.columns:
                existing_df[col] = None
        existing_df = existing_df[OUTPUT_COLUMNS]
        existing_df = existing_df.dropna(subset=["question", "best_temperature"])
        existing_df["question"] = existing_df["question"].astype(str).str.replace("\r", " ", regex=False).str.replace("\n", " ", regex=False).str.strip()
        existing_df["dataset"] = existing_df["dataset"].fillna("").astype(str).str.strip()
        return existing_df.to_dict(orient="records")
    except Exception:
        return []

def save_dataset_rows(rows, output_path=OUTPUT_CSV_PATH):
    """完整保存所有结果（与原逻辑相同）"""
    if rows:
        df = pd.DataFrame(rows)
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[OUTPUT_COLUMNS]
    else:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于向量相似度自动寻找最优温度（DeepSeek生成）")
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=0,
        help="可选：从第 80*n 个样本开始处理；默认自动续跑"
    )
    args = parser.parse_args()

    # 加载数据集
    data = load_local_data()
    print(f"共加载有效样本：{len(data)} 条")

    # 续跑逻辑
    existing_rows = load_existing_results(OUTPUT_CSV_PATH)
    auto_start_index = len(existing_rows)
    manual_start_index = max(0, args.n) * SAVE_EVERY
    start_index = max(auto_start_index, manual_start_index)

    if start_index > 0:
        print(f"已检测到已保存进度：{auto_start_index} 条")
        if start_index != auto_start_index:
            print(f"根据参数 n，实际从第 {start_index + 1} 条样本开始处理")
        else:
            print(f"将自动从第 {start_index + 1} 条样本继续处理")

    # 预加载向量模型（只一次）
    load_embedding_model()

    # 主循环
    for idx in range(start_index, len(data)):
        d = data[idx]
        print(f"\n处理第 {idx+1} 题: {d['question'][:80]}...")
        try:
            best_t = get_best_temp(d["question"], d["ref"])
            row = {
                "question": d["question"],
                "dataset": d.get("dataset", "unknown"),
                "best_temperature": best_t
            }
            append_result_row(row)
            existing_rows.append(row)
            print(f"  最佳温度 = {best_t:.4f}")

            if (idx + 1) % SAVE_EVERY == 0:
                print(f"已保存进度：{OUTPUT_CSV_PATH}（共 {len(existing_rows)} 条）")
        except Exception as e:
            print(f"第 {idx+1} 题失败，已跳过：{e}")

    # 最终保存
    dataset = load_existing_results(OUTPUT_CSV_PATH)
    save_dataset_rows(dataset)
    print(f"\n完成！数据集已保存：{OUTPUT_CSV_PATH}")
    print(pd.DataFrame(dataset))