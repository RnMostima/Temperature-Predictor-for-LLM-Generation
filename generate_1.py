import requests
import json
import pandas as pd
import numpy as np
import re
import argparse
import os
from sentence_transformers import SentenceTransformer   # 新增

# 本地向量模型配置 
LOCAL_EMBEDDING_MODEL_PATH = r""  # 请确保路径正确

# 本地 ollama 配置 
OLLAMA_API = ""    
MODEL_NAME = ""    

# DeepSeek 评分 API 配置 
DEEPSEEK_CONFIG = {
    "endpoint": "",
    "apiKey": "",
    "model": "",
    "timeout": 300
}

# 评分融合配置 
RULE_SCORE_WEIGHT = 0.6
LLM_SCORE_WEIGHT = 0.4

# 数据集评分策略配置 
DEFAULT_DATASET_NAME = "gsm8k"
OBJECTIVE_DATASETS = {"gsm8k"}   # 客观题数据集：只用规则分（相似度），不调用 LLM

# 本地数据集配置 
LOCAL_DATASET_PATH = r""
OUTPUT_CSV_PATH = "question_best_temp.csv"
SAVE_EVERY = 80
OUTPUT_COLUMNS = ["question", "dataset", "best_temperature"]

# 全局向量模型 
embedding_model = None

def load_embedding_model():
    global embedding_model
    if embedding_model is None:
        print(f"加载本地向量模型: {LOCAL_EMBEDDING_MODEL_PATH}")
        embedding_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL_PATH)
    return embedding_model

def get_embedding(text: str) -> np.ndarray:
    model = load_embedding_model()
    # 归一化，便于直接计算余弦相似度（点积）
    emb = model.encode([text], normalize_embeddings=True)
    return emb[0]

def compute_score(pred: str, ref: str) -> float:
    """
    计算生成答案与参考答案的语义相似度（余弦相似度），映射到 0~1。
    替代原来的 rouge 评分。
    """
    if not pred or not ref:
        return 0.0
    emb_pred = get_embedding(pred)
    emb_ref = get_embedding(ref)
    # 余弦相似度（因已归一化，点积即为余弦值），范围 [-1,1]
    sim = np.dot(emb_pred, emb_ref)
    # 映射到 [0,1]
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))

def load_local_data():
    """从本地文件读取问题和标准答案。"""
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
            dataset_name = str(item.get("dataset", DEFAULT_DATASET_NAME)).strip().lower()
            
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
                data.append({
                    "question": question,
                    "ref": ref,
                    "dataset": dataset_name
                })
    
    elif LOCAL_DATASET_PATH.endswith('.csv'):
        df = pd.read_csv(LOCAL_DATASET_PATH)
        for _, row in df.iterrows():
            question = str(row.get("question", "")).strip()
            ref = str(row.get("answer", "") or row.get("ref", "")).strip()
            dataset_name = str(row.get("dataset", DEFAULT_DATASET_NAME)).strip().lower()
            if question and ref:
                data.append({
                    "question": question,
                    "ref": ref,
                    "dataset": dataset_name
                })
    
    return data

def ask_deepseek_judge(prompt):
    """调用 DeepSeek API 进行评审。"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_CONFIG['apiKey']}"
    }
    payload = {
        "model": DEEPSEEK_CONFIG["model"],
        "messages": [
            {"role": "system", "content": "你是严格的回答质量评测员，只输出0到1之间的小数分数。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    timeout = None if DEEPSEEK_CONFIG["timeout"] == 0 else DEEPSEEK_CONFIG["timeout"]
    response = requests.post(
        DEEPSEEK_CONFIG["endpoint"],
        headers=headers,
        json=payload,
        timeout=timeout
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]

def llm_judge_score(question, pred, ref):
    """让 LLM 作为裁判，输出 0~1 的分数。"""
    judge_prompt = f"""
你是严格评测员。请根据“问题、参考答案、模型回答”给模型回答打分。

评分标准：
- 1.0：语义与参考答案高度一致，关键事实正确
- 0.7：大体正确，但有轻微缺失/冗余
- 0.4：部分相关，但关键信息不完整或有偏差
- 0.0：明显错误或不相关

只输出一个 0 到 1 之间的小数，不要输出任何解释。

问题：{question}
参考答案：{ref}
模型回答：{pred}
""".strip()

    try:
        raw = ask_deepseek_judge(judge_prompt)
        match = re.search(r"(0(\.\d+)?|1(\.0+)?)", raw)
        if not match:
            return 0.0
        score = float(match.group(1))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0

def use_llm_judge(dataset_name):
    """根据数据集类型决定是否启用 LLM 评审。"""
    if not dataset_name:
        return True
    return dataset_name.strip().lower() not in OBJECTIVE_DATASETS

def compute_weighted_score(question, pred, ref, dataset_name=None):
    """按数据集策略计算分数：客观题只规则分（相似度），主观题做加权。"""
    rule_score = compute_score(pred, ref)

    if not use_llm_judge(dataset_name):
        return rule_score

    llm_score = llm_judge_score(question, pred, ref)
    final_score = RULE_SCORE_WEIGHT * rule_score + LLM_SCORE_WEIGHT * llm_score
    return final_score

def save_dataset_rows(rows, output_path=OUTPUT_CSV_PATH):
    if rows:
        df = pd.DataFrame(rows)
        for col in OUTPUT_COLUMNS:
            if col not in df.columns:
                df[col] = None
        df = df[OUTPUT_COLUMNS]
    else:
        df = pd.DataFrame(columns=OUTPUT_COLUMNS)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

def append_result_row(row, output_path=OUTPUT_CSV_PATH):
    normalized = {col: row.get(col) for col in OUTPUT_COLUMNS}
    normalized["question"] = str(normalized.get("question", "")).replace("\r", " ").replace("\n", " ").strip()
    normalized["dataset"] = str(normalized.get("dataset", "")).strip()
    df = pd.DataFrame([normalized], columns=OUTPUT_COLUMNS)
    write_header = not os.path.exists(output_path)
    df.to_csv(output_path, mode="a", index=False, header=write_header, encoding="utf-8-sig")

def load_existing_results(output_path=OUTPUT_CSV_PATH):
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

# 调用 ollama 生成答案
def ask_ollama(prompt, temperature=0.7):
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": temperature,
        "stream": False
    }
    response = requests.post(OLLAMA_API, json=data, timeout=300)

    if response.status_code >= 400:
        try:
            err_body = response.json()
        except Exception:
            err_body = response.text
        raise RuntimeError(f"Ollama 请求失败({response.status_code}): {err_body}")

    result = response.json()

    if "response" in result:
        return result["response"]

    if "error" in result:
        raise RuntimeError(f"Ollama 返回错误: {result['error']}")

    if "choices" in result and result["choices"]:
        message = result["choices"][0].get("message", {})
        if "content" in message:
            return message["content"]

    raise RuntimeError(f"Ollama 返回格式异常: {result}")

# 自动找最优温度
def get_best_temp(question, ref_answer, dataset_name=None):
    # 两阶段搜索：先粗筛（coarse），再在优选区间做细化（fine）
    coarse_temps = [0.05, 0.2, 0.5, 0.8, 1.1, 1.5, 2.0]
    score_dict = {}

    # 粗筛阶段：每个温度生成多次取平均
    for t in coarse_temps:
        scores = []
        for _ in range(4):
            pred = ask_ollama(question, temperature=t)
            s = compute_weighted_score(question, pred, ref_answer, dataset_name)
            scores.append(s)
        score_dict[float(t)] = np.mean(scores)

    # 选出粗筛中表现最好的 top-k 温度，构造细化区间
    top_k = 3
    sorted_coarse = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    top_temps = [t for t, _ in sorted_coarse[:top_k]]

    # 细化区间：以 top_temps 的 min/max 为中心扩展一个小范围
    span = 0.25
    low = max(0.01, min(top_temps) - span)
    high = max(top_temps) + span

    # 生成更密的温度网格进行细化搜索
    fine_temps = list(np.linspace(low, high, 9))
    for t in fine_temps:
        t = float(round(t, 4))
        if t in score_dict:
            continue
        scores = []
        for _ in range(4):
            pred = ask_ollama(question, temperature=t)
            s = compute_weighted_score(question, pred, ref_answer, dataset_name)
            scores.append(s)
        score_dict[t] = np.mean(scores)

    # 返回所有采样点的得分映射及最佳温度
    best_t = max(score_dict, key=score_dict.get)
    return best_t, score_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动寻找最优温度并支持分段续跑")
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        default=0,
        help="可选：从第 80*n 个样本开始处理；默认会根据已保存的 CSV 自动续跑"
    )
    args = parser.parse_args()

    data = load_local_data()
    print(f"共加载有效样本：{len(data)} 条")

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

    for idx in range(start_index, len(data)):
        d = data[idx]
        print(f"处理第 {idx+1} 题...")
        try:
            best_t, scores = get_best_temp(d["question"], d["ref"], d.get("dataset", DEFAULT_DATASET_NAME))
            row = {
                "question": d["question"],
                "dataset": d.get("dataset", DEFAULT_DATASET_NAME),
                "best_temperature": best_t
            }
            append_result_row(row)
            existing_rows.append(row)

            if (idx + 1) % SAVE_EVERY == 0:
                print(f"已保存进度：{OUTPUT_CSV_PATH}（共 {len(existing_rows)} 条）")
        except Exception as e:
            print(f"第 {idx+1} 题失败，已跳过：{e}")

    dataset = load_existing_results(OUTPUT_CSV_PATH)
    save_dataset_rows(dataset)
    print(f"完成！数据集已保存：{OUTPUT_CSV_PATH}")
    print(pd.DataFrame(dataset))