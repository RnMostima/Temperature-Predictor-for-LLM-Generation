import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


def resolve_encoder_model_dir(model_dir: str) -> str:
	if os.path.isabs(model_dir) and os.path.isdir(model_dir):
		return model_dir

	script_dir = os.path.dirname(os.path.abspath(__file__))
	primary = os.path.join(script_dir, model_dir)
	if os.path.isdir(primary):
		return primary

	alias_candidates = [
		os.path.join(script_dir, "model_cache", "all-MiniLM-L6-v2"),
		os.path.join(script_dir, "modelcache", "allminillm"),
		os.path.join(script_dir, "modelcache", "all-MiniLM-L6-v2"),
	]
	for candidate in alias_candidates:
		if os.path.isdir(candidate):
			return candidate

	raise FileNotFoundError(
		f"未找到本地向量模型目录: {model_dir}。"
		f"已尝试: {primary} 以及常见目录别名。"
	)


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def load_dataset(csv_path: str) -> Tuple[List[str], np.ndarray]:
	questions: List[str] = []
	temperatures: List[float] = []

	with open(csv_path, "r", encoding="utf-8-sig", newline="") as file:
		reader = csv.DictReader(file)
		for row in reader:
			question = (row.get("question") or "").strip()
			temp_text = (row.get("best_temperature") or "").strip()

			if not question or not temp_text:
				continue

			try:
				temperature = float(temp_text)
			except ValueError:
				continue

			questions.append(question)
			temperatures.append(temperature)

	if not questions:
		raise ValueError(f"在 {csv_path} 中没有读取到有效样本。")

	return questions, np.array(temperatures, dtype=np.float32)


def train_val_split(
	questions: List[str],
	temperatures: np.ndarray,
	val_ratio: float,
	seed: int,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
	indices = np.arange(len(questions))
	rng = np.random.default_rng(seed)
	rng.shuffle(indices)

	val_size = max(1, int(len(indices) * val_ratio))
	val_idx = indices[:val_size]
	train_idx = indices[val_size:]

	train_questions = [questions[i] for i in train_idx]
	val_questions = [questions[i] for i in val_idx]
	y_train = temperatures[train_idx]
	y_val = temperatures[val_idx]

	return train_questions, val_questions, y_train, y_val


class SentenceEncoder:
	def __init__(
		self,
		model_dir: str,
		device: str = "auto",
		pooling: str = "mean",
		normalize_embeddings: bool = True,
		use_fp16: str = "auto",
	) -> None:
		if device == "auto":
			self.device = "cuda" if torch.cuda.is_available() else "cpu"
		else:
			self.device = device

		if pooling not in {"mean", "cls"}:
			raise ValueError("pooling 仅支持 'mean' 或 'cls'")

		self.pooling = pooling
		self.normalize_embeddings = normalize_embeddings

		if use_fp16 == "auto":
			self.use_fp16 = self.device == "cuda"
		elif use_fp16 in {"on", "off"}:
			self.use_fp16 = use_fp16 == "on"
		else:
			raise ValueError("use_fp16 仅支持 auto/on/off")

		self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
		self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
		self.model.to(self.device)
		if self.device == "cuda" and self.use_fp16:
			self.model.half()
		self.model.eval()

	@torch.no_grad()
	def encode(
		self,
		texts: List[str],
		batch_size: int = 32,
		max_length: int = 128,
	) -> np.ndarray:
		all_embeddings = []

		for start in range(0, len(texts), batch_size):
			batch_texts = texts[start : start + batch_size]
			tokens = self.tokenizer(
				batch_texts,
				padding=True,
				truncation=True,
				max_length=max_length,
				return_tensors="pt",
			)
			tokens = {key: value.to(self.device, non_blocking=True) for key, value in tokens.items()}

			if self.device == "cuda" and self.use_fp16:
				with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
					outputs = self.model(**tokens)
			else:
				outputs = self.model(**tokens)

			token_embeddings = outputs.last_hidden_state
			if self.pooling == "cls":
				sentence_embeddings = token_embeddings[:, 0]
			else:
				attention_mask = tokens["attention_mask"].unsqueeze(-1)
				masked_embeddings = token_embeddings * attention_mask
				summed = masked_embeddings.sum(dim=1)
				counts = attention_mask.sum(dim=1).clamp(min=1)
				sentence_embeddings = summed / counts

			sentence_embeddings = sentence_embeddings.float()
			if self.normalize_embeddings:
				sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

			all_embeddings.append(sentence_embeddings.cpu().numpy().astype(np.float32))

		return np.vstack(all_embeddings)


class TemperatureRegressor(nn.Module):
	def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(hidden_dim, 1),
		)

	def forward(self, inputs: torch.Tensor) -> torch.Tensor:
		return self.layers(inputs).squeeze(-1)


@dataclass
class TrainConfig:
	epochs: int
	batch_size: int
	learning_rate: float
	hidden_dim: int
	dropout: float
	device: str


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
	model.eval()
	mse_sum = 0.0
	mae_sum = 0.0
	count = 0

	with torch.no_grad():
		for xb, yb in loader:
			xb = xb.to(device)
			yb = yb.to(device)
			pred = model(xb)
			mse_sum += torch.sum((pred - yb) ** 2).item()
			mae_sum += torch.sum(torch.abs(pred - yb)).item()
			count += len(yb)

	mse = mse_sum / max(1, count)
	mae = mae_sum / max(1, count)
	return mse, mae


def train_regressor(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> Tuple[TemperatureRegressor, dict]:
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    model = TemperatureRegressor(
        input_dim=x_train.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    pin_memory = device == "cuda"

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(x_train_tensor, y_train_tensor),
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(x_val_tensor, y_val_tensor),
        batch_size=config.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    best_state = None
    best_val_mae = float("inf")
    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_mse, train_mae = evaluate(model, train_loader, device)
        val_mse, val_mae = evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "train_mae": train_mae,
                "val_mse": val_mse,
                "val_mae": val_mae,
            }
        )

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

        print(
            f"Epoch {epoch:>3}/{config.epochs} | "
            f"train_mae={train_mae:.4f} val_mae={val_mae:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_val_mae": best_val_mae,
        "last_train_mae": history[-1]["train_mae"] if history else None,
        "epochs": config.epochs,
        "history": history,
    }
    return model, summary


def save_artifacts(
	output_dir: str,
	model: TemperatureRegressor,
	encoder_model_dir: str,
	pooling: str,
	normalize_embeddings: bool,
	use_fp16: str,
	hidden_dim: int,
	dropout: float,
	target_min: float,
	target_max: float,
	summary: dict,
) -> None:
	os.makedirs(output_dir, exist_ok=True)

	model_path = os.path.join(output_dir, "regressor.pt")
	meta_path = os.path.join(output_dir, "meta.json")

	torch.save(
		{
			"state_dict": model.state_dict(),
			"input_dim": model.layers[0].in_features,
			"hidden_dim": hidden_dim,
			"dropout": dropout,
		},
		model_path,
	)

	meta = {
		"encoder_model_dir": encoder_model_dir,
		"pooling": pooling,
		"normalize_embeddings": normalize_embeddings,
		"use_fp16": use_fp16,
		"model_path": model_path,
		"target_min": target_min,
		"target_max": target_max,
		"best_val_mae": summary.get("best_val_mae"),
	}

	with open(meta_path, "w", encoding="utf-8") as file:
		json.dump(meta, file, ensure_ascii=False, indent=2)

	history_path = os.path.join(output_dir, "train_history.json")
	with open(history_path, "w", encoding="utf-8") as file:
		json.dump(summary, file, ensure_ascii=False, indent=2)

	print(f"模型已保存到: {output_dir}")
	print(f"最佳验证 MAE: {summary.get('best_val_mae'):.4f}")


def load_predict_components(
	model_dir: str,
	device: str = "auto",
) -> Tuple[SentenceEncoder, TemperatureRegressor, float, float, str]:
	meta_path = os.path.join(model_dir, "meta.json")
	if not os.path.exists(meta_path):
		raise FileNotFoundError(f"找不到 {meta_path}")

	with open(meta_path, "r", encoding="utf-8") as file:
		meta = json.load(file)

	if device == "auto":
		run_device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		run_device = device

	encoder = SentenceEncoder(
		meta["encoder_model_dir"],
		device=run_device,
		pooling=meta.get("pooling", "mean"),
		normalize_embeddings=bool(meta.get("normalize_embeddings", True)),
		use_fp16=meta.get("use_fp16", "auto"),
	)

	checkpoint = torch.load(meta["model_path"], map_location=run_device)
	regressor = TemperatureRegressor(
		input_dim=checkpoint["input_dim"],
		hidden_dim=checkpoint["hidden_dim"],
		dropout=checkpoint["dropout"],
	)
	regressor.load_state_dict(checkpoint["state_dict"])
	regressor.to(run_device)
	regressor.eval()

	return (
		encoder,
		regressor,
		float(meta["target_min"]),
		float(meta["target_max"]),
		run_device,
	)


def predict_temperature(
	question: str,
	encoder: SentenceEncoder,
	regressor: TemperatureRegressor,
	target_min: float,
	target_max: float,
	device: str,
) -> float:
	embedding = encoder.encode([question])
	tensor = torch.tensor(embedding, dtype=torch.float32, device=device)

	with torch.no_grad():
		pred = regressor(tensor).item()

	pred = float(np.clip(pred, target_min, target_max))
	return pred


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="根据问题预测最佳 temperature 的小模型")
	parser.add_argument("--mode", choices=["train", "predict"], default="train")

	parser.add_argument(
		"--csv_path",
		type=str,
		default="question_best_temp.csv",
		help="训练数据 CSV 路径",
	)
	parser.add_argument(
		"--encoder_model_dir",
		type=str,
		default=r"model_cache/all-MiniLM-L6-v2",
		help="本地向量化模型目录",
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		default="temp_predictor_model",
		help="输出目录（保存回归模型和元数据）",
	)

	parser.add_argument("--val_ratio", type=float, default=0.2)
	parser.add_argument("--epochs", type=int, default=70)
	parser.add_argument("--batch_size", type=int, default=32)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--hidden_dim", type=int, default=128)
	parser.add_argument("--dropout", type=float, default=0.1)
	parser.add_argument("--max_length", type=int, default=128)
	parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])
	parser.add_argument("--normalize_embeddings", action="store_true")
	parser.add_argument("--no_normalize_embeddings", action="store_true")
	parser.add_argument("--use_fp16", type=str, default="auto", choices=["auto", "on", "off"])
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cpu", "cuda"])

	parser.add_argument("--question", type=str, default="", help="预测模式下的输入问题")

	return parser


def run_train(args: argparse.Namespace) -> None:
	set_seed(args.seed)
	if args.device == "auto":
		args.device = "cuda"
	if args.device != "cuda":
		raise ValueError("训练模式要求使用 GPU，请设置 --device cuda")
	if not torch.cuda.is_available():
		raise RuntimeError("未检测到可用 CUDA GPU，无法按要求进行 GPU 训练。")
	if args.normalize_embeddings and args.no_normalize_embeddings:
		raise ValueError("--normalize_embeddings 与 --no_normalize_embeddings 不能同时使用")
	normalize_embeddings = True
	if args.no_normalize_embeddings:
		normalize_embeddings = False
	elif args.normalize_embeddings:
		normalize_embeddings = True

	print("读取数据中...")
	questions, temperatures = load_dataset(args.csv_path)
	print(f"样本量: {len(questions)}")

	print("划分训练/验证集...")
	train_q, val_q, y_train, y_val = train_val_split(
		questions,
		temperatures,
		val_ratio=args.val_ratio,
		seed=args.seed,
	)

	print("加载本地向量模型并编码文本...")
	resolved_encoder_dir = resolve_encoder_model_dir(args.encoder_model_dir)
	print(f"向量模型目录: {resolved_encoder_dir}")
	print(
		f"向量化配置: pooling={args.pooling}, normalize={normalize_embeddings}, use_fp16={args.use_fp16}, device={args.device}"
	)
	encoder = SentenceEncoder(
		resolved_encoder_dir,
		device=args.device,
		pooling=args.pooling,
		normalize_embeddings=normalize_embeddings,
		use_fp16=args.use_fp16,
	)
	x_train = encoder.encode(train_q, batch_size=args.batch_size, max_length=args.max_length)
	x_val = encoder.encode(val_q, batch_size=args.batch_size, max_length=args.max_length)

	cfg = TrainConfig(
		epochs=args.epochs,
		batch_size=args.batch_size,
		learning_rate=args.lr,
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		device=args.device,
	)

	print("训练回归头...")
	regressor, summary = train_regressor(x_train, y_train, x_val, y_val, cfg)

	save_artifacts(
		output_dir=args.output_dir,
		model=regressor,
		encoder_model_dir=resolved_encoder_dir,
		pooling=args.pooling,
		normalize_embeddings=normalize_embeddings,
		use_fp16=args.use_fp16,
		hidden_dim=args.hidden_dim,
		dropout=args.dropout,
		target_min=float(np.min(temperatures)),
		target_max=float(np.max(temperatures)),
		summary=summary,
	)


def run_predict(args: argparse.Namespace) -> None:
	if not args.question:
		raise ValueError("predict 模式需要提供 --question")

	encoder, regressor, t_min, t_max, device = load_predict_components(
		args.output_dir,
		device=args.device,
	)
	pred = predict_temperature(
		question=args.question,
		encoder=encoder,
		regressor=regressor,
		target_min=t_min,
		target_max=t_max,
		device=device,
	)
	print(f"预测 temperature: {pred:.4f}")


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()

	if args.mode == "train":
		run_train(args)
	else:
		run_predict(args)


if __name__ == "__main__":
	main()
