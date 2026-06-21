# Temperature Predictor for LLM Generation

本项目针对大型语言模型（LLM）生成温度参数自动化推荐问题，搭建完整数据集构建、回归模型训练与温度预测全流程。核心方案：对同一问题设置多组温度调用LLM生成回复，通过**语义相似度**评判生成质量，筛选最优温度作为训练标签；借助Sentence\-BERT将问句转为语义向量，搭配轻量MLP回归网络训练，实现输入文本直接预测最优生成温度。

## 项目结构

```shell
.
├── download.py               
├── download_supply.py     
├── generate_1.py              
├── generate_2.py             
├── train.py                   
├── question_best_temp.csv     
├── environment.gpu.yml        
└── README.md                  
```

## 环境配置

需注意默认采取gpu环境

```shell
conda env create -f environment.gpu.yml
conda activate temp-predictor-gpu
```

## 数据集生成

1\. 执行数据下载脚本获取原始问答数据，提前配置 `SAVE_DIR` 存储路径，涉及不同数据集，可进行选择

```shell
python download.py
python download_supply.py
```

2\. 运行生成脚本批量搜索每条问句对应的最优温度，两者分别对应不同方案

```shell
python generate_1.py
python generate_2.py 
```

> **注意事项**
> 
> 1\. 脚本内内置API密钥，使用前**替换为个人有效密钥**，建议改用环境变量/\.env管理
> 
> 2\. 修改 `LOCAL_EMBEDDING_MODEL_PATH` 指向本地 `all-MiniLM-L6-v2` 嵌入模型路径
> 
> 

## 回归模型训练

```shell
python train.py --csv_path question_best_temp.csv --hidden_dim 64 --dropout 0.5 --epochs 50
```

### 可选训练参数

|参数|说明|推荐范围|
|---|---|---|
|\-\-hidden\_dim|MLP隐藏层维度|64\~128|
|\-\-dropout|随机失活比例|0\.3\~0\.6|
|\-\-epochs|训练迭代轮数|50\~100|
|\-\-lr|模型学习率|默认1e\-3|
|\-\-device|运行设备|cuda/cpu|

训练完成后，模型文件与训练日志自动保存至 `temp_predictor_model/`，最优验证集MAE可达 **0\.27\~0\.3**。

## 单条问句温度预测

```shell
python train.py --mode predict --question "在此输入需要预测温度的问题文本"
```

## 数据集字段说明

`question_best_temp.csv` 标准字段：

1\. `question`：用户输入原始问题

2\. `dataset`：数据来源领域（belle、gsm8k、writingprompts、中文知乎、中文豆瓣、Firefly等）

3\. `best_temperature`：实验筛选出的最优生成温度，取值区间 `0.01~2.0`

question_best_temp.csv数据集经过人工拼接和微调，上述下载生成程序不能完整复现示例数据集

## 实验结论

1\. 基于最多1050条小样本数据，冻结Sentence\-BERT仅训练MLP可有效规避过拟合，预测效果远优于随机猜测（随机MAE≈0\.663），验证语义特征对温度预测的有效性。

2\. 数据量为当前性能主要瓶颈，扩充数据集至5000条以上，可放开编码器微调进一步提升预测精度。

3\. 过拟合优化：隐藏层维度设置256易出现过拟合，下调至128搭配适中Dropout，可稳定验证集误差。


## 项目依赖

- Python 3\.10

- PyTorch ≥ 2\.3 \+ CUDA 加速环境

- Transformers、Sentence\-Transformers

- Pandas、NumPy、scikit\-learn

- 可选：Requests、Hugging Face Datasets

