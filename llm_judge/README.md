# Cell Type LLM Judge

使用 DeepSeek API 作为 LLM Judge 来评估细胞类型预测结果的工具。

## 功能特点

- 🤖 使用 DeepSeek API 进行语义化评估
- 📊 六级语义关系分类（equivalent, parent-child, same_major_lineage, partially_related, ambiguous, different）
- 🎯 基于生物学知识的评分系统（0.0-1.0）
- ⚡ 支持异步批量处理
- 📈 详细的统计分析报告

## 评估标准

### 1. equivalent (1.0)
- 同义词、缩写形式
- 例如: "Natural killer cell" vs "NK cell"

### 2. parent-child (0.7-0.9)
- 父子关系（层级关系）
- 例如: "T cell" vs "CD8+ T cell"

### 3. same_major_lineage (0.5-0.7)
- 相同主要谱系但不同分支
- 例如: "NK cell" vs "CD8+ T cell" (both lymphocytes)

### 4. partially_related (0.3-0.5)
- 部分相关但不同细胞身份
- 例如: "Macrophage" vs "Dendritic cell"

### 5. ambiguous (0.1-0.3)
- 过于宽泛或模糊的预测
- 例如: "CD4+ T cell" vs "Immune cell"

### 6. different (0.0-0.1)
- 完全不相关
- 例如: "T cell" vs "Fibroblast"

## 安装依赖

```bash
pip install openai pydantic
```

## 环境变量

设置 DeepSeek API Key:

```bash
export DEEPSEEK_API_KEY="your-api-key-here"
```

## 使用方法

### 方法 1: 使用运行脚本（推荐）

```bash
# 使用默认配置
bash /home/scbjtfy/cell-o1/llm_judge/run_celltype_llm_judge.sh

# 指定输入文件和输出目录
bash /home/scbjtfy/cell-o1/llm_judge/run_celltype_llm_judge.sh \
    /path/to/predictions.json \
    /path/to/output/dir
```

### 方法 2: 直接运行 Python 脚本

```bash
python /home/scbjtfy/cell-o1/llm_judge/celltype_llm_judge.py \
    --predictions_path /path/to/predictions.json \
    --output_dir /path/to/output/dir \
    --max_samples 200 \
    --batch_size 50 \
    --max_concurrent 5 \
    --llm_model deepseek-chat \
    --llm_api_key your-api-key
```

## 输入格式

输入 JSON 文件应包含以下字段：

```json
[
  {
    "model_name": "ncbi/Cell-o1",
    "dataset_id": "A013",
    "index": 331,
    "task_type": "cell type",
    "ground_truth": "Naive CD4 T cell",
    "predicted_answer": "lymphoid",
    "question": "...",
    "full_response": "...",
    "group": "..."
  }
]
```

必需字段：
- `ground_truth`: 真实的细胞类型
- `predicted_answer`: 预测的细胞类型

## 输出文件

### 1. celltype_judged_results.json

包含所有原始预测数据和 LLM 判断结果：

```json
[
  {
    "model_name": "ncbi/Cell-o1",
    "index": 331,
    "ground_truth": "Naive CD4 T cell",
    "predicted_answer": "lymphoid",
    "llm_judgment": {
      "semantic_relation": "parent-child",
      "score": 0.75,
      "explanation": "Prediction is a parent lineage of the ground truth..."
    },
    "judgment_timestamp": "2025-10-29T12:34:56"
  }
]
```

### 2. celltype_judgment_analysis.json

统计分析报告：

```json
{
  "timestamp": "2025-10-29T12:34:56",
  "total_processing_time": 123.45,
  "analysis": {
    "total_samples": 200,
    "semantic_relation_distribution": {
      "equivalent": 50,
      "parent-child": 80,
      "same_major_lineage": 30,
      "partially_related": 20,
      "ambiguous": 15,
      "different": 5
    },
    "score_statistics": {
      "average_score": 0.756,
      "exact_match_rate": 0.250,
      "good_match_rate": 0.650,
      ...
    }
  }
}
```

### 3. celltype_llm_judge.log

详细的运行日志。

## 参数说明

### 必需参数

- `--predictions_path`: 预测结果 JSON 文件路径
- `--output_dir`: 输出目录

### 可选参数

- `--max_samples`: 最大评估样本数（用于测试，-1 表示全部）
- `--random_seed`: 随机采样种子（默认: 42）
- `--batch_size`: 批处理大小（默认: 50）
- `--max_concurrent`: 最大并发 API 调用数（默认: 5）
- `--delay_between_batches`: 批次间延迟秒数（默认: 1.0）
- `--llm_model`: LLM 模型名称（默认: deepseek-chat）
- `--llm_api_key`: DeepSeek API Key
- `--base_url`: API 基础 URL（默认: https://api.deepseek.com）

## 性能调优

### 加快评估速度

- 增加 `--max_concurrent` (例如 10)
- 减少 `--delay_between_batches` (例如 0.5)
- 增加 `--batch_size` (例如 100)

### 避免 API 限流

- 减少 `--max_concurrent` (例如 3)
- 增加 `--delay_between_batches` (例如 2.0)

## 示例

### 快速测试（200 个样本）

```bash
python celltype_llm_judge.py \
    --predictions_path predictions.json \
    --output_dir results/test \
    --max_samples 200
```

### 完整评估（所有样本）

```bash
python celltype_llm_judge.py \
    --predictions_path predictions.json \
    --output_dir results/full \
    --max_samples -1 \
    --max_concurrent 10
```

## 故障排除

### API Key 错误

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

### 速率限制

如果遇到 429 错误，增加 `--delay_between_batches` 或减少 `--max_concurrent`。

### JSON 解析错误

检查输入文件格式是否正确，确保包含 `ground_truth` 和 `predicted_answer` 字段。

## 许可证

MIT License

