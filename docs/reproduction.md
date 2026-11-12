# 使用指南

## 1. 准备环境

训练与模型评测需要 NVIDIA GPU、CUDA，以及 Linux 或 WSL 环境；启动前至少保留 30 GiB 可用磁盘。

先安装与 CUDA 匹配的 PyTorch 和兼容的 verl，再安装项目依赖：

```bash
pip install -r requirements.txt -r requirements-training.in
python -m shopping_agent.training.environment
```

当前依赖清单未锁定完整训练环境，不能保证最新版 verl 兼容；环境检查通过仅表示 CUDA 和所需接口可用。

## 2. 准备数据与模型

将整理后的数据放到项目根目录的 `data/`：

```text
data/
├── train.parquet    训练题，含历史对话
├── test.parquet     开发测试题，含历史对话
└── products.jsonl   商品记录
```

数据包 `shopping-companion-data.zip` 解压到项目根目录即可；数据不随代码提交，可单独作为 GitHub Release 附件发布。

如果已有自行整理的 Stage-2 数据，可用以下命令检查并导入。它不会自动从原始全量数据重建本项目的 136/34 划分：

```bash
python -m shopping_agent prepare \
  --train-source /path/to/train.parquet \
  --test-source /path/to/test.parquet \
  --products-source /path/to/products.jsonl
```

导入要求 `data/` 中没有同名文件；训练启动时也会检查会话隔离、商品与偏好支持、优惠规则。

另外下载 [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)，训练时通过 `--model` 指定本地模型目录。

## 3. 开始训练

```bash
python -m shopping_agent train --model /path/to/Qwen3-4B-Instruct-2507 \
  --output outputs/grpo_new
```

默认配置为 `configs/grpo_136x34_v12.json`，可通过 `--config` 更换。训练会自动执行前后评测，轨迹和检查点保存在输出目录；每次运行须使用新目录。

显存不足时，减小 `--rollout-batch-size` 和 `--ppo-micro-batch-size`。仅查看启动参数可运行 `python -m shopping_agent train --dry-run`。

## 4. 查看评估报告

```bash
python -m shopping_agent evaluate outputs/grpo_new --report
```

该命令复核已保存的奖励和报价，生成对比报告与曲线，不会重新运行模型。
