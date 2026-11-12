# Shopping Companion GRPO

基于 Qwen3-4B 和 GRPO 的个性化购物 Agent。结合用户当前需求与历史偏好，通过多轮工具调用完成商品搜索、属性核验、优惠计算与模拟选购。

## 功能

- **购物环境**：提供商品检索、详情查看和组合报价工具。
- **状态管理**：维护需求清单、商品核验状态与多轮上下文。
- **强化学习**：支持 LoRA 微调、在线 GRPO 训练与批量轨迹生成。
- **评估分析**：支持训练前后评测与轨迹分析。

## 数据集

本项目使用公开数据集 [Shopping-companion](https://huggingface.co/datasets/yuzhan2205/Shopping-companion)，结合用户历史对话、购物需求与商品信息，开展个性化选购任务。

主要使用其中的 Stage-2 单品与组合选购任务：模型需要选择符合用户偏好的商品，并在组合任务中考虑预算和优惠规则。训练与评估使用按会话划分的数据子集，数据需单独准备。

从原始数据中抽取 **200 道候选题**（160 道训练候选、40 道开发候选），经数据质量检查后保留 **136 道训练题和 34 道开发测试题**，并构建包含 **10,358 件商品**的检索库。

## 使用

训练与模型评测需要 NVIDIA GPU 和 CUDA 环境。按[使用指南](docs/reproduction.md)准备环境、数据与模型后运行：

```bash
# 训练并执行前后评测
python -m shopping_agent train --model /path/to/Qwen3-4B-Instruct-2507 --output outputs/grpo_new

# 分析已保存的轨迹
python -m shopping_agent evaluate outputs/grpo_new --report
```

## 项目结构

```text
shopping_agent/
├── __main__.py     统一命令入口
├── tools/          商品工具
├── harness/        交互与上下文管理
├── rewards/        奖励函数
├── training/       GRPO 训练
├── data/           数据处理
└── evaluation/     评估分析
configs/            训练配置
examples/           工具示例
tests/              核心测试
```

## 致谢

基于 [Shopping-companion](https://huggingface.co/datasets/yuzhan2205/Shopping-companion)、[Qwen3](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) 和 [verl](https://github.com/verl-project/verl) 构建。数据、模型及依赖的许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
