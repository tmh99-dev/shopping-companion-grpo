# 来源与许可说明

- 数据集：[yuzhan2205/Shopping-companion](https://huggingface.co/datasets/yuzhan2205/Shopping-companion)，固定修订 `9a8a2a1c13f0d88de070238352bcf71f98ca851f`。数据卡标注 CC-BY-4.0；本仓库记录ID、校验散列及派生数值统计，不重新分发商品正文与对话。
- 模型：[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)，模型卡标注 Apache-2.0。权重不在本仓库。
- 强化学习框架：[verl](https://github.com/verl-project/verl)。运行器调用 DataProto、GRPO advantage、DataParallelPPOActor；没有把第三方框架源码复制进本仓库。
- 推理与参数高效训练：Transformers、PEFT、PyTorch；列为外部运行依赖。

上游许可不会自动成为本项目自有代码的许可。发布整理未替作者选择许可证；若要允许开源复用，请在公开发布前添加作者认可的 LICENSE。
本仓库对官方任务作了模拟选购环境、数据子集和奖励改造，不声称代表数据集作者或官方评测。
