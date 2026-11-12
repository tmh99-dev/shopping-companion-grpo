# 架构与实现边界

## 模型与环境的职责

模型决定需求拆分、搜索词、查看对象、证据引用、优惠参数与最终选择。Harness 负责调用有效性、观察状态、确定性计算和日志，不能把参考答案作为下一步选品建议。

|模块|实现位置|职责|
|---|---|---|
|搜索|`shopping_agent/tools/search.py`|BM25、类别软提示、价格/卖家过滤|
|详情|`shopping_agent/tools/product.py`|原始商品字段、可选规格与规格索引校验|
|商品库|`shopping_agent/data/catalog.py`|加载原始记录，将搜索、查看和报价组合成 ProductStore|
|需求和核验|`shopping_agent/tools/requirements.py`|`set_requirements`/`record_check` 的需求覆盖、引文和逐商品证据校验|
|券和报价|`shopping_agent/tools/voucher.py`|统一优惠解析、金额计算与反馈，内部保留原有版本行为|
|提交|`shopping_agent/tools/finish.py`|已报价 ID 与模型核验记录关联、显式规格选择校验；参考 ID 匹配仅供离线诊断|
|协议|`shopping_agent/harness/protocol.py`|原生工具 Schema、tool_call 解析和保守语法补齐|
|状态与压缩|`shopping_agent/harness/state.py`、`context.py`|重复动作反馈、候选与已看商品状态、逐级压缩展示|
|轨迹调度|`shopping_agent/harness/runner.py`|训练/测试共用工具分发、有效性检查、Token 预算及串行/批量轨迹|
|奖励|`shopping_agent/rewards/scoring.py`|历史版本与 v12 公开商品用途约束|
|训练|`shopping_agent/training/policy.py`、`grpo.py`|模型与 LoRA 加载、生成、GRPO 优势、参数更新及 loss 掩码|
|数据和评估|`shopping_agent/data/`、`evaluation/`|准备与源数据审计、独立奖励/报价复算、报告|

`shopping_agent/__main__.py` 提供 prepare、train、evaluate 三个子命令。训练自动进行前后模型评测；evaluate 分析已保存轨迹，不会重复调用模型。

## 状态

`set_requirements` 保存模型从公开历史中抽取的需求；`record_check` 是对某件**已查看商品**的属性核验，不是第二份独立需求清单。
检查绑定商品 ID；选中另一商品不能继承旧商品勾选。规格存在且有证据时可以标为 `available_unselected`，最终 `finish` 仍需明确选择 `variant_index`。
搜索候选不等于已查看详情。引用路径错误不等于商品不具备属性；未知、反证与支持必须区分。

## 优惠与预算

`same shop` 要求报价组合卖家相同；`exceeds`/`>` 使用原始列价严格大于门槛，`min spend` 使用大于等于。不能先向上取整再判门槛。
数据源的折扣/最终价记法保留其取整约定，不代表所有电商平台规则。券未生效但实付符合预算的组合可以合法提交，前提是其他要求均满足。

## 奖励和策略反馈

最终奖励由原始商品字段与所选规格核验，不按模型自报的勾选直接给满分。保留逐项偏好部分分、提交和完成奖励；条件已就绪后的重复行为可每次扣0.2、相关惩罚上限1。其他惩罚有独立版本定义，不能把所有错误都视为同一扣分。

`quote_decision_gate` 在待决报价阶段仅向模型呈现 `finish` 与 `continue_shopping` 两个工具。模型仍决定是否提交；选择继续后恢复其他工具。这是训练/推理一致的显式动作空间变化，应在消融实验中单独报告，不能描述为“仅增加文字提示”。

当前默认发布配置是历史v12实验配置。通用引用诊断的咖啡小试验属于**后续独立探针**，未合入该配置，避免将训练结果归因于事后修复。

## GRPO

同题4条在线采样计算相对优势，奖励相同则跳过更新。只有新生成动作Token进入response mask，观察与历史上下文不计算动作loss。无效动作不能获得正向优势信用。
模型和LoRA为同一单GPU实例，Transformers generate按活跃轨迹协同批处理。20步后仅存在有用进展时可延长到30步。延长步数不能替代重复循环诊断。

环境与奖励中仍有基于字符串/词项的启发式规则。它们可审计，但不等于通用语义理解；例如，商品名中的主体与配件关系可能被误判，需要结合原始轨迹复核。
