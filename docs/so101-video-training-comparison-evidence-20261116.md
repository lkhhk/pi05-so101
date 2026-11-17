# 视频对照：本地训练证据核对（2026-11-16）

本页只记录本地可核实事实；视频内容由主调查单独获取。没有连接云端或机器人，没有运行模型。历史准备文档已落后于本地产物，不能用“尚未采集”描述当前状态。

## 最新数据与完成进度

- 本地 `data/so101-medicine-retry/so101-medicine-121/meta/info.json:4-12` 声明 121 条、33890 帧、30 FPS，`splits.train=0:121`。`results/lerobot-train-v6-45000-adapter-local/train_config.json:3-5` 指向同名云端数据集且 `episodes=null`，没有在该训练配置中选择留出集。
- `results/lerobot-train-v6-checkpoint-045000/training_state/training_step.json:2` 为 `step=45000`，同目录有优化器、调度器、随机数状态和适配器，支持“本地已有 45000 步检查点”。
- `results/lerobot-train-v6-45000-adapter-local/train_config.json:179-185` 中目标是 80000 步、batch size 2、eval_freq 0。80000 是目标，不能说已经完成；未查询当前云端状态。
- 最新资料中没有标准化真机任务成功率评估表。服务器输出动作块证明推理运行，不等于任务成功。

## QUANTILES 已经启用，并非遗漏的开关

- 最新适配器 `config.json:74-77`、`policy_preprocessor.json` 和 `policy_postprocessor.json` 均声明 STATE/ACTION 使用 QUANTILES；VISUAL 为 IDENTITY。早期 500 步 LeRobot 适配器 `results/lerobot-pi05-so101-000500/config.json:74-77` 与前处理器也已经采用 QUANTILES。
- 直接读取 safetensors 头部和 F32 数据验证：前处理器包含 `action.q01/q99`、`observation.state.q01/q99`，各为六维；对应 count 均为 33890。它们与 121 条数据的 `meta/stats.json` 最大绝对误差小于 3.8e-6（F32 精度差）。后处理器同样保存对应统计。这比只看配置名更能支持“已经使用本任务分位数统计”。
- 本地官方 LeRobot 源码 `vendor/lerobot043/lerobot/policies/pi05/configuration_pi05.py:66-70` 默认也是 STATE/ACTION QUANTILES。
- `vendor/lerobot043/lerobot/processor/normalize_processor.py:362-377` 要求 q01/q99，公式为 `2*(x-q01)/(q99-q01)-1`，反变换为 `(y+1)*(q99-q01)/2+q01`。此分支没有把所有超分位范围数值裁剪到 [-1,1]，因此说明时应说“把 1% 和 99% 分位数映射到 -1 和 1”。
- `vendor/lerobot043/lerobot/policies/pi05/processor_pi05.py:138-155` 先归一化，再把状态离散化用于提示词，输出端做反归一化。不能只改推理端 normalization 类型来补救用另一种尺度训练出的模型。

## 显存、微调与动作窗口

- 最新 `train_config.json:128-161` 为 PEFT、bfloat16、gradient checkpointing、chunk_size=10、n_action_steps=10，配置 batch size 为 2（183 行）。适配器 `adapter_config.json` 为 LoRA，rank=16，主要训练 action expert 的 q/v 投影及动作相关投影层；这不是全参数微调。
- 文件名和 job_name 含 `ga4`，但本地 train_config 没有明确梯度累积字段。本地 vendored `lerobot_train.py:115-132` 每次 backward 后 optimizer.step()/zero_grad()，`Accelerator` 初始化（181 行）没有显式 accumulation 配置。云端可能用过另外的补丁或包装，但本地材料不足以断言真实有效 batch=8，也不能把 45000*8/33890 直接称作实际 epoch 数。
- 官方本地默认 action chunk 是 50（`configuration_pi05.py:37-38`），我们的 checkpoint 改为 10。按数据 30 FPS 计算，10 帧约 0.333 秒，50 帧约 1.667 秒。这是窗口时间含义，不代表视频必然采用默认 50。

## 部署证据与时延疑点

- `configs/so101-45000-local-async.json:3,26-31` 指向本地合并模型、每块 10 动作、客户端 30 FPS。
- `logs/policy-server-45000-8082-retry.err.log:237-283` 的连续预处理与推理时间约 0.40–0.43 秒，输出 shape 为 [1,10,6]。初次调用（222 行）约 7.24 秒。
- 因为 10/30≈0.333 秒，小于上述连续计算时间，持续每 30 Hz 提供不间断新动作在这些样本中有供给不足风险；应核对客户端排队/丢弃过期动作策略和实际控制周期。它是值得排查的推断，不是已经证实的失败原因。
- 本地存在 `results/lerobot-train-v6-45000-merged-local/model.safetensors`；已跨过“只有 15 MB 适配器”的早期阶段。加载日志仍有共享词嵌入相关警告，随后已有多次推理；应结合最终实际加载链路判断，不能仅凭警告宣称完全加载失败。

## 与早期记录的关系

`docs/so101-medicine-pi05-20261115.md` 和 `docs/so101-medicine-lerobot-pi05-20261115.md` 描述 50 条原始数据、42 条训练、5 条留出、500 步。这是早期实验，不能替代最新 121 条/45000 步证据。最新配置没有复用早期显式留出规则；如要报告泛化能力，需要再核实 121 条来源、失败片段筛除及独立测试划分。
