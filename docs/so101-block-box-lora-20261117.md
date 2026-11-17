# 方块入盒：IRA π₀.₅ 底座上的 LoRA 训练

2026-11-17，用户授权修复环境并启动训练，明确保留 20,000 个循环步、每 4,000 步保存。SSH 为 `ssh -p 2222 root@your-gpu-host.example.com`；凭据不写入项目文件。

## 配置

- 原始底座：`/root/autodl-tmp/ira-pi05-v7-full-v2-008000/outputs_V8/train/pi05_6gpu_fsdp_V2/checkpoints/008000/pretrained_model`。
- 兼容配置入口：`/root/autodl-tmp/ira-pi05-v7-full-v2-008000-blockbox-20261117`。权重链接到原始文件；仅在独立配置副本中移除两个 `enabled: false` 的旧版相对/绝对动作处理步骤，并将分词器指向本地路径。
- LeRobot 源码：`/root/autodl-tmp/lerobot-044-src`；Python：`/root/autodl-tmp/pi05-run/.venv/bin/python`。
- 启动脚本：`/root/autodl-tmp/train_so101_block_box_lora.sh`；原脚本备份：`/root/autodl-tmp/so101-block-box-setup-20261117/original-train.sh`。
- LoRA rank 16，使用该版本 PI05 默认 target_modules：动作专家注意力 q/v 及匹配的 state/action/time 投影。以实际可训练参数日志为准。
- BF16、gradient checkpointing；batch 2、gradient accumulation 4，单卡有效 batch 8。
- 峰值学习率 `1e-4`；沿用 warmup 200、cosine decay 5000、最低学习率 `2.5e-5`。
- `steps=20000`、`save_freq=4000`、`log_freq=100`、workers 4；`eval_freq=0`、不上传 Hub、不启用 W&B。
- 此版本 steps 计数的是训练循环；每 4 个循环执行一次 optimizer/scheduler 更新，因此 20,000 个循环约为 5,000 次优化器更新，4,000 个循环的保存间隔约为 1,000 次优化器更新。
- chunk_size/n_action_steps 均为 50；front → desk_view，wrist → wrist_left。

## 数据与统计

原始数据 `/root/autodl-tmp/so101-block-box`，50 段、11,580 帧、30 FPS；6 维关节位置状态与动作。按既有划分使用 42 段训练、8 段留出。留出 episode 为 `[1,9,12,13,14,19,21,45]`。

训练数据入口 `/root/autodl-tmp/so101-block-box-trainstats-20261117` 链接原始 data/videos，复制 metadata，仅替换统计文件。统计由 42 段训练 episode 聚合，覆盖 9,616 帧；action 与 observation.state 的均值、标准差已和实际训练帧独立比较通过。原始数据保持不变。准备脚本及报告位于 `/root/autodl-tmp/so101-block-box-setup-20261117/`。

留出数据不会因 eval_freq=0 自动评估；后续需单独进行离线诊断与真机成功率评估。训练启动不代表任务已学会。

## 兼容性与分词器

之前的启动日志报 `An incorrect transformer version is used`。本次重新检查时，环境已变为 `/root/autodl-tmp/transformers-openpi-src` 的 Transformers 4.53.3，`check_whether_transformers_replace_is_installed_correctly()` 返回 True；本次没有重新安装该依赖。观测版本为 PEFT 0.21.3、tokenizers 0.21.4。

后续启动遇到 Hugging Face 分词器请求超时。将本地缓存 revision `35e4f46485b4d07967e7e9935bc3786aad50687c` 的六个文件上传到 `/root/autodl-tmp/paligemma-tokenizer-35e4f464`，逐文件 SHA-256 与本地一致。AutoTokenizer 离线加载通过，长度 257153、BOS=2、EOS=1。启动脚本设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` 和 `PYTHONUNBUFFERED=1`。

## 运行入口

- 日志：`/root/autodl-tmp/so101-block-box-lora-train-20261117-lr1e4-offline.log`。
- PID 文件：`/root/autodl-tmp/so101-block-box-lora-train.pid`。
- 输出：`/root/autodl-tmp/so101-block-box-lora-run-20261117`。

在服务器查看：

```bash
tail -f /root/autodl-tmp/so101-block-box-lora-train-20261117-lr1e4-offline.log
```

状态以最新日志和进程为准；不要在已有训练进程运行时重复启动脚本。

## 启动验证

09:22:36（北京时间）进入训练循环。日志确认 `All keys loaded successfully!`，PEFT 包装成功，可训练参数 1,287,168，总参数 3,618,044,688。第 100 个循环记录 loss=0.535、grad_norm=0.013、lr=6.6e-6（warmup 期间），训练继续推进至至少 114 个循环。一次显存读数约 8,269 MiB、GPU 利用率 95%；这些是启动阶段观测，不是峰值显存或任务成功率。启动 PID 为 41178。
