# LeRobot 官方 π0.5 微调

2026-11-15：使用 `lerobot.scripts.lerobot_train`、LeRobot 0.4.3 和 PEFT 0.18.1，在云端 RTX 4090 上用 LoRA 完成 500 步，batch size 1。输入为用户录制的 SO-101 front/wrist 视频及六维关节状态/动作。按前次评估划分，源 episode 11、24、25 未纳入；5、15、28、40、49 留出，其余 42 条共 13717 帧用于训练。42 条数据通过官方 `lerobot_edit_dataset --operation.type=delete_episodes` 生成，统计仅使用保留数据；结束帧筛选不保证所有训练示范无误。

基础模型是本机先前校验过的官方 OpenPI `pi05_base`，用官方 `examples/convert_jax_model_to_pytorch.py` 转换为 bfloat16 PyTorch。转换后的 safetensors 将语言模型共享词嵌入作为元数据别名存储，LeRobot 0.4.3 的严格加载器要求显式键；根据文件原有别名关系复制该张量，得到 813 项，加载日志显示全部键成功加载。未转换旧的 OpenPI LoRA。当前 LeRobot 0.4.3 处理器不支持新版基础模型配置里的两个默认关闭的相对动作步骤；仅从预处理配置中移除了两个 `enabled: false` 的步骤。训练使用原生官方 LeRobot 入口及 PEFT，保留量化归一化、模型 10 帧动作块。

原始 front/wrist 分别通过官方 `rename_map` 映射为 `observation.images.base_0_rgb` / `observation.images.left_wrist_0_rgb`，第三摄像头缺省。训练末步报告 loss 0.310（batch 1，单段日志，不是成功率）。500 个样本约占 13717 帧的 0.04 epoch，结果只是短时微调，尚无留出评估和真机任务成功率。

云端适配器与处理器：`/root/autodl-tmp/so101-medicine-run/lerobot-train-v1/checkpoints/000500/pretrained_model`。对应完整训练状态在同级 `training_state`；本地适配器副本为 `results/lerobot-pi05-so101-000500/`。云端训练日志 `lerobot-train-official-v3.log` 已复制到 `results/lerobot-pi05-so101-train.log`，加载检查日志为 `results/lerobot-pi05-so101-reload.log`。

使用 LeRobot `make_policy` 的 PEFT 分支从该适配器及对应基础模型恢复，日志确认 `All keys loaded successfully`、`RELOAD_OK PeftModel lora_tensors 76`；只验证模型加载，没有执行动作或证明任务成功。

注意：适配器并非完整模型，`adapter_config.json` 的基础模型路径目前是 `/dev/shm/lerobot-pi05-converted-base`；这是云机临时内存盘，停止或重建实例后可能丢失。必须在实例重建前保存该基础模型或依据上文重新转换及恢复配置，才能继续加载；仅有本地 15 MB 的适配器副本不够。云端的 OpenPI 检查点和原始数据保持原位。LeRobot 0.4.3 官方同步 `make_policy` 加载适配器的流程已验证；其 `async_inference.policy_server` 直接调用策略类 `from_pretrained`，不能假定它会自动加载 PEFT 适配器，远程真机部署还需确认支持或导出完整模型。
