# 020000 远程推理真机试验

2026-11-17，北京时间约 16:00。用户授权测试，并确认方块、盒子、机械臂起始位置及现场可随时停止。

## 结果

完成一次目标时长 35 秒的同步远程推理试验，共发送 500 帧、10 个动作块，没有完成方块入盒。开始和结束画面中方块均在桌面，抽取的中间帧显示机械臂在方块附近移动。此结果仅代表 020000 的这一次部署测试，不能用于比较全部检查点。

结束时已停止发送动作，机械臂保持结束时测得的位置；没有自动回零或关闭力矩。未执行后续重试。

- 录像：`E:/pi0.5/results/blockbox-remote-20261117/155955/trial.mp4`。
- 汇总：同目录 `summary.json`。
- 逐次推理完整预测、逐帧状态与目标：同目录 `telemetry.jsonl`。
- 起止视角：`front-start.jpg`、`wrist-start.jpg`、`front-end.jpg`、`wrist-end.jpg`；拼接截图 `end.jpg`、`contact-sheet.jpg`。
- 录像只记录发送动作时的画面：500 帧以 30 FPS 编码，未包括推理等待停顿，播放时长不等于完整试验墙钟时间。

## 模型和传输

云端 `/root/autodl-tmp/serve_blockbox_remote.py` 用训练同版 LeRobot 的 make_policy 加载 020000 PEFT 及 IRA 底座，日志确认 `All keys loaded successfully!`；在内存中合并 LoRA 后，使用检查点自带的预处理、归一化和反归一化处理器。完整权重和适配器源文件没有改写。

服务只监听 `127.0.0.1:8091`，通过 SSH 本地端口转发连接，HTTP 请求携带无损 NumPy 压缩数据；云端日志 `/root/autodl-tmp/blockbox-remote-server-020000.log`，启动 PID 716750。此 PID 仅供核对，后续操作需验证当前进程。

本地脚本 `E:/pi0.5/scripts/run_blockbox_remote.py`；云端脚本源码副本 `E:/pi0.5/scripts/serve_blockbox_remote.py`。本地 Python 为 `E:/pi0.5/venvs/lerobot-rtc312/Scripts/python.exe`。如再次执行，必须先准备现场并确认云端服务、隧道可用，不能在已有试验运行时重复启动。

## 硬件和单位验证

使用录制时的 `official_block_follower` 校准，COM6，身体关节单位为度，夹爪为 0–100。录制命令没有覆盖该环境 `use_degrees=True` 默认值；实测起始肩关节约 -105.58 度，与训练数据范围一致。旧药瓶任务的 `use_degrees=False` 配置不可套用。

六个电机的校准寄存器与文件匹配，操作模式均为位置模式，试验前力矩全部关闭。通过总线直接连接、读取、设置当前位置目标后使能力矩，没有执行自动校准或 configure。相机 DirectShow 1 为 front，2 为 wrist，已实际查看画面。图像为 RGB，front → desk_view、wrist → wrist_left。只保留校准物理范围，无另加速度限制。实测没有任何动作帧触发范围裁剪。

## 时序证据和限制

试验记录了 11 次真实推理（包括结束前一次未再执行的预测）：GPU 推理均值约 0.237 秒，端到端观测/传输/推理/返回均值约 1.817 秒，范围 1.296–2.797 秒。动作块内实际发送约 27.9 Hz；块间同步等待造成停顿。不能仅据此将失败归因于网络延迟，也不能凭训练 loss 认定模型已掌握夹取。

后续可先做留出集检查点对比和动作诊断，再考虑异步动作供给或本地推理对照；当前未自动实施这些后续试验。
