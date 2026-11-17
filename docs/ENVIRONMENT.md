# 环境与复现范围

## 本地客户端

记录于 2026-11-17 的 Windows Python 3.12 环境：LeRobot 0.7.0、PyTorch 2.8.0+cu128、NumPy 2.2.6、OpenCV headless 4.13.0.92、pynput 1.8.2、safetensors 0.8.0。`requirements-client.txt` 是客户端依赖入口，不是云端训练依赖锁文件；尚未验证全新环境安装。

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-client.txt
```

按实际平台安装适合的 PyTorch 构建；纯仓库检查不需要这些依赖。

### RTC 队列兼容快照

原实验环境的 LeRobot ActionQueue 包含本项目依赖的 `snapshot()`、generation/cursor 与 continuation 对齐修改。仅安装同名上游版本，不保证具有这些接口。

仓库在 `third_party/lerobot_rtc/action_queue.py` 保留了实际使用的文件以及原 Apache-2.0 许可证。下面的工具默认只显示路径；使用 `--apply` 才会在当前 Python 环境中替换该单个模块，并保存备份。应在项目专用虚拟环境执行。

```powershell
.\.venv\Scripts\python.exe tools/install_rtc_queue.py
.\.venv\Scripts\python.exe tools/install_rtc_queue.py --apply
.\.venv\Scripts\python.exe -m unittest discover -s scripts -p "test_blockbox*.py"
```

该文件是实验兼容快照，不代表整个 LeRobot 包已被完整锁定。guided RTC 去噪引导仍使用 LeRobot 提供的实现。

## 云端训练与推理

历史环境采用 LeRobot 0.4.4 源码、PEFT 0.21.3、OpenPI 兼容 Transformers 4.53.3、tokenizers 0.21.4。训练还使用了每四循环一次优化器更新的累积补丁、本地分词器、兼容处理器配置和训练集统计。不能直接把 Windows 0.7.0 客户端依赖当作训练环境。

本仓库保留配置与启动记录，未包含完整的云端环境锁文件、训练累积补丁或可下载模型，因此不声称可从零一键重训。细节见 `so101-block-box-lora-20261117.md`。云端 shell 脚本中的 `/root/autodl-tmp` 和模型目录需与实际部署对应；服务器名 `your-gpu-host.example.com`、SSH 端口 `2222` 均为占位值。

## 设备与模型

- 方块任务使用 SO-101 从臂、6 维关节、身体关节为度、夹爪为百分比；历史端口为 COM6。
- front=1、wrist=2 是本次实验的相机编号，换机或重插后需查看实际画面。
- 个人录制校准 `official_block_follower.json` 不随仓库分发，需为自己的设备准备匹配校准。
- 原 IRA 底座、020000 LoRA 和处理器需自行获取并配置；共享处理器是本次对照条件的一部分。
- 自动复位为关节空间插值，无环境避障能力；正式试验仍需人工摆放物体和标注。

## 数据与结果

`data/` 中只提供元信息，不能据此直接启动训练。`manifest.json` 和 `audit.json` 保存原始实验路径；这些路径以及 `video_exists` 是当次实验归档状态，不代表所有录像已上传 GitHub。全量数据和权重保留于原实验环境，示例视频只展示部分回合。

历史脚本还涉及 OpenPI、MuJoCo、机器人 URDF 和不同版本 LeRobot；这些依赖按各自实验文档准备，未纳入统一客户端依赖。源码中的本地路径和旧版本接口应在使用前逐项核对。
