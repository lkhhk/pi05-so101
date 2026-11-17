# SO-101 / π₀.₅ 基础模型部署

本次部署使用未经 SO-101 微调的官方 `pi05_base`。模型端接收相机图像、六维诊断状态和任务指令，返回 50×32 的基础模型预测。预测没有 SO-101 动作单位、关节语义或经过验证的反归一化，因此客户端不发送电机目标或力矩指令。这是观测到推理的部署验证，不能视为自主任务执行。

## 云端

- SSH：`ssh -p 2222 root@your-gpu-host.example.com`。密码不保存在本项目文件中。
- 已有 OpenPI：`/root/autodl-tmp/pi05-run`，GPU 为 RTX 4090 24 GB。
- 新部署目录：`/root/pi05-base-deploy`。
- 基础权重：`/root/pi05-base-deploy/checkpoint`；其中最大一个数据分片通过符号链接位于 `/root/autodl-tmp/pi05-base-deploy-shard`。两者都必须保留。
- 下载依据：官方 Google Storage 文件清单和 CRC32C。大文件可通过固定版本镜像传输，但必须通过官方清单校验。
- `VERIFIED` 文件仅在全部文件校验成功后产生。
- 原有 LIBERO 权重和 OpenArm 检查点保留。

启动服务，在云端运行：

```bash
cd /root/autodl-tmp/pi05-run
OPENPI_DATA_HOME=/root/autodl-tmp/openpi-cache \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
.venv/bin/python -u /root/pi05-base-deploy/serve_so101_pi05_probe.py \
  --checkpoint /root/pi05-base-deploy/checkpoint --port 8002
```

服务仅绑定云端 `127.0.0.1:8002`。首次推理包含 JAX 编译，客户端默认等待最多 300 秒。

## 本地

客户端虚拟环境：`E:\pi0.5\vendor\pi05-probe-client`，继承原 `lerobot` 环境的 NumPy、OpenCV 和舵机 SDK，另外安装 WebSocket 与 MessagePack。

建立隧道，在单独的 PowerShell 中运行并保持窗口：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -L 127.0.0.1:8002:127.0.0.1:8002 -p 2222 root@your-gpu-host.example.com
```

本次操作也提供 Python 隧道脚本 `scripts/pi05_ssh_tunnel.py`，读取系统已知 SSH 主机密钥，密码仅通过交互输入或短时环境变量传入，不写入文件。

验证推理服务，使用合成观测：

```powershell
& E:\pi0.5\scripts\run_so101_pi05_probe.ps1 -Synthetic
```

读取 SO-101 和相机并运行一次真实观测推理：

```powershell
& E:\pi0.5\scripts\run_so101_pi05_probe.ps1 -RobotPort COM6 -Camera 2 -WristCamera 0 -Prompt 'Pick up the block and place it in the box.'
```

输出在 `E:\pi0.5\results\so101-pi05-probe-*.npz` 和同名 JSON，包含输入图像、原始关节位置、诊断状态、模型预测、时间与 `executable: false`。

客户端默认使用 `my_follower_ik.json`，读取电机内的限位和 homing offset 验证其匹配后，再读取六个电机的当前位置。校准范围线性缩放到 [-1,1] 仅用于观测链路检查，不是 SO-101 训练统计量；不能据此反算模型动作。

2026-11-15 检查：`COM6` 六个电机可读取且匹配上述校准。明确使用 Windows DirectShow 后，相机 0 为 `2M`（画面可见夹爪）、相机 1 为 `ASUS FHD webcam`（笔记本内置）、相机 2 为 `Web Camera`（另一只外接相机）。两只外接相机均可提供 640×480 图像，默认以 2 为主视角、0 为腕部视角。MSMF 的设备顺序不同，不能混用编号；USB 重插后应重新枚举。旧 `my_follower_arm.json` 不匹配当前校准，不应继续使用。

## 接入自主执行的条件

需要具有明确 SO-101 关节顺序、单位、动作模式、相机定义和训练统计量的兼容检查点，并由本地控制器实现限位、速度限制、时序和断线处理。当前基础模型预测不会被截取前六维后直接发送至电机。

官方参考：

- https://github.com/Physical-Intelligence/openpi
- https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md

## 本次验证结果（2026-11-15）

- 全部 29 个权重及资产文件通过官方 CRC32C 校验；本地保存官方清单 `results/so101-pi05-base-official-manifest.json` 和验证标记 `results/so101-pi05-base-checkpoint-verified.txt`。
- 云端模型服务已加载真正的 `pi05_base`，通过 SSH 通道从本地调用成功。首次合成观测返回 50×32 有限预测，包含模型编译和连接的总耗时约 58.95 秒；相同配置预热后一次调用总耗时约 1.52 秒。这不是实际任务成功率，也不是连续控制频率测试。模型内部 `policy_timing` 沿用 OpenPI 的计时方式，不应单独作为含同步及网络的端到端延迟。
- 记录分别为 `results/so101-pi05-base-synthetic-20261115.json` 和 `results/so101-pi05-base-warm-20261115.json`，同名 NPZ 保存输入和预测。
- 首次真机观测测试时 COM6 曾断开。用户重新连接后，COM6 恢复，六个电机的校准值校验与位置读取通过，两路真实图像（主视角 2、腕部视角 0）及关节状态成功发送到云端 `pi05_base`，返回 50×32 的有限预测。一次调用从建立 WebSocket 连接到验证响应约 2.719 秒，不包含此前的串口读取和相机预热采集耗时。结果为 `synthetic: false`、`executable: false`，证明真实观测推理链路可用，不代表自主抓取或闭环控制成功。
- 未向电机发送位置目标或力矩指令。
- 下载后的系统盘剩余约 3.2 GiB，数据盘仅约 330 MiB。检查点最大分片依赖数据盘的符号链接，不能单独移除；新增数据或训练之前需要先安排空间。
