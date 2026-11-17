# 方块入盒 guided RTC 使用与验证

## 当前状态

2026-11-17：方案 B 已实施，云端 RTC 服务已启动并完成预热；使用 020000 LoRA 与原 IRA 全量微调底座。未修改训练权重，未使用 trained RTC。

新 RTC 服务为 `127.0.0.1:8092`，同步服务 `8091` 保留。云端脚本 `/root/autodl-tmp/serve_blockbox_rtc.py`，日志 `/root/autodl-tmp/blockbox-rtc-server-020000.log`，PID 文件 `/root/autodl-tmp/blockbox-rtc-server-020000.pid`。实施时 PID 为 717745，后续应以进程与健康检查为准。

## 启动一次真机试验

先按录制场景摆好方块和盒子，使用 front=1、wrist=2，确保现场有人能停止。不要同时运行旧版或其他机械臂控制进程。

在 PowerShell 中：

```powershell
Set-Location E:\pi0.5
& ".\venvs\lerobot-rtc312\Scripts\python.exe" -u ".\scripts\run_blockbox_rtc.py" --execute --duration 35
```

客户端先进行 1 次普通请求和至少 5 次 guided 请求预检，确认部署和延迟预算。若启动抖动导致拒绝，会继续补测，最多 15 次 guided 请求；只有最近连续 5 次全部符合原延迟预算才可通过。通过后倒计时 3 秒，再使能力矩并发送动作。按 Esc 停止；正常结束或停止后保持测得的位置，不自动回零。模型是否成功抓取仍需从录像与现场判断。

如果 8092 隧道已断开，在另一个 PowerShell 窗口运行并保持窗口打开：

```powershell
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -L 127.0.0.1:8092:127.0.0.1:8092 -p 2222 root@your-gpu-host.example.com
```

若云端实例重启、服务未运行，可启动并检查日志（模型加载和预热需等待）：

```powershell
ssh -p 2222 root@your-gpu-host.example.com "bash /root/autodl-tmp/start_blockbox_rtc_server.sh"
ssh -p 2222 root@your-gpu-host.example.com "tail -30 /root/autodl-tmp/blockbox-rtc-server-020000.log"
```

不写入密码；使用 SSH 的交互密码输入。启动脚本在已识别到运行中的 RTC 服务时不会重复启动；看到 RTC_READY 后才可请求推理。

## 不动作的验证命令

默认使用先前保存的真实观测，并以 30 Hz 模拟队列消费，不连接机械臂：

```powershell
& E:\pi0.5\venvs\lerobot-rtc312\Scripts\python.exe -u E:\pi0.5\scripts\run_blockbox_rtc.py --duration 30
```

读取真实相机和关节状态、但不发送电机命令：

```powershell
& E:\pi0.5\venvs\lerobot-rtc312\Scripts\python.exe -u E:\pi0.5\scripts\run_blockbox_rtc.py --live-observation --duration 30
```

仅 `--execute` 会使能力矩并发送动作；默认模式和 `--live-observation` 都不会。

## 验证证据

8 项回归检查通过：推理期间消费步数对齐、取消后旧回复失效、墙钟过期、前缀耗尽、非有限/错误单位、裁剪后前缀与实际目标一致、剩余动作不足时拒绝请求、图像颜色/缩放/边框一致性。

测试命令：

```powershell
& E:\pi0.5\venvs\lerobot-rtc312\Scripts\python.exe E:\pi0.5\scripts\test_blockbox_rtc.py
```

| 验证 | 保存观测时序测试 | 实时观测只读时序测试 |
|---|---:|---:|
| 时长 | 30 秒 | 30 秒 |
| 模拟消费周期 | 900 | 900 |
| 成功队列补充 | 59 | 59 |
| 队列断供 | 0 | 0 |
| 电机动作命令 | 0 | 0 |
| guided 请求平均 | 0.340 秒 | 0.344 秒 |
| guided 请求 P95 | 0.404 秒 | 0.435 秒 |
| guided 请求最大值 | 0.437 秒 | 0.516 秒 |
| 平均请求大小 | 155,607 字节 | 145,886 字节 |

保存观测结果目录：`E:/pi0.5/results/blockbox-rtc-20261117/162752-045926`。

实时观测结果目录：`E:/pi0.5/results/blockbox-rtc-20261117/162943-621504`；`summary.json` 为汇总，`telemetry.jsonl` 包含完整原始/反归一化预测、实际消费步数、延迟和队列余量。每个有前缀的请求都验证了官方 RTCProcessor 在全部 10 个去噪步骤运行引导。

这些是部署时序测试，不是 RTC 真机夹取成功率。只读测试期间机械臂没有跟随预测动作，故部分预测触及校准边界也不能作为运动轨迹结论。未来执行日志和录像保存在同一 `results/blockbox-rtc-20261117/` 下的独立时间目录。

## 实现边界

官方 RTC 算法由云端训练同版 LeRobot 提供，客户端用现有 LeRobot ActionQueue。两端通过带协议版本、请求 ID 和固定检查点身份的 SSH/HTTP 长连接交换数据。图像先按模型 resize 规则缩到 168×224，用 PNG 编码并恢复 float=-1 的上下留边；插值结果额外量化误差不超过约 0.5/255。

动作队列同时保存归一化前缀和电机目标；执行期间后台生成下一块，在回复到达时按真实已消费步数删除过时前缀。迟到、前缀不足、队列耗尽、部署/单位不符会停止发送动作。没有给机械臂增加速度限制；只保留录制校准的物理范围。

当前仍使用 COM6、official_block_follower 校准、use_degrees=True，未运行自动校准或写入新的校准配置。码流时序通过不保证模型动作一定能抓取成功。

## 启动延迟拒绝的修复

16:33 的启动记录 `results/blockbox-rtc-20261117/163304-520296` 显示：5 次 guided 请求为 0.688、0.390、0.328、0.391、0.328 秒。旧版仅进行这一次采样，按最大值判定后直接退出，没有允许后续连续稳定样本证明恢复。该次 frames=0、motor_commands_sent=0。

修复保留原准入公式，新增最多 15 次 guided 探测、最近连续 5 次全部通过才进入动作阶段。默认 30 Hz / refill=35 时对应最近最大请求耗时不超过约 0.667 秒。持续慢连接仍拒绝；错误消息包含最近最大耗时和预算。执行期延迟估计从通过准入的窗口开始，避免已拒绝的启动异常值重新污染估计。

新增 `scripts/blockbox_rtc_admission.py` 与 5 项日志回放/持续慢连接/错误传播检查；连同原 8 项回归检查均已通过。用户的原始命令不变，无需重启云端服务。

修复后真实观测只读复测目录：`E:/pi0.5/results/blockbox-rtc-20261117/163626-156018`。30 秒、900 个消费周期、59 次队列补充、0 次断供、0 帧电机动作；guided 请求平均约 0.352 秒、P95 0.435 秒。预检通过窗口为 0.609、0.547、0.437、0.516、0.344 秒，原延迟预算保持不变。
