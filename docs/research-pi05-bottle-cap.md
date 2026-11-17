# π0.5 拧开瓶盖：公开资源调查

调查日期：2026-11-13。范围是“在仿真中用 π0.5 拧开瓶盖”，优先寻找可以直接使用的环境、模型和评估代码。

## 结论

截至本次调查，未找到同时公开并可直接配套运行的「拧开瓶盖仿真环境 + π0.5 任务权重 + 推理评估代码」。这是本次检索的结论，并不证明不存在。当前 `pi05_libero` 权重已在 LIBERO 的单臂任务上验证，其动作、观测和任务分布不能假定适用于其他机器人和旋盖任务。[OpenPI 的 LIBERO 示例](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) 与 [RoboRecover 的 RoboTwin 接入说明](https://github.com/RUCKBReasoning/RoboRecover/blob/main/docs/ROBOTWIN.md)说明后者需要 RoboTwin 专用权重和适配器。

## 候选及限制

1. [OpenArm 控制项目](https://github.com/Manas-arumalla/openarm-control)有 MuJoCo 旋开瓶盖场景及 `openarm unscrew` 命令，是脚本控制示例，项目未提供该任务的 π0.5 策略或兼容适配器。[作者开发记录](https://github.com/Manas-arumalla/openarm-control/blob/main/docs/IMPLEMENTATION_LOG.md)说明最终示例使用固定瓶身、焊接辅助抓取以及瓶盖上升驱动器；应将它视为仿真工程近似，而非现成真实接触动力学或 π0.5 复现。是否能在当前 4090 云主机运行，尚未验证。
2. [RPG 项目](https://rpg-robot.github.io/)展示了拧开瓶盖，但其执行是多模态语言模型调用技能函数，页面标注代码尚待公开；它不是已验证可下载的 π0.5 仿真方案。
3. [RSS 2026 π0.5 专家模型](https://huggingface.co/haruki-abe/rss-2026-posttraining-pi05)公开 `seal-water-bottle-cap` 权重；[作者报告](https://takehirohara.github.io/rss_2026_workshop_posttraining/)说明任务为真实双臂系统把盖放好并拧紧，采用三路摄像头、14 维双臂状态和动作。任务方向为拧紧，不能作为仿真中拧开的现成权重。
4. [RoboTwin 的 `adjust_bottle` 源码](https://github.com/RoboTwin-Platform/RoboTwin/blob/main/envs/adjust_bottle.py)的成功条件是瓶体位置和高度，不能将它称为拧瓶盖任务。虽有 [RoboTwin 专用 π0.5 权重](https://huggingface.co/motus-robotics/pi0.5_robotwin2)，仍需确认其训练任务及与目标环境的匹配。
5. [RoboCOIN 拧开瓶盖数据](https://huggingface.co/datasets/RoboCOIN/Airbot_MMK2_unscrew_bottle_cap)描述真实 Airbot MMK2 双臂/五指夹爪数据；它是数据集而不是现成可运行的仿真策略。

## 可选后续路线

- 如果优先看到仿真拧盖动作：先评估 OpenArm 的脚本示例，并明确它不是 π0.5；随后再评估观测、动作转换以及采集训练数据的工作量。
- 如果优先研究 π0.5：寻找真实或仿真中对应机械臂和动作接口的拧开数据，对适配后的 π0.5 进行微调并评估。单张 24 GB GPU 是否足够取决于实际实现与训练方式，目前未验证。
- 继续查找已发布的同任务权重及仿真适配代码。任何候选都需逐项核对机器人、摄像头、动作表示、归一化统计和成功条件，再决定下载或执行。
