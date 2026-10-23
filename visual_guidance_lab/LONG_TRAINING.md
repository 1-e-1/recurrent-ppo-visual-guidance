# 持续训练批次 01

入口目录：`reports/visual_guidance_long_training_01`。
接续 `visual_guidance_direct_cycles_01/train_100003/mission/visual_policy_final.pt`，
继承上一批 14 次更新，不加载旧残差 PPO 或模仿预训练权重。

4 个周期，每周期 4 场训练 + 1 场冻结验证；共 16 训练场景和 4 验证场景。
新训练 seed 100100–100115；新验证 seed 900010–900013，均未用于上一批。
每场最多 512 次策略采样（本版本增加整场上限，不仅是首次接近阶段），
因此计划新增策略采样上限 8192 次训练 + 2048 次验证；早停会使实际样本更少。
每 16 个训练样本或终止时更新 PPO，一次更新 4 个 optimizer epochs。
验证不更新参数。训练/验证整场成功须通过独立审计，不能以奖励上涨替代。

当前仍是早期课程：1–2 个物体、8 种形状/工具、尺寸/位置/水平角/台面高度随机，
关节在基准初始姿态附近 ±0.12 rad 扰动并做安全准入。倾斜角仍为 0。
本批不是完整随机域、不是百万级 RL 训练，也不是 1000 场正式 99% 验收。
IsaacLab 任务向量化尚未实现，实际使用已有串行 RGB-D 传感和安全后端。

原始 RGB-D 保留，每个新场景采集 786 帧部署前初始扫描。
每场启动前要求 30 GiB 余量；每次动作提案前要求 24 GiB 余量，
不够则在静止状态结束当前任务并保存已有权重/失败，不删除证据。
样本长时记录可能提前触发磁盘门限；计划数量不等于实际完成数量。

`progress.json` 给出已结束回合的累计结果；当前回合细节在
`train_<seed>/mission/learning_progress.json`。
每次 PPO 更新后原子保存 `visual_policy_latest.pt`；回合末保存 final。
验证权重不替换训练权重，断点恢复也不接续验证输出。

恢复命令：

```powershell
& 'D:\conda_envs\env_isaaclab\python.exe' -m visual_guidance_lab.train_cycles --output reports/visual_guidance_long_training_01 --resume
```

必须确认原 runner 及其采集/任务子进程已退出，源码与快照一致。
恢复会保留中断回合为失败，不重选 seed 掩盖中断；可恢复其最后完整 checkpoint。
在批次目录放置 `STOP_AFTER_EPISODE` 文件可让 runner 在回合边界停止；
不能在运行中改写训练源文件。进程身份保存在 `runner.json`，各子任务在其 job/running.json。
