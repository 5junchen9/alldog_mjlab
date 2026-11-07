# AllDog MjLab 当前状态

## 架构与任务

Black 与 Wolf 分别维护独立的 `tasks/<robot>/env_cfg.py`、`rl_cfg.py`、`runner.py` 和 `mdp/`。项目内部没有 `tasks/velocity/` 层级，也没有聚合式人工配置文件。

`env_cfg.py` 直接构造完整 ManagerBasedRlEnvCfg，逐项展示奖励函数、权重和参数。`rl_cfg.py` 定义 PPO/HIM 网络、优化参数及 runner。通用 HIM 算法位于 `algorithms/him/`，维度来自环境 specification。

注册八个任务：

```text
black-flat       black-rough       black-flat-him       black-rough-him
wolf-flat        wolf-rough        wolf-flat-him        wolf-rough-him
```

任务的 flat/rough actor、critic、action specification 一致。HIM 使用相同基础 actor 观测的 6 帧 history、3 维 estimator velocity target，以及 reset 前采样的 terminal successor target。

## 冻结控制与机器人参数

Policy 腿顺序为 FL → FR → RL → RR。Black 每腿 hip → thigh → calf；Wolf 每腿 hip → thigh → calf → foot。MuJoCo 内部排列与 policy 排列分别处理，不能推断为相同。

| 参数 | Black | Wolf |
|---|---|---|
| Action 维度 | 12 | 16，每腿 3 个位置通道 + 1 个轮速通道 |
| Physics dt / decimation | 0.005 s / 4 | 0.005 s / 4 |
| Policy period | 0.02 s，50 Hz | 0.02 s，50 Hz |
| 腿动作 | q_default + 0.25 × raw action | q_default + 0.25 × raw action |
| 腿 Kp / Kd | 40 / 1.2 | hip 90 / 3.3；thigh 80 / 3；calf 80 / 3 |
| 腿 effort limit | 20 N·m | 60 N·m |
| 轮控制 | 无 | Kp=0，Kd=2，effort limit=17 N·m |
| 轮速目标 | 无 | sign × 10 × raw action，单位 rad/s |
| 轮速方向 sign | 无 | FL:+1 / FR:-1 / RL:+1 / RR:-1 |
| 默认 root z | 0.45 m | 0.4432 m |

Black 默认 thigh 为 FL/RL +0.8014、FR/RR -0.8014；calf 为 FL/RL -1.527、FR/RR +1.527。Wolf 默认 thigh 为 FL/RR +0.82、FR/RL -0.82；calf 为 FL/RR +1.43、FR/RL -1.43。hip 和轮位置默认 0。

机器人资产仍位于各自 `xmls/`。MJCF、STL、惯性、关节和物理碰撞结构保持现状。Wolf 轮碰撞半径 0.08 m，半宽 0.0225 m；IMU body 为 imu_Link，安装偏移为 (0,0,0.05975)。缺少 STL 时加载报错。

机器人配置 factory 返回独立 EntityCfg、初始状态和 actuator 配置，避免任务实例相互修改。

训练侧 effort limit、部署 policy torque limit 与硬件安全上限属于不同层。Black 部署 policy torque limit 为 33.5 N·m；训练侧仍为 20 N·m。实机电机规格、IMU 安装方向及硬件安全上限仍需确认。

## 冻结观测与部署接口

| 任务类型 | Actor | Critic | Estimator target | TorchScript 输入 / 输出 |
|---|---|---|---|---|
| Black PPO | `[B,45]` | `[B,259]` | 无 | `[B,45]` → `[B,12]` |
| Black HIM | `[B,6,45]` | `[B,259]` | `[B,3]` | `[B,270]` → `[B,12]` |
| Wolf PPO | `[B,53]` | `[B,56]` | 无 | `[B,53]` → `[B,16]` |
| Wolf HIM | `[B,6,53]` | `[B,56]` | `[B,3]` | `[B,318]` → `[B,16]` |

Black actor term 顺序：command、base_ang_vel、projected_gravity、joint_pos、joint_vel、actions。

Black critic term 顺序：base_lin_vel、base_ang_vel、projected_gravity、joint_pos、joint_vel、actions、command、foot_height、foot_air_time、foot_contact、foot_contact_forces、height_scan。Critic 的 joint_pos/joint_vel 明确使用 FL → FR → RR → RL 的既有排列；actor 和 action 使用 policy order。末尾 height_scan 为 17×11=187 维，scale=0.2。

Wolf actor term 顺序：command、base_ang_vel、projected_gravity、leg_joint_pos、leg_joint_vel、wheel_vel、actions。角速度来自 robot/imu_ang_vel，投影重力来自 robot/imu_upvector；轮速按 policy forward sign 转换。Critic 为同顺序、同 scale 的无噪声 53 维，再追加 scale=1 的 3 维真实 base_lin_vel。

Actor scale：command=(2,2,0.25)，angular velocity=0.25，gravity=1，joint position=1，joint velocity=0.05，wheel velocity=0.05，previous action=1。

Actor raw 均匀噪声幅值：angular velocity ±0.3、gravity ±0.05、joint position ±0.08、joint/wheel velocity ±2；command 和 previous action 无噪声。默认 actor running normalization 关闭、critic normalization 开启。

环境 history 为 oldest→newest；HIM 导出输入为 newest→oldest 展平。Estimator velocity target 的 scale=2。HIM terminal frame 在 reset 前记录，history 由 observation manager 在 reset 时重置。

TorchScript exporter 使用 runner 原生能力，重新加载并检查数值等价。模型 shape、joint mapping、scales、默认姿态及动作语义必须与部署后端显式匹配。ONNX metadata 对多 action term 的支持有限，不修改训练接口来适配它。

## 奖励基线

所有奖励返回 `[num_envs]` raw Tensor，由 RewardManager 统一乘 weight 和 policy dt，不进行总奖励非负裁剪。Tracking sigma=0.25。Black linear tracking 使用 body-frame XY 平方误差和；Wolf 的 X/Y 分别使用对应轴的速度平方误差，两个独立指数奖励的权重均为 1.0。Angular tracking 使用 yaw 平方误差。

| Black 奖励 | Weight |
|---|---:|
| tracking_lin_vel | 1.0 |
| tracking_ang_vel | 0.5 |
| lin_vel_z | -2.0 |
| ang_vel_xy | -0.05 |
| orientation | -0.2 |
| base_height | -1.0 |
| action_rate | -0.01 |
| joint_power | -2e-5 |

Black 高度目标 0.43 m；joint_power=Σ|τ·q̇|，使用关节空间 qfrc_actuator。

| Wolf 奖励 | Weight |
|---|---:|
| tracking_lin_vel_x | 1.0 |
| tracking_lin_vel_y | 1.0 |
| tracking_ang_vel | 1.0 |
| lin_vel_z | -0.8 |
| ang_vel_xy | -0.05 |
| orientation | -0.5 |
| base_height | -10.0 |
| leg_action_rate | -0.01 |
| wheel_action_rate | -0.005 |
| hip_default | -0.3 |
| stand_still | -0.4 |
| run_still | -0.2 |
| dof_pos_limits | -0.2 |
| leg_torques | -1e-4 |
| wheel_torques | -5e-4 |

Wolf flat 注册上述 15 项，rough 另加 wheel_force_lift，权重 0.5，共 16 项。

Wolf 高度目标 0.40 m，orientation 使用 projected gravity XY 的 L1 和。Hip default 为四个 hip 位置误差 L1，乘以 `clamp(1 - 0.2×min(abs(vy_cmd)/0.5,1) - 0.35×min(abs(yaw_cmd)/1.0,1), 0.5, 1)`；横移和转向指令下放宽回中约束。

Stand_still 在 norm(command_xy)<0.1 m/s 且 abs(command_yaw)<0.1 rad/s 时计算 12 腿关节位置误差 L1。Run_still 在 abs(command_x)>0.1 m/s、abs(command_y)<0.1 m/s 且 abs(command_yaw)<0.15 rad/s 时启用相同 L1。两项均排除轮关节。

Leg/wheel action rate 分别选择 12/4 个 raw action 通道计算相邻帧平方差；leg/wheel torques 分别选择 12/4 个 actuator 的实际力矩计算平方和，选择集合互不重叠。Dof_pos_limits 仅惩罚 12 个腿关节超出 soft position limit 的距离，soft limit factor=0.9。

Wolf flat 高度使用 world z；rough 使用中央 7×5=35 条 terrain ray 的局部离地高度均值。两者的高度误差均为 `abs(height - 0.40)`，权重 -10.0；沿用 `base_height_l2_flat` / `base_height_l2_terrain` 函数名。两种地形共用上述 15 项权重和高度目标。Rough 的 wheel_force_lift 显式选择四个轮碰撞 geom 的 world-frame net force：水平力范数 ≥15 N 时，累加对应轮 body 的向上竖直速度（负速度裁为 0）。

## 课程、DR、训练状态

Command 使用 body-frame velocity sampler，每 10 s 重采样；standing fraction=0.1、forward fraction=0.2、world fraction=0、heading 关闭。

Black 初始 vx/vy=[-1,1]，yaw=[-π,π]，课程 vx 上限 ±2。Wolf 初始 vx/vy=[-1,1]，yaw=[-3.14,3.14]，课程 vx 上限 ±3。性能课程只扩展 vx，每次双边扩展 0.1；其余 sampler 设置保持固定。

课程根据已完成 episode 的实际 tracking 累计值评分：Black 读取 tracking_lin_vel，Wolf 读取 tracking_lin_vel_x，仅用 X 跟踪表现推进 vx 课程；Y 项独立参与奖励。累计值除以 episode policy 步数、dt 和实际 reward weight。Buffer min=256、EMA alpha=0.2、low/high 阈值=0.7/0.6、连续通过 2 次再扩范围。

课程 state version：Black=2，Wolf=4；state 保存评分 reward key、robot、stage、range、EMA、streak 和 buffer。同版本 full 恢复要求评分项一致；不兼容统计版本只允许 range/none，auto full 自动降为 range。跨机器人 state 拒绝恢复。

默认 DR profile 的开关、范围和调用频率保留在机器人 mdp/events.py。Black 默认开启摩擦、payload、COM、PD gain、encoder bias 和周期推力。Wolf 默认开启质量、惯性、摩擦、PD/motor strength、轮半径、控制延迟、推力、外力和初始腿姿态缩放；hip motor strength、calf backlash、wheel target bias/scale 和 wheel observation bias 默认关闭。

Wolf 腿/轮最大延迟分别为 3/4 个 policy step，采样更新周期仍为 decimation；外力间隔为 8 个 policy step。Play 移除 DR events、关闭 actor noise 与课程，但保留现有 reset 分布和 actuator delay 配置。

Black run_name=flat/rough/flat_him/rough_him，Wolf run_name=wolf_flat/wolf_rough/wolf_flat_him/wolf_rough_him。新实验为 black_locomotion/wolf_locomotion，默认不加载 checkpoint。Black num_steps_per_env=64，Wolf=100。网络均为 512/256/128，HIM encoder=128/64、latent=16、prototypes=32、temperature=3。

每个机器人单独的 runner.py 保存课程状态，HIM checkpoint 同时保存 estimator optimizer。Resume 严格检查算法、机器人和模型结构；PPO→HIM warm start 仅支持 flat，重新初始化 optimizer、iteration 和 estimator。明确的 checkpoint_path 可读取任意指定文件，输出仍进入新的 experiment。具体命令见根 README。

## 仿真和验证边界

Black flat/rough nconmax=35/128，Wolf flat/rough nconmax=64/128；njmax 均为 256，solver iterations=10、ls_iterations=20。Wolf 模板 root 高度为 flat 0.45 m、rough 10 m；运行时 root reset 由初始状态和地形决定。

项目 train 入口默认启用 EPA 数值过滤与终端 warning 控制；overflow bit 与结构化 metrics 保留。此任务未修改物理资产、碰撞几何、仿真补丁或求解器。

奖励验证工具：`tests/check_wolf_reward_restore.py --light`，仅本地使用，不提交。

Wolf 高度绝对值误差与 -10.0 权重由 `tests/check_wolf_height_l1.py` 单独验证（本地、不提交）：四任务 train/play 配置 PASS；CPU PASS，覆盖目标上下偏差、中央 35 ray 均值、地形整体升高与加权惩罚；CUDA SKIPPED（当前不可用）。本次高度调整未运行环境 rollout 或训练；下一步仍为奖励审查和训练评估。

- IMPLEMENTED：机器人独立任务架构、八任务注册、Black 8 项及 Wolf flat 15 / rough 16 项奖励、课程版本、runner 与文档。
- STATIC CHECKED：Python 语法、imports、八任务注册、精确奖励表和无内部 tasks.velocity 引用。
- CONFIG CHECKED：Wolf 四任务 train/play 的奖励权重、参数和 selector；八任务 train/play 的控制、观测、DR、终止、课程、仿真、机器人及 RL 配置与修改前快照一致。课程、runner 和 RL 源码未变。
- CPU PASS / CUDA PASS：姿态 L1、hip 指令缩放、stand_still 的 XY+yaw 门控、run_still 的直线门控、腿轮动作变化率与力矩分组、抬轮力阈值及向上速度截断。
- CUDA LIGHT TEST PASS / SIM VERIFIED limited：wolf-flat 和 wolf-rough-him 各 1 env，reset 和 2 步 rollout，全部 15/16 项奖励输出为有限的一维 Tensor。实际力矩 selector 为互不重叠的 12/4 个 actuator；轮接触匹配四个指定 geom，force shape=[1,4,3]。Rough 验证将地形行数缩为 2。
- TRAIN VERIFIED：NOT RUN。没有执行 PPO/HIM update、长程训练、收敛验证或实机测试。

已知限制和待审查项：

1. 新奖励基线尚无训练收敛证据；需要独立 Code Review 后再安排训练评估。
2. 未执行全部历史 checkpoint、warm-start 与 exporter 的组合矩阵；结构兼容的文件仍需按实际来源显式加载。
3. 未执行长程/大规模接触场景或 CPU 环境 rollout。接触力 world frame 的依据为 native netforce 实现、配置与真实输出检查，未新增独立斜面接触实验。
4. EPA 过滤不保证覆盖所有退化碰撞，初始 polytope 面和极端复杂接触仍有数值风险；持续监控 NaN 和 overflow。
5. 硬件参数和实机行为未确认；部署 safety 配置不属于训练奖励或 action contract。
