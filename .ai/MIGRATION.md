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
| 腿动作 | q_default + 0.25 × raw action | hip: q_default + 0.125 × raw action；thigh/calf: q_default + 0.25 × raw action |
| 腿 Kp / Kd | 40 / 1.2 | hip/thigh/calf 均为 80 / 2 |
| 腿 effort limit | 20 N·m | 60 N·m |
| 轮控制 | 无 | Kp=0，Kd=0.6，effort limit=17 N·m |
| 轮速目标 | 无 | sign × 5 × raw action，单位 rad/s |
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

HIM source encoder 的内部输入已按用户要求排除全部六帧的 command：`encoder_input_dim = history_length × (single_frame_dim - command_dim)`，Wolf 为 300（6×50），Black 为 252（6×42）。切片集中在 `algorithms/him/spec.py:encoder_history`，训练推理通过 `HIMEstimator.encode` 调用，TorchScript 使用相同 helper。完整观测、rollout storage history、canonical history 与外部 TorchScript 270/318 维输入均保留指令；actor 当前帧仍接收 command，target encoder 输入、velocity target、终止 successor 覆盖与 PPO/estimator 梯度边界保持。

这项修改有意改变此前 HIM source encoder 含 command 的行为。旧 HIM checkpoint 的 encoder 首层尺寸不兼容，严格加载会拒绝，不自动裁剪权重或迁移 optimizer；需重新训练。PPO checkpoint 与 PPO→HIM 显式 warm start 的 actor 映射不受 encoder 切片影响。部署侧继续构造完整 observation/history，新 TorchScript 内部负责切片；尚未运行部署后端 sim2sim。

TorchScript exporter 使用 runner 原生能力，重新加载并检查数值等价。模型 shape、joint mapping、scales、默认姿态及动作语义必须与部署后端显式匹配。ONNX metadata 对多 action term 的支持有限，不修改训练接口来适配它。

## HIM play 速度面板

项目 `play` 入口在调用 MjLab 1.6.0 CLI 期间使用本地 `HimViserPlayViewer`，退出后恢复原生 viewer 类。HIM 策略的 `Controls → Info` 在状态信息下显示两行三列读数：上方 `HIM velocity`、下方 `Actual velocity`，列为 vx/vy/vz。两行均为当前选中环境的 body-frame m/s，保留正负号和三位小数。

估计值通过仅在 play 中注册的 encoder forward hook 获取，不额外推理，并读取 `estimator_velocity/base_lin_vel` 的实际 scale 还原单位。真实速度读取 `robot.data.root_link_lin_vel_b`，与原生 base_lin_vel 观测同源。显示沿用原生状态刷新节奏；暂停时保留最近估计，GUI reset、环境 reset 和 checkpoint reload 后清空估计，下一次推理恢复。切换 policy 时重绑 hook，关闭时移除。PPO、zero/random agent 和 native viewer 的显示保持；训练、checkpoint、观测与动作接口保持。

来源为本地锁定的 MjLab 1.6.0 viewer、play、base_lin_vel 及项目 HIMPolicy。Local verification：`tests/check_him_viser_velocity.py`（不提交）；CPU PASS / CUDA PASS，覆盖 Wolf 53-D/Black 45-D history frame、估计 scale、真实值、环境切换、两类 reset、policy 替换、hook 清理、逐位相同的动作/参数，以及 CLI viewer 注入与恢复。面板初次接入时，Wolf `2026-11-07_19-46-26_wolf_flat_him/model_1000.pt` 在两种设备上均完成原生 2 env ×8 steps、Viser setup、两行读数及 GUI partial/all reset 检查。浏览器连接工具报错，排版尚未完成目视验收。命令：`MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_him_viser_velocity.py --device cpu --smoke`，CUDA 使用同命令的 `--device cuda`。当前去指令 encoder 的面板数值验证见下文；上述旧 HIM checkpoint 已不兼容，下一步使用新训练模型在 Viser 中对照估计与真实速度。

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
| all_position_default | -0.1 |
| stand_still | -0.4 |
| run_still | -0.2 |
| stand_joint_vel | -0.05 |
| stand_wheel_air | -1.0 |
| dof_pos_limits | -0.2 |
| leg_torques | -1e-4 |
| wheel_torques | -5e-4 |
| wheel_lateral_slip_l2 | 0.0 |
| soft_landing | -1e-5 |
| turn_hip_default | -3.0 |
| feet_air_time_ang_z | 50.0 |
| feet_slide_ang_z | -2.0 |
| rotation_gait_status | 2.0 |
| rotation_gait_symmetry | 15.0 |

Wolf flat 注册上述 25 项，rough 另加 wheel_force_lift，权重 0.5，共 26 项。

当前奖励基线按用户要求恢复 `2026-11-07_19-46-26_wolf_flat_him` 的既有奖励权重、公式与门控，仅将 rotation_gait_status 的 target_height 从 0.05 改为 0.04 m。新增 all_position_default、soft_landing、stand_joint_vel、stand_wheel_air 及其权重保持；纵向滑移奖励继续删除，侧向滑移权重仍为 0。HIM 去指令 encoder、53-D 观测、318-D history、300-D encoder 输入、动作、PD、DR、指令采样与 RL 配置保持。未接入的相位草稿已移除，没有相位观测或时序奖励。

Wolf 当前已提交的 RL 配置为 entropy_coef=0.006、num_steps_per_env=64，两次对照运行的 agent.yaml 一致；本次不调整 PPO/HIM 参数。

Wolf 高度目标 0.40 m，orientation 使用 projected gravity XY 的 L1 和。Hip default 为四个 hip 位置误差 L1，乘以 `clamp(1 - 0.2×min(abs(vy_cmd)/0.5,1) - 0.35×min(abs(yaw_cmd)/1.0,1), 0.5, 1)`；横移和转向指令下放宽回中约束。

`all_position_default` 是全局腿姿态惩罚：raw cost=`sum((joint_pos-default_joint_pos)^2)`，明确选择 FL/FR/RL/RR 的 hip/thigh/calf 共 12 个关节，排除 4 个驱动轮；无 command、速度或接触门控，在四个 Wolf 任务的 train/play 下始终生效，权重 -0.1。保留已有 hip_default、stand_still、run_still 等项。MjLab 1.6.0 原生 posture 为指数正奖励，本项以任务本地简单函数表达平方和，再由原生 RewardManager 乘 weight 与 dt。除新增该奖励外，旧奖励、控制、观测和 RL 配置保持。

本地验证 `tests/check_wolf_all_position_default.py`（不提交）：八任务 train/play/RL 快照仅允许 Wolf 新增该项；CPU PASS，覆盖非零默认位置、正负偏差对称、平方和而非均值、轮关节排除、跨指令一致与 weight/dt。原生 flat/rough HIM 各 1 env reset+2 steps，25/26 项注册、selector、观测 shape 与奖励累计 PASS；CUDA SKIPPED（无可用设备）。命令：`MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_all_position_default.py --device cpu --smoke`；未执行训练，下一步随当前奖励组验证训练效果。

Stand_still 在 norm(command_xy)<0.1 m/s 且 abs(command_yaw)<0.1 rad/s 时计算 12 腿关节位置误差 L1。Run_still 在 abs(command_x)>0.1 m/s、abs(command_y)<0.1 m/s 且 abs(command_yaw)<0.15 rad/s 时启用相同 L1。两项均排除轮关节。

零指令还启用 stand_joint_vel（12 腿关节速度平方和，排除轮）和 stand_wheel_air（四轮中 current_contact_time<=0 的数量），门控与 stand_still 相同。相同姿态下，持续循环摆腿和离地不再与静止四轮支撑等价。两项在转向/正常直行时为零，不直接限幅动作或钳制关节。

Leg/wheel action rate 分别选择 12/4 个 raw action 通道计算相邻帧平方差；leg/wheel torques 分别选择 12/4 个 actuator 的实际力矩计算平方和，选择集合互不重叠。Dof_pos_limits 仅惩罚 12 个腿关节超出 soft position limit 的距离，soft limit factor=0.9。

Wolf flat 高度使用 world z；rough 使用中央 7×5=35 条 terrain ray 的局部离地高度均值。两者的高度误差均为 `abs(height - 0.40)`，权重 -10.0；沿用 `base_height_l2_flat` / `base_height_l2_terrain` 函数名。两种地形共用上述 25 项权重和高度目标。Rough 的 wheel_force_lift 显式选择四个轮碰撞 geom 的 world-frame net force：水平力范数 ≥15 N 时，累加对应轮 body 的向上竖直速度（负速度裁为 0）。

## Wolf 奖励组织与软落地

Wolf 自定义奖励统一在 `tasks/wolf/mdp/rewards.py`，已移除 `turning.py`。`common.py` 提供四轮传感器的显式 FL/FR/RL/RR 解析、站立/纯 yaw 指令判定、yaw progress 和接触点运动学。奖励与 metrics 读取公共数据，各自计算返回值；软落地不调用抬腿、滑移或其他奖励。MjLab 原生 joint torque/limit 奖励继续由框架提供。原有 M20 转向奖励的公式、参数、状态、warmup、模式清理及 partial reset 保持，许可证路径已更新。

四个 Wolf 任务的 train/play 新增 `soft_landing`：`sum(norm(net_wheel_force, dim=-1) × compute_first_contact(step_dt))`，权重 -1e-5，无 command 门控，零指令也生效。公共 netforce 传感器与原生 Go1 语义一致，取每轮所有地面接触的合力；原有 rough wheel_force_lift 继续使用相同 netforce。记录 `Metrics/landing_force_mean`。本项采用 MjLab 1.6.0 当前力范数语义，不计算冲量、子步峰值或超阈值平方；20 ms 采样可能漏掉已消退的瞬时冲击。

验证工具 `tests/check_wolf_landing_encoder.py`（本地、不提交）：CPU PASS / CUDA PASS。八任务 train/play/RL 全配置差分仅允许上述 Wolf 奖励/公共合力传感器变化，完整 observation/action 与用户覆盖保持；soft_landing 数值与原生无指令门控版本一致，覆盖 first-contact、三维力范数、求和、日志、weight/dt。原版与搬迁后转向核均通过原有高度/时间窗口、yaw 正反向与停止/partial reset 数值断言；共享滑移运动学通过纯滚动、空转、侧滑、镜像轴与坐标变换检查。

HIM 检查覆盖全部六帧 command 扰动下 encoder 输入/输出逐位不变、command 梯度为零、actor 当前指令保留、原始 history 不变、target/梯度边界、Black/Wolf 动态 batch TorchScript 等价、新 state 重载与旧尺寸拒绝。Viser 单元检查确认新 encoder 下两行速度显示、scale、环境选择、reset 与 hook 清理正常。CPU/CUDA 各完成 Wolf flat HIM、rough HIM、flat PPO 与 Black flat HIM 的 2 env ×8 steps、一次 PPO/HIM 更新；HIM 覆盖 terminal successor、严格 checkpoint 重载、原生 runner JIT 导出并重新加载比对。命令：`MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_landing_encoder.py --device cpu --smoke`，CUDA 使用 `--device cuda`。来源为用户明确的 encoder 设计变更、当前项目 HIM/M20 实现与本地 MjLab 1.6.0；未修改依赖或部署仓库。下一步从新 HIM 结构训练，评估软落地对转向、落地力与抬腿意愿的影响；本次短优化检查不代表训练收敛。

## Wolf 轮地滑移奖励与诊断

Wolf 四任务的 train/play 均注册 `wheel_slip_contact`，按 FL → FR → RL → RR 显式匹配四个轮碰撞 geom；原生 maxforce reduction 每轮保留一个最强接触，读取 found、contact-frame force 和 world-frame pos/normal/tangent。公共 `wheel_force_contact` 在四任务中均使用 netforce，fields=(found, force)、track_air_time=True；供 soft_landing 与 rough 的 wheel_force_lift 独立读取。maxforce 滑移传感器保持。

`mdp/common.py` 提供共享轮接触点运动学，rewards 与 metrics 分别计算自身 cost，不互相调用。其使用 `v_contact = v_link + omega_wheel × (p_contact - p_link)` 计算轮上当前接触材料点相对静止地面的速度。omega 为轮体总角速度，包含机身、腿和轮关节运动。轮轴来自实际轮体姿态下的 MJCF local Z；与接触法线构造滚动和侧向基。仅统计 found>0、法向力绝对值>1 N 且滚动方向有效的接触；离地和未承重轮输出零。

`wheel_lateral_slip_l2` / `wheel_longitudinal_slip_l2` 分别为四轮侧向/纵向滑移速度平方和，单位 m²/s²。四任务 train/play 均保留两个 MetricsTerm，MetricsManager 独立记录未加权、未乘 dt 的 episode policy-step 均值。当前仅侧向项注册 RewardTerm（权重 0）；纵向奖励已删除，原生 RewardManager 对有效奖励乘 weight 和 policy dt。接触门控适用于所有指令，无 yaw 或预设步态门控。Actor/critic/history、action、PD 与 checkpoint contract 保持。原生 reward/step metrics 在最后一次 forward 前计算，派生量滞后一个 physics substep；本地回放 trace 采样于 forward 后。

验证：`tests/check_wolf_wheel_slip.py --device cpu/cuda` 的物理数值用例均 PASS，覆盖纯滚动、贴地空转、横向滑移、接触门控、总角速度、镜像轮轴与坐标变换；八任务 train/play/RL 快照对比确认只有 Wolf 新增上述 sensor/metrics。CUDA 回放 `2026-11-01_15-57-14_wolf_flat_him/model_1000.pt`，8 env ×600 policy steps，hip PD 按训练值设为 90/3.3，其余使用当前 play 配置并保留 actuator delay；全程状态与输出有限。接触点速度与 C MuJoCo Jacobian 独立比较，最大误差 5.7e-6 m/s 以内。Wolf rough HIM 另以缩小地形的 1 env reset+2 步验证 native metrics 有限。

回放去除前 2 s 后，vx=0.5 的侧滑 RMS 为 0.0029 m/s，yaw=+1/-1 时为 0.1178/0.1296 m/s，yaw=+2/-2 时为 0.2068/0.1970 m/s；RMS 以承重轮样本为分母。+1 yaw 下四髋平均绝对偏角约 10.6°/12.1°/13.6°/15.8°，请求目标约 4.3°/0.7°/3.1°/6.7°，存在明显目标跟踪误差与持续力矩。小 yaw 的方向差异明显，-0.5 指令下平均实际 yaw 仅 -0.021 rad/s。结果来自单 seed、每指令一个环境的短回放，尚未验证训练改善；最强单接触也不能覆盖整块轮胎接触区域，零滑移不能排除髋部静态受力。

初始奖励权重按上述回放量级选取：旧策略在 ±1 yaw 下两项加权合计约 -0.12～-0.13，直线 vx=0.5 下约 -0.00007（均未乘 dt）。`tests/check_wolf_wheel_slip.py --device cpu/cuda` 已验证四任务 train/play 的函数、权重、selector 与 metric 一致，除新增两项奖励外其余既有配置保持；CPU PASS / CUDA PASS。`tests/check_wolf_slip_reward_runtime.py` 完成 CUDA flat HIM 2 env ×128 步旧 checkpoint 推理与 rough HIM 1 env ×8 步，确认新增 raw cost 有限、非负，负权重、dt 和总奖励累计一致；未执行 PPO/HIM update 或训练消融。

本地工具 `tests/trace_wolf_wheel_slip.py` 输出 trace.npz、summary.json；结果位于 `tests/wolf_wheel_slip_results/`，不提交。新增 M20 转向行为的训练评估见下节，当前不声称已学会抬腿转向。

## Wolf M20 转向行为与电机对比

来源为 DeepRoboticsLab/rl_training `9c0e710421b165a444b2c73e5a9d65ee3fafa299` 的 M20 rough 配置、velocity/mdp/{commands,rewards,curriculums}.py 和 assets/deeprobotics.py；执行器语义核对其 README 指定的 Isaac Lab v2.3.2 `actuator_pd.py`。许可证保留在根目录 `THIRD_PARTY_NOTICES.md`。迁移使用当前安装的 MjLab 1.6.0 原生 CommandTerm、ContactSensor、ManagerTermBase 和 RewardManager，未改依赖源码。

四个 Wolf 任务的 train/play 共用以下行为：

- `WolfVelocityCommand` 继承原生 UniformVelocityCommand，互斥特殊指令采样为 20% 站立、2% 纯横移、20% 纯前后、20% 纯 yaw、38% 混合。纯前后从最初沿用 M20 的 2% 提高到 20%，加强直行回中任务覆盖。特殊指令沿用初始化范围，不随 vx 课程扩展；原生 reset、GUI 和 command metrics 保留。Wolf 继续直接采样 body-frame yaw 速度，不引入官方 heading 目标和小 XY 指令清零。
- `wheel_slip_contact.track_air_time=True`，复用原生每 physics step 接触/腾空计时，四轮显式顺序仍为 FL/FR/RL/RR。
- `turn_hip_default`：norm(cmd_xy)<0.1 且 abs(cmd_yaw)>0.1 时，惩罚四髋相对默认值的 L2 范数，权重 -3；叠加既有弱 hip_default。Thigh/calf 继续可在 yaw 下自由屈伸。官方 hip-x 在所有运动模式下回中，Wolf 将新增强约束限定于纯 yaw。
- `feet_air_time_ang_z` 恢复原版：首次落地累加 `(last_air_time-0.2)*cleared_ground`，摆腿峰值相对上次接触高度 >0 即认为离地，权重 +50；只要求 abs(cmd_yaw)>0.1，允许混合指令，不封顶腾空时长、不附加 8 cm 高度成本、不乘实际 yaw progress。partial reset 清对应环境的高度状态。
- `feet_slide_ang_z` 保持原公式，权重恢复 -2：纯 yaw 下，对接触力范数>1 N 的轮心相对 root 平移的 body-frame XY 速度范数求和。其不是接触材料点物理滑移；共享物理滑移诊断与侧向权重 0 保持。
- `rotation_gait_status` 恢复二值公式：一组对角轮均接触、另一组平均轮心高度高出 target_height 即得 1，权重 +2；不要求另一组全部离地，不使用高斯高度窗口或实际转速门控。本轮 target_height=0.04 m（旧版 0.05 m）；这是最低门槛，不是最佳高度或上限，更高仍可得分。
- `rotation_gait_symmetry` 恢复原版 5 s 窗口，对角组接触占空比接近 0.5 时得分，std=0.2，权重 +15。窗口在所有指令下累计、只在 episode/partial reset 时清对应行；不在离开转向模式时清窗，不乘实际转速进度。status/symmetry 门控恢复 norm(cmd_xy)<0.5、abs(cmd_yaw)>0.05。

Yaw tracking 恢复 1，stand_still 恢复 -0.4，run_still 恢复 -0.2。新增的全局姿态与停止动态代价继续保留。转向奖励仍不乘上游全局地形 gait_level；不改 action/observation/encoder/控制接口，不迁移完整 M20 参数。

本轮验证工具 `tests/check_wolf_restore_rewards.py`（本地、不提交）：CPU PASS / CUDA PASS。八任务 train/play/RL 全量快照仅允许上述七项既有奖励恢复；新增四项、指令、控制、观测、DR、RL 均保持。参数与旧运行 env.yaml 对照，三项转向核 AST 与 `391c6a6` 原版一致（仅公共 helper 名称适配），通过原版数值/窗口/partial reset 断言，并验证 4 cm 为下限、混合 yaw 和长腾空仍按原版计分。原生 RewardBarPanel 渲染覆盖全部 25/26 项，viewer.reward_bar_max_terms 按 len(cfg.rewards) 设置，修复默认前 20 项截断。

CPU/CUDA 均完成 Wolf flat HIM、rough HIM、flat PPO、Black flat HIM 各 2 env ×8 steps、一次 PPO/HIM update；检查 reward weight/dt、terminal successor、checkpoint 重载、原生 runner JIT 导出等价及 partial reset。命令：`MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_restore_rewards.py --device cpu --smoke`，CUDA 使用 `--device cuda`。另以 `tests/replay_wolf_motion_modes.py --device cpu --steps 600` 对 `2026-11-08_14-31-55_wolf_flat_him/model_1000.pt` 重新计分，8 env ×600 steps 的动作、状态、指令、接触与滑移 trace 逐项完全一致，仅 reward 变化；结果在本地 `tests/wolf_motion_modes_before/` 与 `tests/wolf_motion_modes_restored/`。这是固定策略重新计分，尚未证明训练恢复抬腿或降低高度；下一步用回退基线继续训练/对照，检查转速、离地高度、停止及横移，不加入固定相位。

用户确认腿部为 RobStride RS03、轮端为 RS01。本机手册 RS03(260713)、RS01(260410) 的第 6 页电气特性及机械特性页，和 Wolf 原始 URDF 的 limit 对照如下；空载转速为输出轴数值，内部减速比不再重复乘入关节参数。

| 参数 | Wolf 腿 / RS03 | Wolf 轮 / RS01 | 官方 M20 腿 / 轮训练配置 |
|---|---|---|---|
| 手册额定转矩（100 rpm 测试） | 20 N·m | 6 N·m | 未提供同条件额定规格 |
| 手册峰值转矩 | 60 N·m | 17 N·m | 配置 effort 76.4 / 21.6 N·m |
| 手册空载转速 | 195 rpm≈20.42 rad/s，±10% | 315 rpm≈32.99 rad/s，±10% | 配置 velocity 22.4 / 79.3 rad/s |
| 内部减速比 | 9:1 | 7.75:1 | 不从 actuator 配置推断 |
| 当前名义 Kp/Kd | 80/2 | 0/0.6 | 80/2；0/0.6 |
| Action scale | hip 0.125、thigh/calf 0.25 rad | 5 rad/s | hip-x 0.125、其他腿 0.25；轮 5 |
| Armature | MJCF 未指定，编译值 0 | MJCF 未指定，编译值 0 | 0 / 0.00243216 kg·m² |
| 最大命令延迟 | 12 physics steps=60 ms | 16 physics steps=80 ms | 两者 1 physics step=5 ms |
| 延迟采样 | 每 policy step 重采样 | 每 policy step 重采样 | reset 时采样 |

当前 IdealPdActuator 的力矩是 `clamp(Kp*position_error+Kd*velocity_error, ±effort_limit)`。60/17 是峰值限幅，尚无连续额定、热模型或转速相关转矩包络。URDF 的 20.42/32.99 rad/s 未成为当前 MjLab 理想 PD 的速度约束；不能把 URDF velocity 当成已经执行的速度保护。官方 DelayedPD 同样不是 DC torque-speed 模型；其 velocity 配置不能等同于已标定的实机转矩-转速曲线。

当前 Wolf 腿 PD=80/2、轮 PD=0/0.6、hip action scale=0.125、thigh/calf=0.25、wheel=5，与 M20 对照设置一致。静止且忽略 DR/限幅时，两者轮目标产生的力矩均为 `0.6*5*a=3a` N·m；Wolf 17 N·m 限幅对应 abs(a)>5.67。相比之前 Wolf 轮 Kd=2、scale=10 的 `20a` N·m，此处改变了输出对应的轮速和闭环增益，不能直接解释为相同比例的运动改善。

控制 unit 同时采用腿部与轮部设置，后续再做消融。PD 定义位于 `robots/wolf/wolf_constants.py`，按名称 action scale 位于 `tasks/wolf/env_cfg.py`；calf backlash 与 wheel target bias/scale 可选 action 分支共用相同尺度。Effort 60/17、armature、delay、DR 比例保持。MjLab 1.6.0 原生支持 scale dict，无需自定义 action 或改依赖。

当前 policy/deployment 顺序仍是 FL/FR/RL/RR，每腿 hip/thigh/calf/foot。新策略部署需要：

```text
kp:           [80, 80, 80, 0] × 4
kd:           [2, 2, 2, 0.6] × 4
action_scale: [0.125, 0.25, 0.25,  5,
               0.125, 0.25, 0.25, -5,
               0.125, 0.25, 0.25,  5,
               0.125, 0.25, 0.25, -5]
```

默认姿态与轮速观测 forward sign 保持。部署侧实际路径为 `../Sim_Deploy/quadruped_control`，其现有 Wolf test policy 腿 PD=80/3、轮 PD=0/1、hip scale=0.25、wheel scale=10，与新训练控制 contract 不同；本次未修改部署仓库或已有 policy.pt。新策略使用独立配置同步上述参数，历史策略回放使用各自保存的训练配置。PPO/HIM 输入输出 shape 仍为 53→16、318→16，该控制 unit 当时的旧 checkpoint 能按结构加载，但动作与闭环语义已改变；当前 HIM encoder 结构又已变更，旧 HIM checkpoint 不再兼容，见前文接口说明。

控制 unit 的本地验证工具为 `tests/check_wolf_m20_control.py`（不提交）。八任务 train/play/RL 的逐字段快照仅允许 Wolf damping 和 action scale 发生变化，其余含奖励、DR、delay、effort、观测与 RL 配置完全一致。CPU 原生 flat HIM、rough HIM、flat PPO（启用可选 calf backlash 与 wheel target bias/scale）各 2 env，检查正负动作的实际目标、默认 offset、四腿顺序与轮符号、初始化后执行器的 PD 输出和 60/17 限幅，再完成 8 步非零动作及 partial reset，状态和观测均有限：CPU PASS。CUDA SKIPPED（无可用设备）。命令为 `MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_m20_control.py --device cpu --smoke`；同工具 `--device cuda` 记录跳过。未执行训练或消融，下一步使用这组完整参数训练。

转向奖励 unit（控制参数调整前）的本地验证工具 `tests/check_wolf_m20_turning.py`（不提交）覆盖八任务 train/play/RL 配置差分、100000 次互斥采样与正负 yaw、课程范围隔离、纯 yaw 门控、落地事件与短腾空负值、抬升和接触组合、5 s 窗口语义、warmup 与 partial reset。CPU 数值检查 PASS；CPU 原生 flat/rough HIM 各 2 env，reset+8 步+partial reset，22/23 项 RewardManager 加权/dt/总和一致，观测和 reward 有限。CUDA SKIPPED：当前无可用设备。未运行 PPO/HIM update 或长程训练。

验证命令：`MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_m20_turning.py --device cpu --smoke`；另运行同脚本 `--device cuda`（SKIPPED）与 `tests/check_wolf_height_l1.py`（CPU PASS）。旧 `tests/check_wolf_posture_rewards.py` 仍引用已移除的 `tasks.velocity.wolf`，执行因旧路径 FAIL；本次未重写该历史工具。

早期问题来源为 MjLab play 的 `2026-11-07_19-46-26_wolf_flat_him/model_1000.pt`，其保存配置采用当前 M20 PD/尺度和最初的步态权重。CPU 固定策略 4 env×600 steps（12 s），第 6 s 将两组 ±1 yaw 切为零：切换前末 2 s，实际 yaw 均值 +0.548/-0.507 rad/s、脚峰值离地 0.221/0.210 m；切换后 4～6 s，腿速度 RMS 3.174/2.932 rad/s，四轮仍反复离地，但三项步态奖励均为零。冷启动零指令环境腿速度约 0.0016 rad/s 且四轮全接触。纯 vx=0.5 的实际 vx≈0.361，左右前轮前后差约 6.7 cm。问题是已学到的持续踏步状态与目标失衡，不能解释为零指令奖励门控泄漏。

权重校正依据：最新运行末 50 iterations 的三项步态 episode reward 合计约 1.205，yaw tracking≈0.312（混合指令统计）；固定纯 yaw 回放中三项合计约 3.29/4.18，yaw tracking≈0.467/0.456。原迁移沿用 M20 大步态权重、去掉地形系数，却保留 Wolf yaw tracking=1，且腾空时长未封顶、高度仅有下限；已按上文调整收益与停止/直行目标。PD、动作、观测、HIM、DR、termination 与用户 RL 覆盖保持。

本地工具 `tests/check_wolf_turn_stop.py`（不提交）验证八任务 train/play/RL 快照仅有明确的 Wolf 奖励/纯前后采样变化；覆盖 2～8 cm 峰值窗口、过高摆腿连续代价、0.4 s 腾空收益封顶、同向转速比例与反向/不转零收益、小 yaw 与停止门控互斥、离开转向清窗口、停止速度/触地代价和直行姿态。CPU PASS；原生 flat/rough HIM 各 2 env×8 steps、yaw→zero 和 partial reset，24/25 项 raw reward 全部有限，weight/dt/总和一致。CUDA SKIPPED（无可用设备）。命令：`MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_turn_stop.py --device cpu --smoke`。

诊断工具 `tests/replay_wolf_turn_stop.py` 将修改前与重新计分结果保存在 `tests/wolf_turn_stop_before/`、`tests/wolf_turn_stop_rescored/`（均不提交）。同一旧 checkpoint 的物理轨迹逐项完全一致；新目标下高抬腿项由 +0.89/+1.12 变为约 -3.40/-3.12，零指令持续踏步新增关节速度代价约 -6.04/-5.16、离地代价约 -2.66/-2.65，直行回中代价为原来的 5 倍。上述为固定旧策略的重新计分，不代表已经学会停止或低抬腿；本次未重新训练。

下一 behavior unit：用当前回退后的奖励目标重新训练或继续训练，按相同 yaw→zero、纯直行协议检查实际转速、脚离地高度、停止后腿速度/接触和左右前后差；之后再做奖励/PD 消融及电机包络评估。

## 课程、DR、训练状态

Command 使用 body-frame velocity sampler，每 10 s 重采样。Black standing fraction=0.1、forward fraction=0.2；Wolf 使用上节互斥特殊采样。两者 world fraction=0、heading 关闭。

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

- IMPLEMENTED：机器人独立任务架构、八任务注册、Black 8 项及 Wolf flat 25 / rough 26 项奖励、Wolf 全局腿姿态惩罚、原版转向奖励回退、4 cm 门槛与新增停止目标、课程版本、runner 与文档。
- STATIC CHECKED：Python 语法、imports、八任务注册、精确奖励表和无内部 tasks.velocity 引用。
- CONFIG CHECKED：Wolf 四任务 train/play 的奖励权重、参数和 selector；八任务 train/play 的控制、观测、DR、终止、课程、仿真、机器人及 RL 配置与修改前快照一致。课程、runner 和 RL 源码未变。
- CPU PASS / CUDA PASS：姿态 L1、hip 指令缩放、stand_still 的 XY+yaw 门控、run_still 的直线门控、腿轮动作变化率与力矩分组、抬轮力阈值及向上速度截断。
- CUDA LIGHT TEST PASS / SIM VERIFIED limited：此前奖励恢复验证中，wolf-flat 和 wolf-rough-him 各 1 env，reset 和 2 步 rollout，原有全部 15/16 项奖励输出为有限的一维 Tensor。实际力矩 selector 为互不重叠的 12/4 个 actuator；轮接触匹配四个指定 geom，force shape=[1,4,3]。Rough 验证将地形行数缩为 2。新增两项滑移奖励的验证见上节。
- TRAIN VERIFIED：已读取用户最新 1000-iteration 训练的日志/配置并复现其三项行为问题；当前奖励修正未执行 PPO/HIM update、重新训练或实机验证。

已知限制和待审查项：

1. 当前奖励修正尚无重新训练后的收敛证据；需要独立 Code Review 后再安排训练评估。
2. 未执行全部历史 checkpoint、warm-start 与 exporter 的组合矩阵；结构兼容的文件仍需按实际来源显式加载。
3. 已完成有限 CPU 环境 rollout 与固定 checkpoint 回放，未执行长程/大规模接触或独立斜面接触实验。接触力 world frame 的依据为 native netforce 实现、配置与真实输出检查。
4. EPA 过滤不保证覆盖所有退化碰撞，初始 polytope 面和极端复杂接触仍有数值风险；持续监控 NaN 和 overflow。
5. 硬件参数和实机行为未确认；部署 safety 配置不属于训练奖励或 action contract。
