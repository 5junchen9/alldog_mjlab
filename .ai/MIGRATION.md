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
| tracking_ang_vel | 0.75 |
| lin_vel_z | -1.0 |
| ang_vel_xy | -0.05 |
| orientation | -0.5 |
| base_height | -10.0 |
| leg_action_rate | -0.01 |
| wheel_action_rate | -0.01 |
| hip_default | -0.5 |
| stand_still | -0.5 |
| dof_pos_limits | -0.2 |
| leg_torques | -5e-4 |
| wheel_torques | -5e-4 |

当前四个 Wolf 任务的 train/play 共用上述 14 项基础奖励，Viser 奖励条上限同步为 14。按用户确认的 Go2W 风格基线，采用本地 HIMLoco-for-Go2W 的高度、倾斜与 hip 回中平方误差语义及对应基础权重；保留当前 X/Y 独立跟踪、腿轮分组和 soft joint limit。物理 DR 修正、HIM 去指令 encoder、53-D 观测、318-D history、300-D encoder 输入、动作、PD、指令采样和 RL 配置保持。

Wolf 高度目标仍为 0.40 m。Flat 使用 world z；rough 使用中央 7×5=35 条 terrain ray 的局部离地高度均值，两者返回 (height-0.40)^2，先对中央射线求均值再计算平方误差。orientation_l2 返回 projected gravity XY 平方和；hip_default_l2 返回四个 hip 相对默认位置的平方误差和，在所有指令下生效，无指令缩放。

stand_still 在 norm(command_xy)<0.1 m/s 且 abs(command_yaw)<0.1 rad/s 时计算 12 腿关节位置误差 L1，排除 4 个轮关节。此处保留当前 yaw 门控，相对 Go2W 仅检查 XY 的实现，纯 yaw 不再触发全腿站立回中。

Leg/wheel action rate 分别选择 12/4 个 raw action 通道计算相邻帧平方差；leg/wheel torques 分别选择 12/4 个 actuator 的实际力矩计算平方和，选择集合互不重叠。Dof_pos_limits 仅惩罚 12 个腿关节超出 soft position limit 的距离，soft limit factor=0.9。相对 Go2W，继续使用两个独立线速度指数奖励（各权重 1），总奖励保持不做非负裁剪。没有新增 legacy 的 collision、feet_stumble、dof_vel、dof_acc 或 run_still。

## Wolf 奖励组织与保留实现

Wolf 自定义奖励统一在 tasks/wolf/mdp/rewards.py，公共接触数据、轮接触点运动学和指令判定在 mdp/common.py。MjLab 原生 joint torque/limit 奖励继续由框架提供。当前未注册 all_position_default、run_still、stand_joint_vel、stand_wheel_air、soft_landing、wheel_lateral_slip_l2、turn_hip_default、feet_air_time_ang_z、feet_slide_ang_z、rotation_gait_status、rotation_gait_symmetry；rough 的 wheel_force_lift 也不再注册。这些奖励函数和原有 L1 姿态函数均保留，用于后续消融。没有固定相位奖励或相位观测。

soft_landing 的保留实现仍使用首次接触时四轮 net force 范数之和，公共 netforce 传感器与 maxforce 滑移传感器保持。当前不产生 landing_force_mean 奖励内日志；两个轮地滑移 metrics 继续独立记录。

Local verification：tests/check_wolf_clean_rewards.py（本地、不提交）。八任务 train/play/RL 快照只允许 Wolf 的 14 项奖励及 viewer 条数变化，其余物理、控制、观测、指令、RL 完全一致；原有奖励实现除两个高度函数恢复平方外，AST 保持不变。核对上下高度误差、中央 35 ray 均值和地形平移不变性、非零默认姿态与四 hip selector、平方和、纯 yaw 释放站立约束、X/Y 独立跟踪及 14 条 Viser 渲染。CPU/CUDA 运行与更新结果见下节验证边界。Wolf entropy_coef=0.006、num_steps_per_env=64 保持。

## Wolf 轮地滑移奖励与诊断

Wolf 四任务的 train/play 均注册 `wheel_slip_contact`，按 FL → FR → RL → RR 显式匹配四个轮碰撞 geom；原生 maxforce reduction 每轮保留一个最强接触，读取 found、contact-frame force 和 world-frame pos/normal/tangent。公共 `wheel_force_contact` 在四任务中均使用 netforce，fields=(found, force)、track_air_time=True；供保留的 soft_landing 与 wheel_force_lift 实现使用；当前基础奖励组不调用这两项。maxforce 滑移传感器保持。

`mdp/common.py` 提供共享轮接触点运动学，rewards 与 metrics 分别计算自身 cost，不互相调用。其使用 `v_contact = v_link + omega_wheel × (p_contact - p_link)` 计算轮上当前接触材料点相对静止地面的速度。omega 为轮体总角速度，包含机身、腿和轮关节运动。轮轴来自实际轮体姿态下的 MJCF local Z；与接触法线构造滚动和侧向基。仅统计 found>0、法向力绝对值>1 N 且滚动方向有效的接触；离地和未承重轮输出零。

`wheel_lateral_slip_l2` / `wheel_longitudinal_slip_l2` 分别为四轮侧向/纵向滑移速度平方和，单位 m²/s²。四任务 train/play 均保留两个 MetricsTerm，MetricsManager 独立记录未加权、未乘 dt 的 episode policy-step 均值。当前两项均不注册 RewardTerm；原生 RewardManager 对基础奖励乘 weight 和 policy dt。接触门控适用于所有指令，无 yaw 或预设步态门控。Actor/critic/history、action、PD 与 checkpoint contract 保持。原生 reward/step metrics 在最后一次 forward 前计算，派生量滞后一个 physics substep；本地回放 trace 采样于 forward 后。

验证：`tests/check_wolf_wheel_slip.py --device cpu/cuda` 的物理数值用例均 PASS，覆盖纯滚动、贴地空转、横向滑移、接触门控、总角速度、镜像轮轴与坐标变换；八任务 train/play/RL 快照对比确认只有 Wolf 新增上述 sensor/metrics。CUDA 回放 `2026-11-01_15-57-14_wolf_flat_him/model_1000.pt`，8 env ×600 policy steps，hip PD 按训练值设为 90/3.3，其余使用当前 play 配置并保留 actuator delay；全程状态与输出有限。接触点速度与 C MuJoCo Jacobian 独立比较，最大误差 5.7e-6 m/s 以内。Wolf rough HIM 另以缩小地形的 1 env reset+2 步验证 native metrics 有限。

回放去除前 2 s 后，vx=0.5 的侧滑 RMS 为 0.0029 m/s，yaw=+1/-1 时为 0.1178/0.1296 m/s，yaw=+2/-2 时为 0.2068/0.1970 m/s；RMS 以承重轮样本为分母。+1 yaw 下四髋平均绝对偏角约 10.6°/12.1°/13.6°/15.8°，请求目标约 4.3°/0.7°/3.1°/6.7°，存在明显目标跟踪误差与持续力矩。小 yaw 的方向差异明显，-0.5 指令下平均实际 yaw 仅 -0.021 rad/s。结果来自单 seed、每指令一个环境的短回放，尚未验证训练改善；最强单接触也不能覆盖整块轮胎接触区域，零滑移不能排除髋部静态受力。

初始奖励权重按上述回放量级选取：旧策略在 ±1 yaw 下两项加权合计约 -0.12～-0.13，直线 vx=0.5 下约 -0.00007（均未乘 dt）。`tests/check_wolf_wheel_slip.py --device cpu/cuda` 已验证四任务 train/play 的函数、权重、selector 与 metric 一致，除新增两项奖励外其余既有配置保持；CPU PASS / CUDA PASS。`tests/check_wolf_slip_reward_runtime.py` 完成 CUDA flat HIM 2 env ×128 步旧 checkpoint 推理与 rough HIM 1 env ×8 步，确认新增 raw cost 有限、非负，负权重、dt 和总奖励累计一致；未执行 PPO/HIM update 或训练消融。

本地工具 `tests/trace_wolf_wheel_slip.py` 输出 trace.npz、summary.json；结果位于 `tests/wolf_wheel_slip_results/`，不提交。新增 M20 转向行为的训练评估见下节，当前不声称已学会抬腿转向。

## Wolf M20 转向参考与电机对比

来源为 DeepRoboticsLab/rl_training `9c0e710421b165a444b2c73e5a9d65ee3fafa299` 的 M20 rough 配置、velocity/mdp/{commands,rewards,curriculums}.py 和 assets/deeprobotics.py；执行器语义核对其 README 指定的 Isaac Lab v2.3.2 `actuator_pd.py`。许可证保留在根目录 `THIRD_PARTY_NOTICES.md`。迁移使用当前安装的 MjLab 1.6.0 原生 CommandTerm、ContactSensor、ManagerTermBase 和 RewardManager，未改依赖源码。

以下记录历史 M20 对照版本的采样、转向奖励公式和权重；转向奖励函数保留，当前 14 项基础奖励组不启用这些转向项。当前统一随机采样见“课程、DR、训练状态”。传感器保持：

- 历史 WolfVelocityCommand 曾使用 20% 站立、2% 纯横移、20% 纯前后、20% 纯 yaw、38% 混合，特殊指令范围不随课程扩展。该分层逻辑现已移除，rel_pure_lateral_envs / rel_pure_longitudinal_envs / rel_pure_yaw_envs 配置字段已删除。
- `wheel_slip_contact.track_air_time=True`，复用原生每 physics step 接触/腾空计时，四轮显式顺序仍为 FL/FR/RL/RR。
- `turn_hip_default`：norm(cmd_xy)<0.1 且 abs(cmd_yaw)>0.1 时，惩罚四髋相对默认值的 L2 范数，权重 -3；叠加既有弱 hip_default。Thigh/calf 继续可在 yaw 下自由屈伸。官方 hip-x 在所有运动模式下回中，Wolf 将新增强约束限定于纯 yaw。
- `feet_air_time_ang_z` 恢复原版：首次落地累加 `(last_air_time-0.2)*cleared_ground`，摆腿峰值相对上次接触高度 >0 即认为离地，权重 +50；只要求 abs(cmd_yaw)>0.1，允许混合指令，不封顶腾空时长、不附加 8 cm 高度成本、不乘实际 yaw progress。partial reset 清对应环境的高度状态。
- `feet_slide_ang_z` 保持原公式，权重恢复 -2：纯 yaw 下，对接触力范数>1 N 的轮心相对 root 平移的 body-frame XY 速度范数求和。其不是接触材料点物理滑移；共享物理滑移诊断与侧向权重 0 保持。
- `rotation_gait_status` 恢复二值公式：一组对角轮均接触、另一组平均轮心高度高出 target_height 即得 1，权重 +2；不要求另一组全部离地，不使用高斯高度窗口或实际转速门控。本轮 target_height=0.04 m（旧版 0.05 m）；这是最低门槛，不是最佳高度或上限，更高仍可得分。
- `rotation_gait_symmetry` 恢复原版 5 s 窗口，对角组接触占空比接近 0.5 时得分，std=0.2，权重 +15。窗口在所有指令下累计、只在 episode/partial reset 时清对应行；不在离开转向模式时清窗，不乘实际转速进度。status/symmetry 门控恢复 norm(cmd_xy)<0.5、abs(cmd_yaw)>0.05。

历史 M20 回退版本曾使用 yaw tracking=1、stand_still=-0.4、run_still=-0.2；当前启用项与权重以本文件“奖励基线”的 14 项表为准。

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

下一 behavior unit：使用修正物理配置和当前 14 项基础奖励训练，按 yaw→zero、纯横移与直行协议检查实际速度、髋部偏转、离地高度和停止行为；之后逐项加入奖励做消融。

## 课程、DR、训练状态

Command 使用 body-frame velocity sampler，每 10 s 重采样，reset 时也重采样。Black standing fraction=0.1、forward fraction=0.2 保持。Wolf 所有环境均使用 MjLab 1.6.0 UniformVelocityCommand 的独立均匀采样，不分站立/直行/横移/旋转，也不按普通/高速环境分组；standing/forward/world fractions 均为 0，heading 关闭，yaw 直接采样角速度。

Black 初始 vx/vy=[-1,1]，yaw=[-π,π]，课程 vx 上限 ±2。Wolf 初始 vx/vy=[-1,1]，yaw=[-3.14,3.14]，课程 vx 上限 ±3。性能课程只扩展 vx，每次双边扩展 0.1；Wolf 扩展后的同一范围作用于全部环境，其他轴范围保持。

Wolf 在原生 _resample_command 后增加 small_xy_threshold=0.2 m/s：采样出的 norm(cmd_xy)<=0.2 时同时清零 vx/vy，保留 yaw。阈值为 0 可关闭该清零。初始 [-1,1]×[-1,1] 范围下理论清零比例为 π×0.2²/4≈3.14%，其余约 96.86% 样本保留横向指令；XY 清零样本仍可要求纯旋转，不是固定比例站立。该处理仅作用于随机重采样，原生 Viser 手动指令可设置小速度和全零。相对 Go2W，采用相同的小 XY 清零方式，取消其前 20% 高速组，保留 Wolf 的直接 yaw 速度指令。

课程根据已完成 episode 的实际 tracking 累计值评分：Black 读取 tracking_lin_vel，Wolf 读取 tracking_lin_vel_x，仅用 X 跟踪表现推进 vx 课程；Y 项独立参与奖励。累计值除以 episode policy 步数、dt 和实际 reward weight。Buffer min=256、EMA alpha=0.2、low/high 阈值=0.7/0.6、连续通过 2 次再扩范围。

课程 state version：Black=2，Wolf=4；state 保存评分 reward key、robot、stage、range、EMA、streak 和 buffer。同版本 full 恢复要求评分项一致；不兼容统计版本只允许 range/none，auto full 自动降为 range。跨机器人 state 拒绝恢复。

统一采样验证工具 tests/check_wolf_uniform_commands.py（本地、不提交），CPU PASS / CUDA PASS。八任务 train/play/RL 快照只允许 Wolf 指令配置变化，RL 工作区参数保持。每种设备各采样 200000 次，与同 seed 的原生 UniformVelocityCommand 比较：除小 XY 清零外逐位一致，yaw 不变；验证五个环境 ID 区间同分布、课程扩展覆盖所有区间、阈值边界/关闭、partial reset 与计时隔离、原生 GUI 小指令和全零。Wolf flat/rough PPO/HIM 及 Black flat HIM 各 2 env×8 steps、一次 PPO/HIM update；reward/dt、终止目标、checkpoint 重载、JIT 等价和 partial reset 通过。命令：MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_uniform_commands.py --device cpu --smoke；CUDA 使用 --device cuda。尚未重新训练验证步态，下一步按新分布训练并分非零指令评估实际 vy/yaw。

默认 DR profile 的开关、范围和调用频率保留在机器人 mdp/events.py。Black 默认开启摩擦、payload、COM、PD gain、encoder bias 和周期推力。Wolf 默认开启质量、惯性、摩擦、PD/motor strength、轮半径、控制延迟、推力、外力和初始腿姿态缩放；hip motor strength、calf backlash、wheel target bias/scale 和 wheel observation bias 默认关闭。

Wolf 腿和轮的惯量随机化均对每个刚体独立采样一个正缩放系数，三个主惯量共用该系数；腿范围 [0.9,1.1]、轮范围 [0.8,1.2]，保持主惯量比例与三角不等式。各次 reset 从 nominal 惯量采样，不累积缩放；仍使用原有 set_const_0 重计算。名义质量、惯量和资产未改。

Wolf 默认接触摩擦范围改为 [0.8,1.2]，以 nominal/play 的 1.0 为中心；额外 wheel_friction 乘数默认关闭，显式开启选项保留。训练沿用轮/机器人碰撞 geom 赋值、terrain sliding friction 置零的方式，使 MuJoCo 的接触组合读取实际采样值。Play 仍使用 nominal 摩擦，不新增随机化。

Wolf 外力使用任务本地 RootBodyFrameImpulse，复用本地锁定 MjLab 1.6.0 apply_body_impulse 的触发、到期清零和 partial reset。disturbance_force_range=[-35,35] N 表示机体局部坐标系中持续一个 physics step（5 ms）的参考力；适配器读取当前 free-joint quaternion，将新采样力旋转到世界系，缩放为 physics_dt/step_dt=0.25，并持续一个 policy step（20 ms），保持参考冲量。当前实际局部力幅值为 ±8.75 N；这不是 5 ms 力波形的精确复现。每 8 个 policy step（160 ms）触发，使用与原生 EventManager 一致的 1e-6 s 容差避免浮点计时多等一步。零指令也可受扰动；每个环境 reset 后重新计时，力只作用于 base_link 质心，不额外施加力矩。

Wolf 腿/轮最大延迟分别为 3/4 个 policy step，采样更新周期仍为 decimation。Play 移除 DR events、关闭 actor noise 与课程，但保留现有 reset 分布和 actuator delay 配置。奖励、HIM encoder、观测/动作、PD、延迟与求解器均不属于本次三项 DR 修正。

本单元来源为本地锁定的 MjLab 1.6.0 事件实现、HIMLoco-for-Go2W 的施力调用和 Isaac Gym SDK 对 immediate timestep 的定义。Local verification：tests/check_wolf_dr_fixes.py（不提交），CPU PASS / CUDA PASS。八任务 train/play/RL 完整快照仅允许上述 Wolf DR 变化；每种设备各检查 64 env ×16 bodies ×4 resets=4096 组惯量，三角不等式违规为 0，并验证摩擦范围、均值、跨 reset 无累积和 partial reset 隔离。四组姿态覆盖未 forward 的新 qpos；记录四个真实 physics substep 的施力，用全机器人动量变化核对参考冲量，误差 <3e-5 N·s；触发步为 8/16/24/32/40，下一步清零，partial reset 不影响其他环境。另完成 Wolf flat HIM、rough HIM、flat PPO、Black flat HIM 各 2 env ×8 steps、一次 PPO/HIM 更新、终止目标、checkpoint 重载和原生 JIT 导出等价。命令：MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_dr_fixes.py --device cpu --smoke；CUDA 使用 --device cuda。尚未重新训练验证步态改善，下一步用修正后的物理配置训练并对照 yaw、横移和髋部偏转。

Black run_name=flat/rough/flat_him/rough_him，Wolf run_name=wolf_flat/wolf_rough/wolf_flat_him/wolf_rough_him。新实验为 black_locomotion/wolf_locomotion，默认不加载 checkpoint。Black/Wolf num_steps_per_env 均为 64。网络均为 512/256/128，HIM encoder=128/64、latent=16、prototypes=32、temperature=3。

每个机器人单独的 runner.py 保存课程状态，HIM checkpoint 同时保存 estimator optimizer。Resume 严格检查算法、机器人和模型结构；PPO→HIM warm start 仅支持 flat，重新初始化 optimizer、iteration 和 estimator。明确的 checkpoint_path 可读取任意指定文件，输出仍进入新的 experiment。具体命令见根 README。

## 仿真和验证边界

Black flat/rough nconmax=35/128，Wolf flat/rough nconmax=64/128；njmax 均为 256，solver iterations=10、ls_iterations=20。Wolf 模板 root 高度为 flat 0.45 m、rough 10 m；运行时 root reset 由初始状态和地形决定。

项目 train 入口默认启用 EPA 数值过滤与终端 warning 控制；overflow bit 与结构化 metrics 保留。此任务未修改物理资产、碰撞几何、仿真补丁或求解器。

历史奖励验证工具 `tests/check_wolf_reward_restore.py --light` 仅适用于旧基线；当前以以下检查为准。

当前奖励验证工具为 tests/check_wolf_clean_rewards.py（不提交）。命令：MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_clean_rewards.py --device cpu --smoke；CUDA 使用 --device cuda。

- IMPLEMENTED：八任务架构；Black 8 项奖励保持；四个 Wolf 任务共用 14 项基础奖励，高度/倾斜/hip 使用平方误差，原有额外奖励函数保留。
- CONFIG CHECKED：八任务 train/play/RL 快照，只有 Wolf reward 配置与 viewer.reward_bar_max_terms 改变；惯量/短时扰动/摩擦修复、PD、动作、观测、HIM、课程、终止、仿真和 RL 参数保持。
- CPU PASS / CUDA PASS：平方误差数值、rough 中央射线和地形平移、hip/default selector、站立门控、跟踪公式、原有奖励函数 AST 与 14 条 Viser 渲染。HIM 全历史指令切片、梯度边界和导出数值检查保持通过。
- CPU PASS / CUDA PASS：Wolf flat HIM、rough HIM、flat PPO、rough PPO 与 Black flat HIM 各 2 env ×8 steps，全部 raw reward 有限、逐项 weight/dt 与总和一致；完成一次 PPO/HIM update，HIM 终止目标、严格 checkpoint 重载、原生 JIT 导出等价和 partial reset。
- TRAIN NOT VERIFIED：当前基础奖励尚未重新训练，不据此声称已经恢复旋转或横移步态。下一步从该基线训练并进行固定指令评估。

已知限制和待审查项：

1. 当前 14 项奖励基线尚无重新训练后的收敛证据。
2. 未执行全部历史 checkpoint、warm-start 与 exporter 的组合矩阵；结构兼容的文件仍需按实际来源显式加载。
3. 已完成有限 CPU 环境 rollout 与固定 checkpoint 回放，未执行长程/大规模接触或独立斜面接触实验。接触力 world frame 的依据为 native netforce 实现、配置与真实输出检查。
4. EPA 过滤不保证覆盖所有退化碰撞，初始 polytope 面和极端复杂接触仍有数值风险；持续监控 NaN 和 overflow。
5. 硬件参数和实机行为未确认；部署 safety 配置不属于训练奖励或 action contract。
