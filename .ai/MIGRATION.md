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
| wheel_lateral_slip_l2 | -2.0 |
| wheel_longitudinal_slip_l2 | -0.5 |
| turn_hip_default | -3.0 |
| feet_air_time_ang_z | 50.0 |
| feet_slide_ang_z | -2.0 |
| rotation_gait_status | 2.0 |
| rotation_gait_symmetry | 15.0 |

Wolf flat 注册上述 22 项，rough 另加 wheel_force_lift，权重 0.5，共 23 项。

当前工作区有两项物理滑移奖励权重=0 的用户覆盖，继续保留为未提交修改；上表 -2/-0.5 描述 Git 提交基线。转向迁移时保留的用户 hip 80/3 已由当前 M20 对照控制基线 80/2 替换。本地验证使用含滑移权重覆盖的实际工作区。

Wolf 高度目标 0.40 m，orientation 使用 projected gravity XY 的 L1 和。Hip default 为四个 hip 位置误差 L1，乘以 `clamp(1 - 0.2×min(abs(vy_cmd)/0.5,1) - 0.35×min(abs(yaw_cmd)/1.0,1), 0.5, 1)`；横移和转向指令下放宽回中约束。

Stand_still 在 norm(command_xy)<0.1 m/s 且 abs(command_yaw)<0.1 rad/s 时计算 12 腿关节位置误差 L1。Run_still 在 abs(command_x)>0.1 m/s、abs(command_y)<0.1 m/s 且 abs(command_yaw)<0.15 rad/s 时启用相同 L1。两项均排除轮关节。

Leg/wheel action rate 分别选择 12/4 个 raw action 通道计算相邻帧平方差；leg/wheel torques 分别选择 12/4 个 actuator 的实际力矩计算平方和，选择集合互不重叠。Dof_pos_limits 仅惩罚 12 个腿关节超出 soft position limit 的距离，soft limit factor=0.9。

Wolf flat 高度使用 world z；rough 使用中央 7×5=35 条 terrain ray 的局部离地高度均值。两者的高度误差均为 `abs(height - 0.40)`，权重 -10.0；沿用 `base_height_l2_flat` / `base_height_l2_terrain` 函数名。两种地形共用上述 17 项权重和高度目标。Rough 的 wheel_force_lift 显式选择四个轮碰撞 geom 的 world-frame net force：水平力范数 ≥15 N 时，累加对应轮 body 的向上竖直速度（负速度裁为 0）。

## Wolf 轮地滑移奖励与诊断

Wolf 四任务的 train/play 均注册 `wheel_slip_contact`，按 FL → FR → RL → RR 显式匹配四个轮碰撞 geom；原生 maxforce reduction 每轮保留一个最强接触，读取 found、contact-frame force 和 world-frame pos/normal/tangent。Rough 的原有 netforce 传感器继续独立服务 wheel_force_lift。

`mdp/metrics.py` 使用 `v_contact = v_link + omega_wheel × (p_contact - p_link)` 计算轮上当前接触材料点相对静止地面的速度。omega 为轮体总角速度，包含机身、腿和轮关节运动。轮轴来自实际轮体姿态下的 MJCF local Z；与接触法线构造滚动和侧向基。仅统计 found>0、法向力绝对值>1 N 且滚动方向有效的接触；离地和未承重轮输出零。

`wheel_lateral_slip_l2` / `wheel_longitudinal_slip_l2` 分别为四轮侧向/纵向滑移速度平方和，单位 m²/s²。四任务 train/play 同时将这两个 raw 函数注册为 RewardTerm 和 MetricsTerm：奖励初始权重分别为 -2.0、-0.5，原生 RewardManager 再乘 policy dt；MetricsManager 独立记录未加权、未乘 dt 的 episode policy-step 均值。接触门控适用于所有指令，无 yaw 或预设步态门控。Actor/critic/history、action、PD 与 checkpoint contract 保持。原生 reward/step metrics 在最后一次 forward 前计算，派生量滞后一个 physics substep；本地回放 trace 采样于 forward 后。

验证：`tests/check_wolf_wheel_slip.py --device cpu/cuda` 的物理数值用例均 PASS，覆盖纯滚动、贴地空转、横向滑移、接触门控、总角速度、镜像轮轴与坐标变换；八任务 train/play/RL 快照对比确认只有 Wolf 新增上述 sensor/metrics。CUDA 回放 `2026-11-01_15-57-14_wolf_flat_him/model_1000.pt`，8 env ×600 policy steps，hip PD 按训练值设为 90/3.3，其余使用当前 play 配置并保留 actuator delay；全程状态与输出有限。接触点速度与 C MuJoCo Jacobian 独立比较，最大误差 5.7e-6 m/s 以内。Wolf rough HIM 另以缩小地形的 1 env reset+2 步验证 native metrics 有限。

回放去除前 2 s 后，vx=0.5 的侧滑 RMS 为 0.0029 m/s，yaw=+1/-1 时为 0.1178/0.1296 m/s，yaw=+2/-2 时为 0.2068/0.1970 m/s；RMS 以承重轮样本为分母。+1 yaw 下四髋平均绝对偏角约 10.6°/12.1°/13.6°/15.8°，请求目标约 4.3°/0.7°/3.1°/6.7°，存在明显目标跟踪误差与持续力矩。小 yaw 的方向差异明显，-0.5 指令下平均实际 yaw 仅 -0.021 rad/s。结果来自单 seed、每指令一个环境的短回放，尚未验证训练改善；最强单接触也不能覆盖整块轮胎接触区域，零滑移不能排除髋部静态受力。

初始奖励权重按上述回放量级选取：旧策略在 ±1 yaw 下两项加权合计约 -0.12～-0.13，直线 vx=0.5 下约 -0.00007（均未乘 dt）。`tests/check_wolf_wheel_slip.py --device cpu/cuda` 已验证四任务 train/play 的函数、权重、selector 与 metric 一致，除新增两项奖励外其余既有配置保持；CPU PASS / CUDA PASS。`tests/check_wolf_slip_reward_runtime.py` 完成 CUDA flat HIM 2 env ×128 步旧 checkpoint 推理与 rough HIM 1 env ×8 步，确认新增 raw cost 有限、非负，负权重、dt 和总奖励累计一致；未执行 PPO/HIM update 或训练消融。

本地工具 `tests/trace_wolf_wheel_slip.py` 输出 trace.npz、summary.json；结果位于 `tests/wolf_wheel_slip_results/`，不提交。新增 M20 转向行为的训练评估见下节，当前不声称已学会抬腿转向。

## Wolf M20 转向行为与电机对比

来源为 DeepRoboticsLab/rl_training `9c0e710421b165a444b2c73e5a9d65ee3fafa299` 的 M20 rough 配置、velocity/mdp/{commands,rewards,curriculums}.py 和 assets/deeprobotics.py；执行器语义核对其 README 指定的 Isaac Lab v2.3.2 `actuator_pd.py`。许可证保留在根目录 `THIRD_PARTY_NOTICES.md`。迁移使用当前安装的 MjLab 1.6.0 原生 CommandTerm、ContactSensor、ManagerTermBase 和 RewardManager，未改依赖源码。

四个 Wolf 任务的 train/play 共用以下行为：

- `WolfVelocityCommand` 继承原生 UniformVelocityCommand，仅补足互斥特殊指令采样：20% 站立、2% 纯横移、2% 纯前后、20% 纯 yaw、56% 混合。特殊指令沿用初始化范围，不随 vx 课程扩展；原生 reset、GUI 和 command metrics 保留。Wolf 继续直接采样 body-frame yaw 速度，不引入官方 heading 目标和小 XY 指令清零。
- `wheel_slip_contact.track_air_time=True`，复用原生每 physics step 接触/腾空计时，四轮显式顺序仍为 FL/FR/RL/RR。
- `turn_hip_default`：norm(cmd_xy)<0.1 且 abs(cmd_yaw)>0.1 时，惩罚四髋相对默认值的 L2 范数，权重 -3；叠加既有弱 hip_default。Thigh/calf 继续可在 yaw 下自由屈伸。官方 hip-x 在所有运动模式下回中，Wolf 将新增强约束限定于纯 yaw。
- `feet_air_time_ang_z`：有 abs(yaw)>0.1 时，在首次落地累加 `(last_air_time-0.2)*cleared_ground`，权重 +50；高度取该轮离开上次接触后到达的最大 world z，高度差阈值沿用官方 0。首次下落和环境 reset 不继承上一 episode 的离地高度状态。
- `feet_slide_ang_z`：纯 yaw 门控同 turn_hip_default。当前最强接触力范数>1 N 的轮，body-link 速度减 root-link 平移，转到 body frame 后累加 XY 范数，权重 -2；官方使用 net force 的短历史门控。此项沿用官方步态约束含义，不扣轮自转；既有接触材料点物理滑移函数、诊断及用户权重保持。
- `rotation_gait_status`：norm(cmd_xy)<0.5 且 abs(yaw)>0.05 时，奖励 FL+RR 支撑且另一对平均轮心高出 >0.05 m，或反向组合，权重 +2。它本身不强制时间交替。
- `rotation_gait_symmetry`：上述门控下，5 s 滑动窗口内两组对角轮平均接触占空比接近 0.5，std=0.2，权重 +15。沿用官方窗口和核函数；修正为每环境独立有效样本计数与 warmup，空缓冲不视作腾空，partial reset 只清对应环境。这是接触奖励，不是 policy mirror loss。

有意区别于官方：新增转向奖励直接按配置生效，不乘其全局 `gait_level`（该值依赖地形课程，flat 关闭地形课程时会保持初始零）。保留 Wolf 的基础 tracking、高度绝对值 -10、姿态、动作平滑、终止、DR、vx 课程和 HIM；不迁移整套 M20 rough 参数、joint_mirror 或电机限幅。Actor/critic/history、16 通道 action 顺序、默认姿态、effort 与 checkpoint shape 保持。PD 和动作尺度采用下述 M20 对照控制基线；训练分布、优化目标和控制响应的改善仍需训练评估。

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

本次控制 unit 同时采用腿部与轮部设置，后续再做消融。仅修改 `robots/wolf/wolf_constants.py` 的名义 PD、`tasks/wolf/env_cfg.py` 的原生按名称 action scale；calf backlash 与 wheel target bias/scale 可选 action 分支共用相同尺度。Effort 60/17、armature、delay、DR 比例和全部奖励均保持。MjLab 1.6.0 原生支持 scale dict，无需自定义 action 或改依赖。

当前 policy/deployment 顺序仍是 FL/FR/RL/RR，每腿 hip/thigh/calf/foot。新策略部署需要：

```text
kp:           [80, 80, 80, 0] × 4
kd:           [2, 2, 2, 0.6] × 4
action_scale: [0.125, 0.25, 0.25,  5,
               0.125, 0.25, 0.25, -5,
               0.125, 0.25, 0.25,  5,
               0.125, 0.25, 0.25, -5]
```

默认姿态与轮速观测 forward sign 保持。部署侧实际路径为 `../Sim_Deploy/quadruped_control`，其现有 Wolf test policy 腿 PD=80/3、轮 PD=0/1、hip scale=0.25、wheel scale=10，与新训练控制 contract 不同；本次未修改部署仓库或已有 policy.pt。新策略使用独立配置同步上述参数，历史策略回放使用各自保存的训练配置。PPO/HIM 输入输出 shape 仍为 53→16、318→16，旧 checkpoint 虽能按结构加载，动作与闭环语义已改变；本次不执行历史 checkpoint 迁移或新训练。

控制 unit 的本地验证工具为 `tests/check_wolf_m20_control.py`（不提交）。八任务 train/play/RL 的逐字段快照仅允许 Wolf damping 和 action scale 发生变化，其余含奖励、DR、delay、effort、观测与 RL 配置完全一致。CPU 原生 flat HIM、rough HIM、flat PPO（启用可选 calf backlash 与 wheel target bias/scale）各 2 env，检查正负动作的实际目标、默认 offset、四腿顺序与轮符号、初始化后执行器的 PD 输出和 60/17 限幅，再完成 8 步非零动作及 partial reset，状态和观测均有限：CPU PASS。CUDA SKIPPED（无可用设备）。命令为 `MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_m20_control.py --device cpu --smoke`；同工具 `--device cuda` 记录跳过。未执行训练或消融，下一步使用这组完整参数训练。

转向奖励 unit（控制参数调整前）的本地验证工具 `tests/check_wolf_m20_turning.py`（不提交）覆盖八任务 train/play/RL 配置差分、100000 次互斥采样与正负 yaw、课程范围隔离、纯 yaw 门控、落地事件与短腾空负值、抬升和接触组合、5 s 窗口语义、warmup 与 partial reset。CPU 数值检查 PASS；CPU 原生 flat/rough HIM 各 2 env，reset+8 步+partial reset，22/23 项 RewardManager 加权/dt/总和一致，观测和 reward 有限。CUDA SKIPPED：当前无可用设备。未运行 PPO/HIM update 或长程训练。

验证命令：`MPLCONFIGDIR=/tmp/wolf-m20-mpl WARP_CACHE_PATH=/tmp/wolf-m20-warp .venv/bin/python tests/check_wolf_m20_turning.py --device cpu --smoke`；另运行同脚本 `--device cuda`（SKIPPED）与 `tests/check_wolf_height_l1.py`（CPU PASS）。旧 `tests/check_wolf_posture_rewards.py` 仍引用已移除的 `tasks.velocity.wolf`，执行因旧路径 FAIL；本次未重写该历史工具。

下一 behavior unit：先用当前完整转向奖励与 M20 对照 PD/尺度训练，固定命令检查 yaw MAE、承重横纵滑移、hip target/actual 偏差与力矩、轮离地高度和接触时序；后续再对奖励、PD/尺度做消融，并单独评估延迟与 RS03/RS01 速度-转矩包络。

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

- IMPLEMENTED：机器人独立任务架构、八任务注册、Black 8 项及 Wolf flat 22 / rough 23 项奖励、Wolf M20 转向采样/奖励、课程版本、runner 与文档。
- STATIC CHECKED：Python 语法、imports、八任务注册、精确奖励表和无内部 tasks.velocity 引用。
- CONFIG CHECKED：Wolf 四任务 train/play 的奖励权重、参数和 selector；八任务 train/play 的控制、观测、DR、终止、课程、仿真、机器人及 RL 配置与修改前快照一致。课程、runner 和 RL 源码未变。
- CPU PASS / CUDA PASS：姿态 L1、hip 指令缩放、stand_still 的 XY+yaw 门控、run_still 的直线门控、腿轮动作变化率与力矩分组、抬轮力阈值及向上速度截断。
- CUDA LIGHT TEST PASS / SIM VERIFIED limited：此前奖励恢复验证中，wolf-flat 和 wolf-rough-him 各 1 env，reset 和 2 步 rollout，原有全部 15/16 项奖励输出为有限的一维 Tensor。实际力矩 selector 为互不重叠的 12/4 个 actuator；轮接触匹配四个指定 geom，force shape=[1,4,3]。Rough 验证将地形行数缩为 2。新增两项滑移奖励的验证见上节。
- TRAIN VERIFIED：NOT RUN。没有执行 PPO/HIM update、长程训练、收敛验证或实机测试。

已知限制和待审查项：

1. 新奖励基线尚无训练收敛证据；需要独立 Code Review 后再安排训练评估。
2. 未执行全部历史 checkpoint、warm-start 与 exporter 的组合矩阵；结构兼容的文件仍需按实际来源显式加载。
3. 未执行长程/大规模接触场景或 CPU 环境 rollout。接触力 world frame 的依据为 native netforce 实现、配置与真实输出检查，未新增独立斜面接触实验。
4. EPA 过滤不保证覆盖所有退化碰撞，初始 polytope 面和极端复杂接触仍有数值风险；持续监控 NaN 和 overflow。
5. 硬件参数和实机行为未确认；部署 safety 配置不属于训练奖励或 action contract。
