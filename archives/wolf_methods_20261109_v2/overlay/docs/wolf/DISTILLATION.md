# Wolf 可选 BC / DAgger 蒸馏路径

2026-11-09 实现。**未启动机器人训练，未认证任何旧策略教师**。现有四个 PPO/HIM 任务、默认参数和旧权重不变。此模块是项目内原创工程实现，不宣称复现 PLANC、HIM 或其他论文的完整算法。

## 已接通的路径

入口：`uv run python -m mjlab.rl.wolf_distillation.run`。

- `core.py`：`DaggerConfig`、纯部署输入 MLP 学生、有限 CPU replay、冻结教师、采集/更新/恢复、性能接管 gate。
- `contracts.py`：名义物理目标映射、同状态仿射观测桥、教师资格报告一致性校验。
- `run.py`：Wolf及Methods软/硬环境构建、纯学生验证、显式 preflight/train、checkpoint、TorchScript 导出。
- `bridge.py`：Methods checkpoint 严格重载、历史actor导出、同状态数值一致性与无特权依赖检查，输出未认证manifest。
- `tests/test_wolf_distillation.py`：合成单元与 CLI 完整编排测试。

支持 `wolf-flat`、`wolf-rough`、`wolf-flat-him`、`wolf-rough-him`，以及新注册 `wolf-methods-soft` / `wolf-methods-hard` 的当前观测契约。HIM 任务在此只提供 6 帧历史输入，**学生是历史展开 MLP，不是完整 HIM estimator**。普通输入为 53，历史输入为 318，动作输出为 16。学生接口只接收 `actor`；不读取 critic、真实线速度或 estimator GT。教师可显式读取 actor 或 critic，特权只影响标签，不进入学生输入、replay 输入或学生归一化。

教师必须是可信本地导出的、无隐式循环状态的 tensor-input TorchScript。其归一化必须随导出冻结。可接当前 PPO/HIM 的 `as_jit()` 导出；通用 RNN、裸 RSL checkpoint、未认证旧 export 不会被自动猜测加载。没有照搬外部项目代码或增加依赖。

## 数据与优化

1. 从上一次 `step/reset` 返回值取得同一份观测，不再重新 `observation_manager.compute()`，避免历史或噪声被推进两次。
2. 教师和学生均读取该状态的快照，教师标注后才执行一步。
3. 逐环境 Bernoulli 选择教师或学生的完整动作，而不是平均两套动作。
4. replay 只保存部署输入和对应教师动作。跨迭代聚合，满后 FIFO 替换，是**有界 DAgger 变体**，不是无限保存全部历史数据。
5. 默认使用 PyTorch `huber_loss(delta=1)`，可切换 MSE。不是一般 delta 下等价的 SmoothL1。损失作用于已转换到当前 Wolf raw-action 坐标的标签，delta 需按这一尺度消融。
6. 学生 optimizer 不包含教师参数；教师独立复制、`eval()`、`requires_grad=False`，包括归一化 buffers。教师没有在线统计更新。学生目前无自适应归一化，不会从验证数据估计统计。

### 默认参数与消融

JSON `--config` 可覆盖 `DaggerConfig` 字段，例如：

```json
{
  "loss": "huber",
  "huber_delta": 1.0,
  "dagger_enabled": true,
  "max_iterations": 2000,
  "replay_capacity": 100000,
  "rollout_steps": 32,
  "batch_size": 1024,
  "updates_per_iteration": 4,
  "eval_interval": 25,
  "takeover_step": 0.1,
  "minimum_tracking_ratio": 0.9,
  "maximum_failure_rate": 0.05,
  "minimum_support_ratio": 0.9,
  "minimum_teacher_support": 0.5,
  "required_passes": 2,
  "seed": 101
}
```

- `loss="mse"`：关 Huber。
- `dagger_enabled=false`：始终教师控制、同状态在线 BC，不允许 stale gate 状态使学生接管。
- 默认 64 个训练环境、16 个验证环境，可按显存调整。CPU replay 约 `(input_dim+16)*100000*4` 字节，不含 checkpoint/采样开销。
- `train_iter` 是一次 rollout 聚合再执行配置的若干优化步骤。单独记录 `optimizer_steps` 和 `transitions`。**2000 次蒸馏迭代不等于 2000 次 PPO 迭代，也不保证等样本或等算力。**

## 性能驱动接管

初始学生控制概率 0。每 25 迭代，在独立验证环境、固定验证 seed 上，只用学生完整跑完每个环境的第一个 episode。验证集绝不写入 replay。教师在相同验证条件上单独跑参考 episode。

跟踪指标为 `0.5*(exp(-vx_error²/0.25)+exp(-yaw_error²/0.25))` 的 episode 时间均值，再跨环境平均。真实速度只用于评价。异常终止与 timeout 分开；最后一步同时失败和 timeout 仍记失败。

默认学生跟踪达到教师的 90%、异常终止率不超过 5%，连续两次满足才增加 0.1；不满足则减少 0.1。教师参考也须满足最低跟踪 0.1、异常终止率不超过 5%。这些是**可调工程阈值，不是已验证验收标准或安全保证**，均值不能替代左右转向、饱和、稀疏落脚、视野外失败等分场景指标。

蒸馏入口固定初始 command/terrain 课程阶段（`cfg.curriculum={}`），仅修改深拷贝/新建的本次配置。否则训练域扩展，而验证仍停留在容易域，会导致错误放权。此阶段不实现学生 PPO 微调；若要扩课程或改变任务域，须重新评价教师和学生，不可直接沿用资格报告。

验证 seed 必须与训练 seed 不同。默认验证 `900101`，它是调参与接管的验证集，**不是最终测试集**。最终独立多 seed、多场景测试仍需另外完成。

## 新旧 Wolf 控制契约

当前名义：FL/FR/RL/RR，每腿 hip/thigh/calf/foot；腿 `q_target=offset+scale*a`，hip `.125`、thigh/calf `.25`；轮 `dq_target=5*forward_sign*a`；腿 PD `80/2`、轮 `0/.6`，50 Hz。

`TargetSpaceAdapter` 显式按关节名重排，再做：

```
physical_target = source_offset + source_scale * source_raw_action
current_raw_action = (physical_target - current_offset) / current_scale
```

它处理动作单位、顺序、符号和默认姿态。拒绝不同控制周期、关节集合或 position/velocity 类型。**不会暗中裁剪，也不会把 PD 变化、延迟、DR、friction、限幅或真实动力学当作等价。** 旧 `.25/10` 与新 `.125/.25/5` 可以算出目标映射，但这不代表 `80/3,0/1` 与 `80/2,0/.6` 闭环行为相同。

观测桥是显式逐帧 `old_frame = current_frame @ matrix.T + bias`；外部历史可选 newest/oldest-first。旧 `last_action`、默认姿态差、历史 reset、joint 顺序、normalizer 均须单独核验。不能只填写 identity、让 shape 匹配就宣称兼容。非线性、隐式循环、时间重采样或超出仿射表达能力的旧观察需要另写桥，不会自动开放。

当前不实现按命令区域/按动作通道选择性混用旧教师；资格覆盖的是整个指定目标域。未通过该域资格的旧策略继续仅作诊断，不能因为直行看起来可用就标注整个转向域。

## 教师资格入口（默认 fail-closed）

先查看实际当前契约，不创建教师资格：

```sh
uv run python -m mjlab.rl.wolf_distillation.run --task wolf-flat --describe-contract
```

`--manifest` JSON 必须包含：

- `schema_version: 1`、精确 `teacher_sha256`、`format: "torchscript_tensor"`。
- 完整 `source_control`，由实际源资料确认；`target_task`、`target_contract_sha256`、`target_runtime_sha256` 与当前描述匹配。
- `input_group: "actor"` 或 `"critic"`，`history_length`、`history_order`（`newest_first` / `oldest_first`）。critic 当前只支持单帧。
- `observation_matrix`、`observation_bias`；`observation_contract`、`normalization_contract`、`history_contract` 的可审计说明。
- `curriculum_policy: "frozen_initial_config"`。
- 真正完成资格检查后才可记录 `qualification: "qualified"`。未训练 fixture 不具备此资格。
- `evidence` 中 `same_state_observation`、`closed_loop_target`；源/目标控制契约不同时另需 `cross_contract_mapping`。每项为 `{"path": "/absolute/report.json", "sha256": "..."}`。

每份报告 JSON 必须绑定 `passed: true`、自身 `validation_kind`、精确 `teacher_sha256`、目标控制/runtime/task、`curriculum_policy` 及 `adapter_sha256`。后者用 `contracts.adapter_digest(manifest)` 计算，绑定实际动作、观察、normalizer 和历史桥。报告还应包含真实测试的场景、seed、指标和判定依据；校验器只核对报告一致性，**不能验证一个人编造的通过声明是否真实**。

没有提供虚假的 qualified 示例或让随机 fixture 获得资格。修改桥、权重、目标契约、Wolf 环境源码时旧报告会被拒绝。runtime 指纹覆盖 Wolf 的 Python/XML/JSON 文件，不是整个仓库/驱动/依赖的内容寻址快照；正式实验还必须记录软件/驱动/资产整体版本。

## 执行与产物

```sh
# 默认仅4步preflight，学生optimizer不更新。
uv run python -m mjlab.rl.wolf_distillation.run \
  --task wolf-flat --teacher /path/teacher_actor.pt --manifest /path/teacher.json

# 只有今后明确要训练时才使用；本次没有执行。
uv run python -m mjlab.rl.wolf_distillation.run \
  --train --task wolf-flat --teacher /path/teacher_actor.pt \
  --manifest /path/teacher.json --config /path/distill.json \
  --output logs/wolf_distillation/run_seed101
```

正式 `--train` 拒绝 `<2000 max_iterations`，输出目录必须是新目录，防止覆盖旧试验。checkpoint 保存学生、optimizer、replay、采样 RNG、接管状态、独立计数、教师 manifest/hash 与目标控制契约。`--resume` 必须匹配原 config/teacher/task/runtime；也要指定新 output 目录。**恢复 optimizer/replay/RNG，但仿真 episode 重新开始，不宣称逐步精确恢复 simulator。**

最终 `student_actor.pt` 是只接受 `[B,53]` 或 `[B,318]` 的部署 MLP；318 为帧优先、最新到最旧。checkpoint 与 `teacher_manifest.json` 记录来源。没有自动上传、发布、改已有 play 工具或实机配置。

## 本次验证边界

- 合成测试运行真实 loss/backward/optimizer、冻结、replay、恢复与导出。CLI 编排测试从**人工构造的 iteration=1999 合成状态**走一次更新到终点，只为覆盖 final checkpoint/export 分支，绝非完成2000轮训练。
- 实际隔离 Wolf 环境：四任务各教师4步＋学生4步；flat 另各1个完整 episode。显式禁用所有 backward 与 optimizer.step，教师/学生 state_dict 前后不变。fixture 明确未训练，因此这些结果不能证明教师质量、稳定走行或收益。
- 详细证据：`${WOLF_EVIDENCE_ROOT}/distillation_implementation_20261109/`。
- 全仓测试、训练收敛、真实旧教师资格、实际部署性能均未验证。此路径与其他新研究模块的联合训练尚需集成测试。


## Methods → DAgger 安全阶段桥（集成补充）

这是一条**分阶段**路径：MethodsPPO 教师训练 → 冻结完整动作actor → 独立 MLP 学生 BC/DAgger。不是同一个 optimizer 联合执行 MethodsPPO 与 DAgger。Methods 的 `terrain_teacher_*` 是训练期辅助表征，不是输出动作的专家。这里导出的是 Methods actor 的 HIM encoder、历史特征与 residual 动作网络，包含它已经学到的表征；导出推理仍只依赖318维本体历史。

```sh
# 只导出已有、可信Methods checkpoint并检查软件桥，不训练。
uv run python -m mjlab.rl.wolf_distillation.bridge \
  --task wolf-methods-soft --checkpoint /path/methods_checkpoint.pt \
  --output /path/fresh_export_directory
```

需要匹配消融checkpoint时，显式使用 `--no-him`、`--no-distillation`、`--no-prediction` 或 `--no-dual-critic`；不猜测不匹配的架构。使用注册的 `WolfMethodsOnPolicyRunner` 严格重载，检查 methods/dual-critic 保存契约。导出助手还要求传入actor的每个state tensor与源checkpoint逐一相等；拒绝仅维度相同却换了权重的模型。

桥会检查目标 soft/hard 地形碰撞配置、命令域、53维观察项/scale和6帧历史契约。Mjlab 构建地形时会在原配置中展开子地形size，比较前显式归一到实际父地形size，避免把正常构建误认为换域。soft/hard不能因动作尺度相同就混为一套资格。

在同一份当前状态观测上检查原actor、script export、`FrozenTeacher`适配输出一致；把所有非actor观察置为NaN后原actor动作必须不变。不会更新源actor的权重、模式、normalizer或梯度设置。仅这些检查不能证明会走路或有教学价值。

产物为 `methods_actor.pt`、`export_parity.json`、`teacher_manifest_UNQUALIFIED.json`。**导出一律未认证**。完整资格通过之后，才能使用常规蒸馏入口的 `--task wolf-methods-soft` 或 `wolf-methods-hard` 加载该export。manifest绑定确切目标阶段、checkpoint/export hash、控制、观察和runtime；同状态配对和独立纯学生验证仍由原DAgger路径执行，特权地形扫描不会写入学生输入/replay。

旧checkpoint一般不包含可核验的训练任务ID，因此manifest明确记 `source_training_task: not_verified_by_checkpoint`，保留源checkpoint hash，并不把当前用于导出的目标场景伪称为原训练场景。必须从原训练记录及目标闭环验证补全资格；特别是soft→hard迁移不自动获得hard教师资格。

部分旧教师的“有效区域mask”尚未实现：只丢弃无效标签却仍让不合格教师驱动环境不安全，直接切换随机学生也没有已验证fallback。需要明确有效命令/状态域、同状态标签覆盖与可用fallback之后再接，不用空mask宣称已完成选择性旧教师示教。


### Methods 支撑覆盖评价补充

Methods 专项接管不能只看速度。独立验证现在额外读取已配置轮子支撑ray与接触力，以原 `wheel_support_coverage` 几何规则计算加载轮的支撑覆盖率；无加载轮帧计0而非满分。指标按每个完整episode时间均值再跨环境平均，不进入学生输入或replay，也不重新执行reward函数或改变训练奖励。

默认教师覆盖至少0.5，学生至少达到教师覆盖的90%，并同时满足跟踪和失败率门槛，才允许增加接管比例。它仍是可调工程阈值，不是落脚安全证明。缺失/不一致的支撑评价契约会拒绝放权。

最终软件验证：27项本包/阶段桥测试、Ruff、ty通过。最新Methods v2签名的软/硬任务使用注册runner，各自严格保存/重载未训练checkpoint、export往返、教师4步和学生4步同状态收集，再各跑教师与学生一个完整episode。源actor、冻结teacher、student的全部state hash不变；所有backward/optimizer均禁止。HIM开/关均有export→冻结teacher→同状态标签的单元回归。

两种场景中未训练策略也可能取得较高静态支撑覆盖，这正说明**软件桥和泛化资格不同**：上述单episode结果绝不能作为教师验收、通过石块/平衡木、学到转向或实机安全证据。新导出manifest始终保持未认证，公开训练入口仍拒绝加载它。
