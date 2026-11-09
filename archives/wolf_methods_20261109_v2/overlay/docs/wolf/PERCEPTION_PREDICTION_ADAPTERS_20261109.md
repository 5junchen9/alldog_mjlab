# Wolf 感知与预测适配（2026-11-09）

本交付是 mjlab/Wolf 的可关闭实验分支，不是 GLAD 或 WM-LOCO 论文复现，也不代表已经学会地形通过。没有启动训练。

## 实际接线

`methods_cfg.wolf_methods_env_cfg` 增加训练专用的两组 ray 观测；`wolf_methods_runner_cfg` 使用现有 `WolfHimOnPolicyRunner`，以 `PerceptionHIMPolicy` + `MethodsPPO` 构造真实 RSL-RL 训练路径。它与独立实现的 `DualRewardPPOMixin` 配合，不修改原有四个基线任务。

- 局部：1.6×1.0m、0.1m 间距，187条 ray。
- 全局：3.2×2.0m、0.4m 间距，54条 ray。
- 每条 ray 输入 x/y、归一化相对高度、有效位。观测组总计 748+216 维，严格不进入 actor 原始观测或部署输入。
- 两套扫描器均读取目标地形的 group 0，soft 阶段不会误读 group 3 的平面物理地板。
- 教师采用全局 attention pooling 与本体历史条件化的局部 top-8 attention。局部排序使用 query/key 点积，因此随历史变化，不是可被 softmax 抵消的共同加法项。
- 教师用地形重建损失学习；历史学生用 stop-gradient 教师 latent 的 MSE 蒸馏。部署 actor 的 residual 分支实际读取该学生 latent，零初始化 residual 保留初始 HIM 动作。
- 第二个历史编码器用已执行 action 预测下一步非 command、非 previous-action 的34维本体量，SmoothL1 损失训练。actor 同时读取此预测 latent。
- PPO 仅更新原 actor、residual 和 critic；新增辅助编码器在整次 PPO update 后单独更新，避免同次 PPO epochs 内漂移。
- 原有 HIM 的估计器训练与 normalizer 修复由其独立实现继续负责。

## 不能声称的能力

**学生是盲本体历史学生，不是视觉学生。** 当前没有相机、深度图、LiDAR 端到端前端或视觉学生蒸馏。全局/局部教师只生成训练目标，不是一个已训练的特权动作专家，也没有 GLAD actor/critic 的完整结构。盲学生不能凭空识别视野外或尚未接触的缺口。

**预测是一步、前馈、本体预测，不是 WM-LOCO RSSM。** 没有深度重建、循环随机状态、长时序损失、AMP、MoE、想象轨迹或规划。新方法收益必须后续多seed评估，不能从合成测试或有限损失推出。

## 终止、断点与部署

MethodsPPO 要求 done ID 与 pre-reset terminal ID 完全一一对应，拒绝缺标签、重复/越界/错误 ID、错误 shape 或非有限值。无 done 时丢弃陈旧 terminal payload。预测目标沿用 HIM pre-reset recorder，避免把 reset 后下一回合首帧当成上回合 successor。

训练 checkpoint 包含新增 encoder、教师、predictor、actor residual、辅助 optimizer 以及方法签名。不同开关/latent/目标维度/损失配置的恢复会先拒绝，再加载任何权重。原 PPO/HIM checkpoint 不能直接当 methods full resume。

JIT 只导出原 HIM actor、本体学生编码器与 residual；不包含训练教师、地形 decoder、predictor 或真实速度标签。输入仍是 canonical newest→oldest 的 `[B,318]` 本体历史。

## 开关与资源

Python 配置工厂是完整消融边界：

- `wolf_methods_env_cfg(distillation=False)` 删除额外教师观测/全局扫描器。
- `wolf_methods_runner_cfg(distillation=False)` 删除地形学生和教师网络。
- `wolf_methods_runner_cfg(prediction=False)` 删除预测编码器/预测头。
- `dual_critic=False` 同时选择单值 critic 和单reward PPO返回路径。
- `foot_placement=False` 与 `clf=False` 移除相应环境奖励。
- 不使用落脚分量时，也必须将 dual critic 关闭；不能让双分支期待不存在的奖励。

CLI actor 开关只能关网络，不能自动删掉环境传感器；完整显存消融需要上述匹配的配置工厂，不应宣称单改一个 CLI bool 就移除了全部开销。

新专项默认256环境、32步 rollout、辅助最多2048样本、每批256。这个配置是8GB设备的保守起点，不是已完成8GB训练容量验证。2环境 smoke 的显存数字不能外推至训练峰值。

专项地形只给前进0.2–0.8m/s，vy/yaw=0，禁自动速度扩展。它只评估前行落脚，不覆盖原任务的转向目标。原 flat/rough 全向命令未改，必须保留全向回归。

## 验证

`tests/test_wolf_perception_models.py` 覆盖：四种感知/预测开关、all-off 与原 HIM 输出一致、特权输入污染不影响actor、独立梯度/优化器参数边界、全局与局部 attention 条件性、miss/nonfinite mask、终止标签覆盖、配置隔离，以及实际 runner 构造→合成 rollout→单/双 critic 返回→update hook→checkpoint。update smoke 将所有 Adam.step 替换为 no-op，并禁 HIM 估计器更新，不执行训练。

`scripts/check_wolf_methods.py --stage soft|hard` 执行小规模真实仿真+runner采样+强制timeout+有限辅助损失+JIT一致性，绝不调用 learn 或优化器。随机策略走几步不是通过率测试。

针对性 ruff/ty 检查已执行。pyright 在当前系统 Node v12.22.9 下解析工具自身的现代 JS 失败，未产生可靠代码诊断；不标记为通过。完整仓库测试和训练稳定性不在以上小规模验证范围。

## 参考边界

- GLAD: https://arxiv.org/html/2606.00637v2
- WM-LOCO: https://arxiv.org/html/2609.02542v1
- HIM: https://arxiv.org/html/2312.11460v3

新增实现独立编写，未复制未确认许可的论文代码。

## 同任务 HIM 机制消融补充

`WolfMethodsModelCfg.use_him_estimator` 默认为 True；配置工厂接受
`wolf_methods_runner_cfg(use_him_estimator=False)`。关闭后：

- 原 actor 基础通路只读取当前53维本体帧；估计速度/隐变量19维固定为零，以保留相同 actor MLP 结构和预算。
- HIM estimator 不执行 encode 或 update，所有参数 requires_grad=False，PPO参数集合不包含它，梯度为 None。
- checkpoint 的 methods_signature 升为 version 2 并绑定 HIM 开关，拒绝混用开关恢复。
- JIT 不包含 HIM encoder 参数。为保持通用方法训练接口，训练态仍分配冻结估计器及历史/目标 buffer，因此此开关是机制消融，不是优化显存的纯PPO实现。
- 若 GLAD/预测仍开启，对应 residual 仍可使用历史，这与 HIM 关闭并不矛盾。

仅关闭 distillation/prediction 的旧 all-off 测试是“新增两个辅助模块关闭”，不是所有方法关闭。
同任务基础对照需同时关闭 HIM、GLAD、预测、dual critic、落脚奖励和CLF，保留相同地形/命令/训练预算；soft/hard 地形阶段也是独立实验因素。

新回归覆盖 HIM×GLAD×预测 的8种导出组合和 HIM×辅助开关×单/双critic 的8种无优化器runner组合，验证关闭后无编码器调用/训练/梯度且旧帧不影响当前帧基础actor。
另注入“loss仍有限但梯度为NaN”的情况，确认 `clip_grad_norm_(error_if_nonfinite=True)` 在 optimizer.step 前拒绝；参数不改变。
