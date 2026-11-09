# Wolf 方法接入与消融接口

本页说明可执行接口，不是训练成绩。未启动正式训练，未认证任何旧权重教师。现有四个任务与旧模型目录保留。

## 入口与训练预算

独立注册任务 `wolf-methods-soft`、`wolf-methods-hard`。它们是前进0.2–0.8m/s、vy/yaw=0的支撑地形专项，不能替代原全向任务的转向与停稳验收。

`WolfMethodsOnPolicyRunner` 检查网络所需教师观测和双critic所需reward通道，并拒绝少于2000次原生PPO迭代的正式learn调用。原四任务注册未改。默认新任务256环境、32步rollout仅为资源起点，未验证长训练容量。

推荐统一配置入口：

```bash
# 默认只打印计划，绝不训练。
bash ${WOLF_EVIDENCE_ROOT}/run_python.sh scripts/train_wolf_methods.py   --stage soft --num-envs 128 --rollout-steps 100 --seed 101
```

只有显式加 `--train` 才调用训练入口；正式迭代数默认2000。`--checkpoint` 是显式同方法恢复，不能拿不兼容权重猜测加载。只使用tensorboard，默认不上传模型。当前训练入口不自动选择best、安排soft到hard晋升或认证动作教师。

单独开关：`--no-him`、`--no-terrain-distillation`、`--no-prediction`、`--no-dual-critic`、`--no-foot-placement`、`--no-clf`。关闭落脚reward会同时关闭对应双critic，避免通道缺失。完整开关通过环境与网络配对工厂实现，而不是只改变actor的CLI字段。原生train命令仍可使用两任务，但消融推荐上述配对入口。

`--no-him` 保留相同53+19维actor架构，将19维HIM特征置零，冻结且不调用估计器推理/更新；这用于机制消融，不应宣传训练态显存中的估计器参数和历史buffer已全部移除。部署导出确实不包含HIM encoder。HIM、感知/预测开关及dual合同写入checkpoint签名；签名不匹配拒绝恢复。

## 真实链路与论文边界

| 机制 | 实际接入 | 必须保留的边界 |
|---|---|---|
| HIM | 历史速度估计+prototype/Sinkhorn损失，独立optimizer，修复critic normalizer | 不是VAE；仅本体历史不能预知未接触缺口 |
| Huber/性能DAgger | 独立BC/DAgger入口、冻结同状态教师标签、有界replay、纯学生验证放权 | 不是MethodsPPO中的联合BC loss；需要合格动作教师 |
| BeamDojo-inspired | 双独立value、reward分流、分别GAE/归一化、soft/hard物理阶段、轮支撑多点reward | 不是双Q取小；轮足采样近似，非原人形落脚实现 |
| GLAD-inspired | 全局attention、条件局部top-K、重建训练教师latent，蒸馏到本体历史学生特征 | 教师是训练表示网络，不是已合格特权动作专家；学生不是视觉学生 |
| WM-LOCO-inspired | action-conditioned一步34维本体预测，预测latent进入actor残差 | 不是RSSM，没有深度输入、循环随机状态、多步预测、AMP或MoE |
| PLANC-inspired | 有限适用域LIPM参考、支撑/摩擦可行性判别、CLF-like奖励 | 不是hybrid MLIP足步规划、全身动力学可行性或稳定性证书 |

新增辅助编码器在整轮PPO epochs结束后更新。privileged map只进辅助损失；actor和JIT部署接口只接原本体历史。checkpoint包括新增网络/optimizer及方法签名。该结构能运行训练代码，不等于已经验证有效、收敛或优于基线。

## Methods与DAgger之间是分阶段桥

1. 在methods任务上训练并独立验证动作策略。
2. 严格匹配配置加载checkpoint，导出318维canonical本体历史到16维动作的冻结TorchScript，并检验导出数值一致。
3. 生成**未认证**清单，再进行真实同状态观测、目标域闭环与控制适配资格检查。
4. 合格后才能在相同soft/hard目标域进入独立DAgger，学生为展开历史MLP。

这个学生没有HIM估计器、dual critic或在线世界模型训练。它可学习教师动作表现，但不能说DAgger优化器直接联合训练了全部方法。也没有将该MLP自动转回MethodsPPO继续微调的逆向桥。

旧策略物理目标映射可用，但目标相同不保证不同PD、延迟和动力学下行为相同。按部分命令域选择性混用旧教师仍须明确有效域与合格fallback控制器；只丢弃无效标签，却继续执行无资格教师或随机学生，不构成安全的选择性示教。当前没有合格教师身份或fallback证据时，正式DAgger仍关闭。

## 需要单独记录的验收

- 原四任务注册/观测/动作/奖励不受新增任务影响。
- 八种HIM/地形蒸馏/预测组合的JIT一致性，关闭分支不推理或反传；训练态残留buffer明确披露。
- 双reward只计算一次，timeout按各value独立bootstrap；终止帧必须在reset前，缺标签/重复标签拒绝。
- 真实环境小规模采样、所有值有限、严格保存/加载与JIT往返；用UNTRAINED fixture只能验证通路。
- 独立DAgger资格、stage/config绑定、replay中无特权输入，验证集不进replay，methods支撑覆盖不能只用vx/yaw均值代替。
- 同一场景、命令、预算、seed矩阵下消融。2000次PPO迭代与2000次DAgger迭代不是等样本/等算力。
- 固定原全向任务的纯直行、纯转向、停稳、左右切换与扰动测试，不能用前行专项代替。

`scripts/check_wolf_release.py` 是双机发布验证入口。它验证源文件manifest、原任务保留、五组bounded真实仿真、强制timeout、JIT/checkpoint往返和权重不变，显式禁止learn/update/backward/Adam.step。输出报告必须新建，不覆盖旧报告。它不是GPU容量基准、训练效果或教师能力验收。

本次源快照和各机器实际结果保存在版本化发布目录及其外部验证报告中。仅源码相同与依赖版本相同不意味着CUDA训练逐位可复现。系统Node12不能启动pyright的工具分析，不能把该失败写成type check通过。
