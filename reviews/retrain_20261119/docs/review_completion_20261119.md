# 2026-11-19中断改动补完与审查索引

本次以原版提交`323b2e7b043b79d89e443b03619ff14f0606ade9`为比较基准。接手时工作树已有24个修改文件，用户粘贴的执行记录停在额度耗尽，入口、训练恢复、测试和文档尚未接齐。保留已完成的方向，逐项核对14项审查意见，补齐调用链并修正新增接口断裂。当前源码在`/home/wujuncheng/rl_train/retrain`。

教师顺序保持基础平地→轴向起停→综合六域→六个专项分支。全部从新随机参数开始，专项继承本条新训练线的合格综合教师。没有启动正式长训、自动跟进或真实硬件。

## 14项对照

|序号|原版/中断状态|本轮整合结果|代码与反例入口|
|---|---|---|---|
|1|Wolf桥桩评分仅使用箱顶，遗漏mesh坡面|原场景箱顶与mesh顶面统一投影为平面，正反方向坡面可通过，核心区底板仍失败|`wolf/eval_score.py`；`wolf/tests/test_review_fixes.py::test_mesh_ramp_support_passes_but_core_floor_fails`|
|2|Wolf无通关奖励；Black五专项无尾值补偿|Black六项都有12补偿、原事件总额3和总额1的49步稳停塑形；Wolf六项有7.5通关、单调进度、总额1稳停塑形和停滞扣分，配置核对正奖励尾值上界5.5；硬失败优先|`support/black/specialty_env.py`、`completion_value.py`；`wolf/rewards.py`、`curriculum.py`|
|3|Black从零仍5e-5固定学习率|基础/轴向1e-3，综合3e-4，专项1e-4，均KL自适应；配方绑定恢复|`support/black/config.py`、`black/run.py`|
|4|FirstBad每个子步同步并复制状态|默认仅控制步边界检查qpos/qvel/qacc/warmstart/ctrl；两入口显式`--debug-first-fault`接通完整探针|两种runtime/run、`black/fault_capture.py`|
|5|完整episode重算、Python逐lane缓存、1次Adam；中断版只改Black接口|Black/Wolf复用一个学生循环。向量化保存50张新有效帧，满段清零；采样、BPTT、JIT采用相同边界。每轮4epoch，每批256lane，每次Adam前按当前参数重算burn-in，随后24步BPTT；恢复完整Adam计数|`black/student_training.py`、`support/black/causal_visual_gru.py`、`black/export.py`、`wolf/training.py`|
|6|空隙1m饱和值污染抬脚/落地峰值|无支撑射线不参与抬脚高度和峰值；专项仅罚抬脚不足，允许跨高障碍抬腿|`src/mjlab/tasks/velocity/mdp/rewards.py`；`test_gap_miss_does_not_poison_clearance_or_next_landing_peak`|
|7|Wolf全流程低速单轴；低速核太宽；vy与侧滑冲突|分阶段扩速度并增加复合命令；低速宽细核混合；横移/偏航需求均降低侧滑成本；专项只用前进+偏航纠偏；资格逐分量检查复合命令|`wolf/curriculum.py`、`rewards.py`、`eval_protocol.py`、`eval_score.py`|
|8|Black平地屏蔽躯干/低机身失败和碰撞代价，非平地大腿单帧即死|平地保留躯干、低机身和倾角失败，所有地面保留碰撞代价；大腿持续0.15秒失败，专项增软代价|`src/mjlab/tasks/velocity/config/black_rough/mdp.py`、`support/black/specialty_env.py`|
|9|Wolf无翻车/越界保护，综合直接随机0..4档|70°倾角失败；综合保留边界、从第1档起，按真实闭合episode的位移与跟踪质量逐域晋级，不用行政reset投票|`wolf/curriculum.py`|
|10|CNN末层ELU约束输出下界|32维视觉输出改为线性，与教师地形特征值域一致|`support/black/depth_him.py`|
|11|Black HIM编码命令；原型原地修改；小尾批；交换损失缩小32倍|Black HIM编码5×42无命令历史，目标为42本体+3真实速度。均衡切批避免2048+1，分母保护；原型只在Adam后统一归一化；交换损失对原型求和再对样本均值，温度0.5|`support/black/him_base.py`、`src/mjlab/rl/wolf_him/estimator.py`、`black/run.py`|
|12|Black动作延迟历史跨episode保留|reset仅清对应lane历史，前1–2步回到默认PD目标，其他lane不受影响；导出manifest同步|`black_actuation/actuation.py`、`support/black/config.py`、`black/export.py`|
|13|Black不保存课程；稳停只塑形5步；专项平地只低速|保存综合各域窗口/上限、专项高度/角度/历史、楼梯课程票据，冷启动不拼接在途物理。稳停覆盖1..49步。专项平地增加全向/中高速回放。critic增加课程时间、稳停进度、已入场3维，避免用相同价值输入混淆倒计时|两种curriculum/runtime/training；`support/black/exit_stability.py`、`critic.py`|
|14|无深度范围、无成像扰动；固定GPU0锁|0.2–4m外置无效，保留真实源像素，训练期在原始采集处统一加噪后生成教师/学生输入；每episode安装微扰。GPU锁按物理UUID共享，不同卡隔离，CPU进程互不阻塞|`black/terrain.py`、`camera_randomization.py`、`runtime.py`、`eval_collect.py`|

## 本轮主动修正的中断版问题

- Wolf仍用Black旧`_state_slice/_sequence`参数，重构为复用当前训练循环，并用Wolf自己的身份与存档模块。
- 仅在Adam后截断prefix会让训练与推理记忆不一致；改成显式50张有效新帧的记忆段，JIT新增`fresh_count`输入/输出。它改变了长记忆能力，应在实训中比较，不能称完整episode精确反传。
- 多个minibatch/epoch会改变视觉参数；每次更新前都重新计算其burn-in，避免继续用采样时的旧hidden。
- 中断版把10m以上深度夹到10m，会制造虚假平面且未处理0.2m盲区；现统一按声明的0.2–4m有效范围处理。
- 成像噪声若每控制步重抽，会让相同帧身份对应多张图；现每次10Hz曝光只扰动一次，教师几何和学生图像共享，评价保持标称采集。
- 课程恢复同时保存已结算窗口与时间戳，重新分配物理episode；不复用旧延迟、相机、GRU或在途通关信用。
- GPU独立恢复实测还补齐了首步前终止字段、综合课程初次reset、稳停账本与episode日志字段三条实际断裂；反例覆盖冷恢复和日志精度，专项短采样强制超时结算。

## 训练与梯度

Black共享actor仍96→12，Wolf仍104→16，每种机器人各有一套权重。HIM在采集/PPO时冻结，PPO更新actor、可见地形编码器、critic及动作分布；随后辅助Adam只更新HIM历史/目标编码器和原型。Blackcritic为原446+课程3=449，Wolf为原430+课程3=443，这3项不送入actor或学生。

学生由自身动作访问状态，冻结教师在该状态提供标签。动作Huber穿过冻结actor更新CNN/GRU；特征Huber也只更新视觉。教师、HIM、地形编码器均冻结。50帧段的burn-in无梯度，本轮24步有梯度；默认4096环境每轮`4×ceil(4096/256)=64`次Adam，32环境每轮4次。缓存不再随120秒episode持续增长。

当前合同为`black_retrain_shared_actor_v2`、`wolf_shared_actor_v2_scratch`。此前v1的证据和参数不能作为当前源码初始化或资格；完整恢复要求源码、资产、运行库、课程、cap与训练配方一致。

## 验证与索引

新证据根目录为`evidence/review_completion_20261119/`，原版`evidence/black/`、`evidence/wolf/`保留历史归属。

- `VALIDATION.json`：本轮CPU、GPU、同步与能力边界的汇总。
- `GPU_MATRIX.json`与`gpu/`：逐项运行结果、源码身份与有限优化器次数。
- `SOURCE_INDEX.json`：本轮修改文件、真实路径、行数和SHA256，供逐文件审计。
- `scripts/verify_gpu.py`：独立进程有限更新、冷启动恢复、NaN注入和4096短采样复现入口。学生检查明确绕过资格门，只测试优化链，不能用作生产入口。

完整常规检查为`make check`，新增反例位于`black/test_review_fixes.py`和`wolf/tests/test_review_fixes.py`。最终结果以`VALIDATION.json`为准。当前JC与Windows各258项CPU测试通过，JC静态和格式检查通过；34组独立进程GPU检查通过，包含两种机器人的教师/学生续训、综合课程保留、Black平台/楼梯及Wolf楼梯续训、六专项强制timeout、两组NaN注入，以及两种机器人4096环境×4步、0优化。另有7组真实原始采集/复评分，共174条截短attempt，全部保持未完成、成功数0、资格不放行。Windows本轮仅验证CPU，未验证GPU优化。

## 仍需取得的证据

旧自然NaN没有原故障完整求解器状态，根因尚未确定。控制步检查与debug探针用于尽早停止、保存新现场；人工注入只证明取证流程。审查文本所述“慢5–20倍”和“100%误杀”等数值并未直接作为实测结论，实际修复依据为调用链和具体反例。

温度0.5、奖励系数、噪声参数和速度上限均为新从零训练的仿真起点，尚无收敛对比。固定段记忆是否足够、范围内飞点鲁棒性、细物体/遮挡表现需要闭环数据。深度量程外表面不再提供可见几何，长距离规划能力要单独验证。

4096学生的Black采样/缓存估算约5.33GiB、Wolf约5.40GiB，需要当时可用CPU内存至少约6.7GiB才通过80%准入；这是缓冲估算，不代表仿真、显存和真实峰值已验收。短规模检查不证明长期稳定、课程通过或实机能力。三组评价reset种子也不等于三条独立学习seed。

Black学生ONNX、Wolf新架构导出及硬件适配仍未完成；Wolf支架和相机外参还需定稿与实测。这些事项不阻止仿真框架的审计，实机前必须完成对应验证。
