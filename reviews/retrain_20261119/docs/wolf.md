# Wolf从零训练与共享actor

当前入口为`python -m wolf.run`，运行前设置`PYTHONPATH=src:.`并使用`uv run --no-sync --offline`。默认只打印计划；`--execute`才创建仿真和优化器。新合同`wolf_shared_actor_v3_scratch`拒绝旧Wolf、Black及不同源码/资产的权重。所有模块从新随机种子初始化，阶段间仅传递本轮新合同教师。

## 教师与学生

|分支|输入与更新|
|---|---|
|本体|当前53维；历史6×53，含命令、角速度、重力方向、12腿关节位置/速度、4轮速度、16个上次动作|
|HIM|历史去掉每帧命令后估计下一步真实速度3维和16维隐变量；速度监督为2×m/s|
|教师可见地形|与学生相机同帧同视野的192个真实射线点及有效位，共768维，经编码得到32维|
|共享actor|53+19+32=104维输入，输出16维；教师和学生引用同一对象|
|critic|无噪本体53+真实速度3+全地形高度187+有效位187=430维，另加课程时间/稳停进度/已入场3维，共433维；原430维在线归一化，末尾course3保持原值；特权量只帮助价值学习|
|学生|深度CNN+因果GRU输出32维，送入冻结共享actor；只更新CNN和GRU|

每轮24控制步：冻结HIM采样并完成整轮PPO，PPO更新actor、可见地形编码器、critic和动作分布；随后冻结其他模块，用独立Adam更新HIM的下一步速度Huber损失及原型交换损失。终止监督在reset前复制；真实失败行不进入HIM监督，纯超时用真实末帧价值估计。

学生用自己访问的同一物理状态取得教师标签。动作损失经过冻结actor把梯度传给视觉分支，教师标签detach；特征损失只对学生视觉32维求梯度。视觉学生与Black复用同一训练循环：50张健康新帧为最大缓存，后25帧预热新状态并逐渐混合输出；满窗后提升已预热状态，episode重置整体清零。每轮24控制步、4个epoch、每批最多256条lane；每次Adam前用当前参数重算当前段burn-in，梯度仍限24步。4096环境估算缓存约2.36GiB，总采样/缓存约5.40GiB，超出可用内存80%时在采样前停止。教师与学生使用不同checkpoint模块，Wolf完整恢复同时绑定course/cap。

## 轮足动作、奖励和相机

动作顺序固定FL→FR→RL→RR，每腿hip/thigh/calf/foot。腿目标为标称姿态加动作×(.125,.25,.25)rad；轮目标速度为动作×5rad/s，轮方向符号为(+,-,+,-)。腿PD为80/2、力矩上限60Nm；轮PD为0/.6、力矩上限17Nm。保留机械资产和控制合同来源分别为原资产提交`886015a`及控制提交`2913d3f`。这些是声明的仿真配置，实际电机曲线尚未测定。

基础奖励保留速度跟踪、姿态、相对地形高度、动作平滑、关节限位和力矩成本。新增轮胎材料接触点的纵向/横向滑移成本：纯滚动不会因轮心前进被误罚；原地转向和主动横移均降低横向擦滑权重；专项纠偏采用前进加偏航，避免主动横蹭。低速跟踪混合宽核与细核，0.2m/s指令下静止的x跟踪分值降至约0.494。停车分别约束机身速度、轮速度和腿速度。移除障碍上强制全部腿回标称姿态的站立奖励，保留hip弱约束。未套用Black的足式腾空、落足节奏要求。

相机集中定义于`wolf/camera.py`。按用户确认，沿用Black的424×240参考分辨率、内参、87°×58°视场、10Hz采样，160×90渲染后保留最近有效像素降采样到64×48。Wolf候选镜头安装在base_link相对位置(.245,0,.105)m，俯角约31.3°，避开机身碰撞盒前缘。安装架未画，状态明确记为仿真候选；实际支架定稿后必须更新外参并重做可见性和实机标定。未制作虚构支架CAD或声称已标定。

## 课程与资格

顺序为flat_foundation→flat_axes→general→专项。general沿用Black的上坡、下坡、上楼梯、下楼梯、碎地、波浪六域，25%平地；运动段延长至10秒，避免一直待在生成地形中央安全平台。专项为stairs/platform/high_wall/slope/broken_bridge/plum_stakes，共用原真实障碍碰撞几何，分别保留25%平地回放。

专项必须显式传`--max-level`，表示从0到指定关的连续范围。教师从第0关起步，最高关最近16次真实attempt至少14次完整通过才晋级；已解锁关卡有25%回放。通过要求入场、有效接触、通过出口并四轮接触停稳1秒；楼梯须按序由前后轮经过每个台阶及平台；桥桩核心区任何轮触底板均失败。超时没有完成停稳不晋级。此训练门是采样课程控制，独立资格仍以原始轨迹复评分为准。

`wolf.eval_collect`固定seed1911701/1911702/1911703，记录原始逐控制步动作、实际接收动作、命令与本体命令项、速度、姿态、接触位置、原生失败/超时。首次reset后不把新episode拼到旧attempt。截短采集保留incomplete，不能作资格。每个JSONL、manifest、教师权重、源码、资产、运行库均绑定摘要，`wolf.qualify`重新读取原始轨迹，不信手填passed或SCORE.json。

学生按当前course及前置课程取得资格。例如stairs学生需当前同一教师的基础、纯轴、general和所选楼梯范围证据，无需先学完所有专项最大障碍。学生运行范围锁定资格cap，不自行解锁更高关；缩小cap只缩小声明的能力范围，不能省略范围内中间关。全部六专项最大尺寸验收单独报告，不能由一个小范围资格替代。

```bash
# 查看计划；开始基础教师训练只需添加--execute
PYTHONPATH=src:. uv run --no-sync --offline python -m wolf.run --mode teacher --course flat_foundation --num-envs 32 --rounds 1 --out runs/wolf/foundation
# 当前教师的固定3seed原始评价（目录必须全新）
PYTHONPATH=src:. uv run --no-sync --offline python -m wolf.eval_collect --course flat_foundation --bundle runs/wolf/foundation/teacher.pt --out runs/wolf/eval_foundation --execute
# 建立JSON索引，如{"attempts":["eval_foundation"]}，路径相对索引位置
PYTHONPATH=src:. uv run --no-sync --offline python -m wolf.qualify --index runs/wolf/index.json --bundle runs/wolf/foundation/teacher.pt --course flat_foundation --student
# 资格通过后训练该范围的视觉学生；视觉分支重新初始化
PYTHONPATH=src:. uv run --no-sync --offline python -m wolf.run --mode student --course flat_foundation --bundle runs/wolf/foundation/teacher.pt --qualification runs/wolf/index.json --out runs/wolf/student_flat --execute
```

继续同一阶段用`--resume-learner`恢复actor/critic/HIM/视觉、优化器、计数、随机状态、综合六域窗口与专项解锁记录；物理、相机及episode重新开始，不能称故障现场精确重放。切换阶段用`--bundle`传新合同教师，再提交独立评价索引。严禁用短GPU接线测试结果填充正式资格。

源码/资产摘要与运行库版本分别保存。跨JC/Win的`--bundle`权重加载允许Torch/CUDA运行库不同，源码和资产必须一致；完整`--resume-learner`要求两者均与存档一致。资格证据始终绑定采集时运行库，换机后必须在目标机重新评价，不能直接复用JC资格启动Win学生。测试与说明文档不参与训练源码摘要，运行实现及机械/场景资产参与。

## 验证入口和限制

`PYTHONPATH=src:. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --offline pytest -q wolf/tests`覆盖共享actor梯度隔离、HIM单独更新、16维PPO、旧合同拒绝、课程/动作配置、连续cap、真实几何面板、原始评分及伪造/截短拒绝。

GPU短测试只能证明环境、相机、优化器及恢复链能工作。多seed训练收敛、4096环境长期稳定及完整专项能力仍需独立验收。正常训练逐控制步检查物理有限性，诊断时加`--debug-first-fault`启用首坏子步记录，保留物理状态和动作证据；旧NaN的根因仍不能由重新初始化后未复现推定已修复。

当前Wolf未实现教师或学生的TorchScript/ONNX导出入口，也没有新104→16共享actor的部署适配器。现有`.pt`文件用于同源Python训练、恢复和仿真评估，不能直接交给旧实机控制器。后续部署还需实现并验证模型导出、本体历史排列、轮速符号、动作延迟、相机外参与GRU状态管理，再完成实机闭环测试；Black已有导出入口不能直接替代Wolf这条链。

旧的双头/盲学生方法试验、stones/beams替代专项、未启用软件速度guard和旧HIM内置优化器入口已移除。本地Git保存必要源代码和机械资产；私有Wolf网格不得随公共Black审查PR上传。权重、日志及渲染缓存置于Git忽略的runs，避免反复打整目录备份包。

## 本轮审查补完

基础阶段保持0.2–0.3m/s前后起停，flat_axes加入最高0.8m/s及转向复合命令，general最高1.2m/s并保留低速/全向；专项速度0.2–0.5m/s，平地回放覆盖综合速度范围。3m/s未作为已标定可用能力加入训练。

general从各域第1档开始，真实episode结束时同时核对离开起始平台、未越界、有效需求至少1m、投影前进达到需求80%、至少90%运动步达到半目标速度；当前档至少32条记录、最近最多64条成功率90%后解锁相邻一档，上限4。全阶段保留70°倾角失败，general保留地形越界失败，专项采用自身廊道失败。

六专项统一发放一次性完成奖励7.5，大于2.75×0.02/(1−0.99)=5.5的正奖励尾值上界；增加单调进度、完整49步停稳塑形及停滞扣分。硬失败优先于同一步通关。桥桩原始评分包含真实mesh入口/出口坡面，核心区底板仍失败。flat_axes资格新增中速和复合命令，逐分量核对转向与横移，v2增加命令覆盖；当前v3还要求健康且同回合的相机流，空视野可以健康。详见[边界复审报告](boundary_review_20261119.md)。

相机深度范围、成像噪声和安装微扰与Black共用实现，只是仿真候选，参数与验证边界见[补完报告](review_completion_20261119.md)。
