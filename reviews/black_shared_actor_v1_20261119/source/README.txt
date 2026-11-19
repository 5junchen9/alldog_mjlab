Black共享actor候选v1，2026-11-19

当前已完成隔离代码、CPU更新接线测试、有限GPU零更新、教师PPO更新、原生reset复核、周期存档和新进程学习器恢复，以及独立教师诊断采集与CPU评分。单seed完整GPU诊断：flat8通过29/32，axes通过10/24，最低4cm双向正面阶梯通过0/2。当前候选仍有明确能力缺口，完整教师资格尚未注册，单seed诊断不能证明全部能力合格。学生正式训练入口保持关闭，正式长训与自动跟进保持停止；未部署到Win，旧训练NaN尚未修复。

一、模型和执行链

教师与学生使用同一actor输入：当前p45＋冻结HIM19＋地形32，共96维，输出12维动作均值。HIM19保留next_velocity3和单位化隐变量16的原语义。

教师阶段用PPO更新actor、几何编码器和独立critic。actor几何只来自同一相机可见深度；完整raw446只进入critic。学生阶段冻结完整教师、actor和HIM，只训练CNN+GRU。学生实际控制环境，教师查询学生访问的同一状态，监督动作均值及地形32；两个目标均detach，动作损失仍通过冻结actor传到学生视觉分支。

160×90深度固定采样到64×48，保留真实源pixel与射线对应。学生使用这份depth64；教师从相同depth64选12×16固定子网格，用实际相机内参回投xyz/4和valid，共768维，再编码为32维。相机10Hz、控制50Hz；原metadata决定held/fresh帧和episode reset。旧MetricReducer保持原样，固定像素采样的薄物体漏采与抗噪表现仍需独立验收。

二、已实现文件

run.py：默认plan；probe与teacher入口、关闭的student资格门、显式学习器恢复、完整轮次周期存档、输出拒覆盖、GPU互斥锁及最初异常保留。
model.py：共享actor、冻结HIM、可见几何768→32，以及r800权重初始化。
terrain.py：固定源pixel采样、实际相机内参和可见几何回投。
runtime.py：原生环境、相机帧缓存、episode/reset时序及首次坏状态探针；可选evaluation接入不改变默认训练路径，构造失败时保留首个异常和故障证据。
ppo.py：原生RSL PPO接口；actor重新编码可见几何，critic读取完整446维。
student.py：共享冻结模块、CNN+GRU前向、连续序列和两个Huber监督项。
student_training.py：学生控制采样、CPU保存24步序列、128环境微批BPTT、Adam及状态刷新，支持周期存档和显式学习器恢复。
checkpoint.py：原子不覆盖权重bundle、模型身份和输入约定校验；完整教师资格协议尚未注册并接入，继续拒绝学生正式训练。
learner_state.py：新版本学习器配方、更新计数和参数绑定顺序校验，Python/NumPy/Torch及显式CUDA设备随机状态存取。
eval_protocol.py：固定flat8、纯轴低速和阶梯清单、时序、原绝对动作阈值及明确的not_registered资格状态。
eval_adapter.py：构造前固定命令/路线，原生history与奖励命令见证，reset前阶梯和执行器证据；不修改旧依赖源码。
eval_collect.py：默认plan，显式execute后冻结选定教师、运行真实相机/原生环境、保存完整分母及身份/hash，自动调用CPU评分。
eval_score.py：读取存档、核验身份/时间/完整分母、分项评分和覆盖报告；不开放教师资格。
EVALUATION_PROTOCOL.txt：已实现的独立诊断执行规范，保留旧阈值来源、课程成功与完整过梯稳停的区别，以及尚未注册的资格边界。
test_*.py：CPU模型、梯度、冻结、帧序列、入口、更新与异常保护测试。

内部学生训练函数的首版配方固定为每round24控制步、每128个环境分块、按环境数量加权累计，每round执行1次Adam；学习率1e-4，特征权重1.0，梯度范数上限1.0。这些值属于未调优实验配方。loss非有限、实际参与计算的参数梯度缺失或实际梯度非有限直接停止，没有重试和自动恢复。整个窗口没有fresh valid帧时，GRUCell未调用，其grad允许为None；Adam跳过这些参数，已有动量和步数不变，不将None填成零梯度。教师更新前也检查梯度。

学生跨round保留detached GRU hidden；更新后从本round起始hidden重放这24步，刷新下一round的acting hidden。更早的prefix hidden来自上一模型版本，属于截断近似；尚未实现整个长episode的精确重放。

三、从项目根使用

以下命令只说明入口，不表示已执行。每次执行都需要新的--out目录。

cd ${PROJECT_ROOT}
CAND=preparation/rough_v2/shared_actor_v1_20261119

默认plan只打印输入约定与停止状态，不创建环境或优化器：

uv run --no-sync --offline python "$CAND/run.py"

显式教师控制探针，32个环境×24步，0次更新：

uv run --no-sync --offline python "$CAND/run.py" --mode probe --execute --controller teacher --num-envs 32 --controls 24 --out "$CAND/evidence/probe_teacher_manual001"

显式学生控制探针，32个环境×24步，0次更新；probe不要求教师能力合格，不构成学生训练许可：

uv run --no-sync --offline python "$CAND/run.py" --mode probe --execute --controller student --num-envs 32 --controls 24 --out "$CAND/evidence/probe_student_manual001"

显式教师有限更新示例，1个round包含24步采样；PPO实际optimizer调用数另写入结果，不把round等同于一次Adam：

uv run --no-sync --offline python "$CAND/run.py" --mode teacher --execute --num-envs 32 --rounds 1 --out "$CAND/evidence/teacher_update_manual001"

学生正式训练目前不能由CLI启动。--mode student仍要求--bundle或--resume-learner、--teacher-evaluation与--rounds，但完整教师资格协议尚未注册并接入，资格检查始终拒绝开放，任何手填passed摘要均无效。内部train_student()保留用于接线测试，不能用它替代正式能力验收；学生控制probe继续可用。

上一轮发现旧检查仅要求teacher_id和flat、stairs、low_speed_turn、reverse四项布尔通过，即使complete=False、trials=[]也可能放行。已关闭该漏洞：四项摘要通过后仍报“完整能力评估协议尚未接入”。当前诊断采集/评分已接通，正式资格要求仍未注册，不提供开放学生正式训练的JSON示例。

独立诊断每次选择一个suite：flat8为32条×400控制步；axes为24条×900控制步，覆盖3轴×2幅值(0.10/0.25)×2符号×2朝向，每条含起动、反转和停车；stairs默认为9高度(4/6/8/10/12/15/20/25/30cm)×2方向×3入场角(-15/0/15度)=54条，每条最多6000控制步。所有批次实际使用32环境，axes补8条、stairs末批补10条padding，padding有独立身份且不计分。--seed为基础种子，各批使用seed+batch_id；单次面板不是多seed资格验收。

默认plan打印清单，不读取权重或创建环境：

uv run --no-sync --offline python "$CAND/eval_collect.py" --suite flat8
uv run --no-sync --offline python "$CAND/eval_collect.py" --suite axes
uv run --no-sync --offline python "$CAND/eval_collect.py" --suite stairs

显式执行示例使用已有新教师bundle，它尚未获得完整资格。每次重新生成RUN_TAG，--out必须是新目录；采集完成后自动保存SCORE.json：

BUNDLE="$CAND/evidence/resume_teacher_continue001/teacher.pt"
RUN_TAG=$(date +%Y%m%d_%H%M%S_%N)
EVAL_OUT="$CAND/evidence/eval_flat8_manual_$RUN_TAG"
uv run --no-sync --offline python "$CAND/eval_collect.py" --suite flat8 --seed 119041 --bundle "$BUNDLE" --execute --out "$EVAL_OUT"

另行执行axes或stairs时也使用新目录，不与其他GPU任务并行：

RUN_TAG=$(date +%Y%m%d_%H%M%S_%N)
uv run --no-sync --offline python "$CAND/eval_collect.py" --suite axes --seed 119041 --bundle "$BUNDLE" --execute --out "$CAND/evidence/eval_axes_manual_$RUN_TAG"
RUN_TAG=$(date +%Y%m%d_%H%M%S_%N)
uv run --no-sync --offline python "$CAND/eval_collect.py" --suite stairs --seed 119041 --bundle "$BUNDLE" --execute --out "$CAND/evidence/eval_stairs_manual_$RUN_TAG"

短检查可加--max-controls 12；未完成既定时限的trial保持incomplete，不能作为通过。真实失败仍保留失败结果，不用截短掩盖。RESULT表示采集流程正常结束，行政截短也会发布RESULT，但collection_complete=false，SCORE保持incomplete；异常FAILED输出不能作为独立评分输入。stairs可用--heights-cm和--angles-deg显式缩小清单，缩小范围写入MANIFEST，不补充未测格。

对上述EVAL_OUT进行CPU复评分，不再调用模型、仿真或优化器；复评分文件也拒绝覆盖：

SCORE_TAG=$(date +%Y%m%d_%H%M%S_%N)
CUDA_VISIBLE_DEVICES='' uv run --no-sync --offline python "$CAND/eval_score.py" --manifest "$EVAL_OUT/MANIFEST.json" --result "$EVAL_OUT/RESULT.json" --out "$EVAL_OUT/SCORE_RECHECK_$SCORE_TAG.json"

CPU测试命令（禁用本机ROS的pytest自动插件，避免无关插件依赖干扰）：

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' uv run --no-sync --offline pytest -q "$CAND"

四、存档与显式恢复

无模型来源参数时，从校验过的r800模型样本初始化，建立新学习器。--bundle只加载teacher/CNN/GRU模型权重，重新建立critic、动作分布、归一化统计、Adam和轮数。--resume-learner恢复本版本保存的学习器，两个参数互斥；它只适用于teacher或student更新模式。

--rounds始终表示本次新增轮数。--save-every默认100，按累计完成轮号保存teacher_update_000100.pt或student_update_000100.pt。存档发生在整轮更新和状态刷新完成后；请求轮数全部成功后另外保存teacher.pt或student.pt。下一轮失败时，之前发布的周期存档保留，没有自动重试或自动恢复。probe仍只保存candidate.pt。

教师恢复actor、可见几何编码器、动作分布std、critic及归一化统计、Adam、学习率、累计轮数和随机状态；HIM继续冻结。学生恢复CNN+GRU与对应Adam，教师身份固定。两侧都要求阶段、完整配方、环境数量和参数顺序一致；缺失新schema的早期实验包只能经--bundle作为权重初始化。CPU和CUDA随机状态不能跨类型冒充完整恢复。CUDA始终显式使用指定设备。

物理、课程、相机和episode在新进程重建；学生GRU记忆从空状态开始。恢复学习器不会重建原故障物理瞬间，也不提供连续轨迹逐位等价保证。完整长episode的GRU重放尚未实现。

以下仅是有限续算的使用示例，不会自动执行；输出目录必须不存在：

uv run --no-sync --offline python "$CAND/run.py" --mode teacher --execute --num-envs 32 --rounds 1 --save-every 1 --resume-learner "$CAND/evidence/resume_teacher_seed001/teacher_update_000001.pt" --out "$CAND/evidence/teacher_resume_manual001"

当前存档采用同目录临时文件、flush/fsync及原子不覆盖发布；写到一半报错不留下正式.pt名称，已有有效包保持完整。故障记录、首次坏状态写入或close再次失败时保留最初异常，附加错误写入failure_details/可写出的FAILED.json；没有重试。

五、当前证据与边界

首轮代码与CPU：28项测试通过，耗时4.15s，产生1个Torch上游弃用警告；覆盖真实Torch模型配合合成物理的teacher/student优化器更新、冻结模块保护、连续GRU帧/重置时序和NaN梯度阻止Adam。当轮Ruff、ty通过，Pyright使用应用自带Node和候选pyrightconfig.json后为0errors、0warnings。这份首轮记录保留；合成物理更新测试证明更新接线，不提供机器人通过能力证据。

上一轮完整CPU复测：39项通过，1个Torch上游弃用警告；Ruff、格式检查、ty通过，Pyright为0errors、0warnings，最终回执为evidence/REVIEW_VALIDATION.json。新增检查覆盖全无效相机窗口的GRUCell/Adam状态、中断与重复目标原子保存、原异常保留、终止优先，以及不完整资格摘要拒绝放行。

GPU零更新：
- evidence/gpu_teacher001/RESULT.json：teacher控制，32环境×24步，optimizer_calls=0，reset_events=0；共享head在同输入上精确一致，教师权重不变；HIM next_velocity逐轴MAE=[0.0470,0.0314,0.0512]。
- evidence/gpu_student002/RESULT.json：student控制，32环境×24步，optimizer_calls=0，reset_events=0；共享head在同输入上精确一致，教师权重不变；HIM next_velocity逐轴MAE=[0.0441,0.0329,0.0459]。

以上两次早期GPU结果各统计768条非终止下一步速度误差。这些MAE仅描述对应短轨迹，未设置能力通过线；这两次运行均没有优化器更新或reset。

本轮GPU复核：
- evidence/review_teacher_update001/RESULT.json与WEIGHT_CHECK.json：32环境×24步、1round、20次Adam；actor与geometry encoder改变，HIM/CNN/GRU不变，critic的raw72与terrain归一化计数均768，全部存档张量有限。未做能力验收。
- evidence/review_native_reset002/RESULT.json：在终止优先修复后的runtime上，向lane0注入原生timeout，32环境运行3控制步、0更新；history重置、相机新帧和GRU新episode状态通过，其余31lane相机缓存与GRU hidden保持，权重不变。未做能力验收。review_native_reset001保留为先前复核证据。

原生runtime.step现已让真实终止优先于同时出现的timeout。纯timeout仍沿用上游V(s_t)近似，未实现终止观测的精确bootstrap。

evidence/ORIGINAL_RECHECK.json及本轮evidence/REVIEW_ORIGINAL_RECHECK.json均确认51个原文件指纹一致。旧入口、旧reducer及历史证据保持。

evidence/gpu_student001/FAILED.json和对应stdout.log保留早期启动失败：KeyError('source_actual_updates')。后续gpu_student002成功结束不删除或替代这份失败证据。

本轮恢复验证：evidence/resume_teacher_seed001完成32环境×24步、1round、20次Adam；新进程resume_teacher_continue001从周期包恢复，再完成32环境×24步，累计2round、40次Adam。两组critic归一化计数从768接到1536，HIM/CNN/GRU不变，存档张量均有限；见evidence/RESUME_GPU_CHECK.json。每次都是新物理episode，不构成能力验收。学生学习器恢复本轮只有合成CPU证据，未运行学生GPU更新。

学习器恢复轮CPU共59项通过，耗时6.26s，1个Torch上游弃用警告。19个当轮实现/测试Python文件的Ruff与格式检查通过，ty通过，Pyright为0errors、0warnings；历史诊断脚本原样保留。回执见evidence/RESUME_VALIDATION.json；51个原文件再次校验一致，见evidence/RESUME_ORIGINAL_RECHECK.json。

独立评估接入轮CPU共101项通过；Ruff、格式检查、ty和Pyright均通过。新增反例覆盖固定路线和命令时序、reset前证据、重复试验/错误身份/缺帧拒绝、缺测不冒充通过、padding不计分，以及构造期故障写入或close再次失败仍保留最初异常。

独立诊断GPU短检查：evidence/eval_flat_smoke001运行32环境×12控制步、optimizer_calls=0；32条全部started，32条incomplete，complete=0、success=0、invalid_evidence=0，教师未更新。短检查证明采集/评分链可运行，没有形成动作通过结论。

完整flat8诊断：evidence/eval_flat_full001的32条均完整计分，29pass/3fail；失败包含2条原生失败和1条停车未达标。invalid_evidence=0、optimizer_calls=0。collection_complete=true表示分母与结束记录完整，不能解读为32条全部通过。

完整axes诊断：evidence/eval_axes_full001的24个纯轴命令/朝向格均分配并完整计分，10pass/14fail；失败包含2条原生失败和12条动作或停车未达标。invalid_evidence=0、optimizer_calls=0。以上都是本次单seed面板结果，不能替代多seed证据或完整教师资格；学生正式门继续关闭。

阶梯短检查：evidence/eval_stairs_smoke001仅测4cm、双向正面2条真实trial，另30条padding不计分；各运行120控制步，2incomplete、0physical_failure、0invalid_evidence、0更新，ordered_treads=0。短检查未形成踏面或完整过梯证据。

阶梯完整诊断：evidence/eval_stairs_full001对同一4cm双向正面2条路线运行到原生结束，分别在343步和481步原生失败，终止前有序踏面分别为0和3；0通过、0invalid_evidence、0更新，教师权重不变。当前候选未通过最低4cm这两条诊断；采集器正常保存了失败和终止前证据，没有采集器报错。仍无完整过梯成功的GPU样本，默认54格未完成覆盖。

尚未完成：固定像素对薄物体/噪声的验收、完整能力覆盖与多seed证据、教师资格阈值注册及资格门接入、长episode精确GRU重放和长时间运行验证。独立诊断采集/评分器已实现；EVALUATION_PROTOCOL.txt保留旧课程pass_timeout_no_stop不等于完整过梯稳停、旧配对差值阈值和课程升级成功率不能直接转成新教师资格的边界。有限GPU更新与一次原生reset已有上述证据。未部署Win、未启动正式长训，旧NaN故障尚未修复；当前探针帮助记录首次坏状态，不承诺完整故障重放。
