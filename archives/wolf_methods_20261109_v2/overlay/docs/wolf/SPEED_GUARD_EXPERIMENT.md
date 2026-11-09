# Wolf 可选软件速度保护实验

2026-11-09。仅新增独立实验配置，不改变设计者原四任务、不覆盖旧策略profile，也不启动训练。

## 证据与含义

源URDF提交 `886015aef351f1e5517cd50e47628be35ec315ed` 声明腿关节20.42035 rad/s、轮关节32.98672 rad/s。它们分别约等于RS03的195 rpm和RS01的315 rpm输出轴空载转速。

[厂家官方资料](https://github.com/RobStride/Product_Information/blob/d739e98128afb8371dd177400c44ad54682ece23/README.md)明确为空载转速且有±10%公差，不是刚性机械速度上限。项目电机型号信息与这些数值一致，不等于本次独立验证了实际装机电机、驱动参数、电压、减速器或固件。

已有[部署轮速处理](https://github.com/coverMoon/quadruped_control/blob/7fcc1ff2a1de4b2352f082a491385dbfccdafc29/motion/src/rl_controller.cpp#L445-L460)只夹紧velocity_target，[MuJoCo后端](https://github.com/coverMoon/quadruped_control/blob/7fcc1ff2a1de4b2352f082a491385dbfccdafc29/backends/mujoco/src/joint_control.cpp)按实际qvel计算PD并夹力矩，没有本实验的同向驱动力矩门控，也没有硬改qvel。部署配置里的腿30/轮80 rad/s仍标注simulation-only候选值。本实验不是找到了设计者已有保护，而是新增可选软件保护。

不采用现有线性DcMotor模型：只有空载转速、峰值/额定扭矩，不能辨识实际转矩-转速曲线、电流平台、反电动势、制动区、母线电压与温升。厂家不同资料版本的额定扭矩也有差异，本次不修改额定/峰值参数。不编造热模型或过载持续时间。

## 实现

- `src/mjlab/asset_zoo/robots/wolf/speed_guard.py`：独立WolfSpeedGuardActuator。
- `src/mjlab/tasks/velocity/config/wolf/speed_guard.py`：独立factory，读取本地源URDF阈值，deepcopy原环境后仅替换执行器类型。没有注册新默认任务。
- `tests/test_wolf_speed_guard.py`：CPU/CUDA与物理反例测试。

每个5ms物理子步，沿现有“目标延迟→读取当前无延迟反馈→执行器→最终ctrl”的路径：

1. 把最终物理velocity_target限制在±URDF声明速度；不改位置target、raw policy action或previous_action。
2. 使用原Kp/Kd、实际qpos/qvel和feedforward计算净力矩，保留原60/17N·m effort饱和。
3. qvel≥+阈值时禁止正向净力矩；qvel≤−阈值时禁止负向净力矩。
4. 反向制动力矩照旧允许，但仍受原effort上限限制。

限值内且速度目标未越界时，与原IdealPd逐位一致。门控使用关节物理坐标，不使用带forward_sign、观测噪声或scale的策略速度。继承现有fused actuator、DR gain/effort setter和命令延迟路径；没有在门控后额外加入力矩。

**绝不裁剪qpos/qvel。** 外力、重力、接触、关节耦合、惯性与离散积分仍可造成实际超速。门控不会自动产生额外制动，也不保证在超速时立刻减速。阈值开关可能产生抖动，本次没有暗加滞回或降额曲线。允许理想PD反向制动也不表示已模拟真实再生制动。

## 显式选择实验

原有四个Task ID的行为完全保持。实验只能显式构造：

```python
from mjlab.tasks.velocity.config.wolf.speed_guard import wolf_speed_guard_env_cfg

cfg = wolf_speed_guard_env_cfg("wolf-flat", play=False)
```

也可把已构造的Wolf环境传入`with_wolf_urdf_speed_guard(cfg)`，返回独立副本。只接受16个逐关节原IdealPd执行器，未知配置和重复套用会报错。该实验不要冒充原版profile保存或用于认证旧策略。

可复查的无训练A/B程序：
`${WOLF_EVIDENCE_ROOT}/speed_guard_experiment_20261109/compare_profiles.py`。
它只严格加载现有本地`UNTRAINED_FIXTURE`，禁用learn/update/backward/optimizer.step，不保存新训练权重。

## 验证

- 45项focused测试通过，包括本次29项与此前16项源/控制契约测试。
- Ruff与新文件ty通过；全仓测试未运行。Pyright仍受既有Node12环境限制，本轮未安装或更改软件。
- 验证两种阈值正负边界、反向制动、输入不变、非有限反馈不隐藏、限值内逐位等价、镜像、速度目标clamp、force cap、fused路径、DR增益/力矩更新、延迟后真实反馈、partial reset不串环境。
- 额外MuJoCo单关节试验施加外力，实际速度继续超过阈值而未被裁剪。该测试故意证明保护不是运动学硬限速。
- 四任务原版/实验各1环境×1000控制步×4物理子步，共8000控制步、32000物理子步。保持非play随机化与终止项，所有首回合均在第1000步原生timeout，观测/动作/奖励/力矩有限，策略前后哈希相同。
- 每一对使用同seed、同未训练权重，初始观测哈希相同。每个物理子步检查最终ctrl，实验全部满足“达到阈值后不再同向驱动”；力矩不超60/17N·m。没有拿策略步采样冒充物理步验证。

下面仅是每任务一个seed的功能数据，不是性能比较或安全认证。速度单位rad/s；“超速关节样本”为逐关节逐物理步累计，不是独立事件数；“超速同向驱动样本”在执行前的实际qvel和最终ctrl上检查。

| Task / profile | 腿峰值 | 轮峰值 | 超速关节样本 | 超速同向驱动样本 |
|---|---:|---:|---:|---:|
| wolf-flat/original | 33.21587 | 13.73944 | 10 | 5 |
| wolf-flat/software_guard | 26.23889 | 12.90615 | 9 | 0 |
| wolf-rough/original | 50.15071 | 15.77680 | 26 | 14 |
| wolf-rough/software_guard | 34.61265 | 15.52088 | 15 | 0 |
| wolf-flat-him/original | 39.58565 | 17.50791 | 34 | 14 |
| wolf-flat-him/software_guard | 27.68106 | 12.82715 | 21 | 0 |
| wolf-rough-him/original | 22.31123 | 11.24988 | 1 | 0 |
| wolf-rough-him/software_guard | 22.31123 | 11.24988 | 1 | 0 |

实验仍记录到腿速度超阈值，不能宣称“实际关节速度已硬限制”。本次fixture没有触发轮速度目标clamp，其执行路径由边界单测覆盖。rough-HIM没有触发门控，原版/实验数据相同。
完整逐关节峰值、超速采样时长、门控次数、初始观测与策略哈希见`comparison.json`。guard内部计数包含reset时的scene写入；物理步速度/最终ctrl统计严格只在sim.step计数。
真实已训练策略、长期稳定性、阈值抖动影响、实机驱动一致性、电机曲线与热模型仍待确认。
