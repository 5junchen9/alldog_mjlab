# Wolf 源数据对齐与版本边界

核验日期：2026-11-09。没有训练，没有改物理参数、PD、动作尺度、随机化或终止规则。

## 采用的来源

- 机械源：用户已下载的 `XJTURoboCon_quadruped_assets-main.zip`，ZIP GitHub 注释提交 `886015aef351f1e5517cd50e47628be35ec315ed`。SHA256 `7c0f5a066ca52713aac0a0e07470a7e8cfe57d10f080d93f022768c093713752`。请求的 N-W-wolf 资产仓库当前 GitHub API 返回 404，因此这里只称已核验历史快照，不声称远端最新版。
- 训练控制：[alldog_mjlab 2913d3f449f8eb5b0505156c460ad2f61e4c49fd](https://github.com/N-W-wolf/alldog_mjlab/tree/2913d3f449f8eb5b0505156c460ad2f61e4c49fd)。本地归档和 N-W-wolf、coverMoon 两个远端 main 均为此提交。N-W-wolf 是 coverMoon 的 fork，不因仓库组织名推断硬件版本。
- 资产 README 指定 URDF 是几何、惯量、层级的源；training_mjlab 与 sim2sim 是不同用途派生层，不能互相覆盖。

## 对齐结果

本地 URDF 与源 URDF 逐字一致，训练 MJCF 与资产库训练 MJCF、alldog 的训练 MJCF 均逐字一致。17 个 STL 与源 URDF meshes 逐字一致。完整文件 SHA256 和源关节/惯性数据保存在 `source_contract.json`，由 `tests/test_wolf_source_contract.py` 回归检查。

名义总质量 33.6686 kg。四腿 policy 顺序 FL、FR、RL、RR，每腿 hip、thigh、calf、foot。FL/RR hip 限位 [-0.6,0.8]，FR/RL [-0.8,0.6]；thigh ±3.14159、calf ±2.72 rad，轮为连续关节。腿/轮 effort 为60/17 N·m。轮半径0.08m、半宽0.0225m，机身碰撞盒Y半宽0.10m，IMU偏移(0,0,0.05975)m。

现行控制参数继续与所选 alldog 提交一致：腿80/2、轮0/0.6；hip scale0.125、thigh/calf0.25、轮5 rad/s，FL/RL为正、FR/RR为负。physics dt0.005s、decimation4、50Hz。默认root z0.4432m；thigh/calf分别为FL/RR +0.82/+1.43、FR/RL -0.82/-1.43。此轮没有“为了对齐”改动已经一致的数据。

## 不应混用的旧配置

- 10月资产包 training_mjlab：腿80/3、轮0/1。其文档明确实机电机规格仍待确认。
- alldog M20修改前：hip90/3.3、thigh/calf80/3、轮0/2，腿scale0.25、轮10。
- [旧 Wolf test 部署配置](https://github.com/coverMoon/quadruped_control/blob/7fcc1ff2a1de4b2352f082a491385dbfccdafc29/configs/policies/wolf/test.yaml)：腿80/3、轮0/1、腿scale0.25、轮10。它不能作为当前训练配置的教师或兼容性证据。
- 当前参数源于 [391c6a6 M20对照改动](https://github.com/N-W-wolf/alldog_mjlab/commit/391c6a6ccf48cb14edb9d687d83dc32eef49a3cf)，不是重新辨识机械设计。
- [86ad1d88 HIM改动](https://github.com/N-W-wolf/alldog_mjlab/commit/86ad1d88fa0d7d048177c7bf8eb1431a7d2088e1) 把内部source encoder输入从318改为300，外部TorchScript仍318→16。仅shape一致不能证明控制兼容。

历史策略必须用其训练保存的资产、观测归一化、关节顺序、增益、动作尺度和时间步；没有完整清单时不能认证。本次只回放本地明确标记的未训练fixture验证软件通路，不声称真实策略行走或教师通过。

## 仍未覆盖的物理与软件边界

URDF记录腿速度20.42035、轮32.98672 rad/s。当前IdealPd只限制力矩，不强行限制实际关节速度，也没有转矩转速包络、额定/峰值持续时间或热模型。未凭两个峰值点虚构电机曲线。部署侧30/80 rad/s是仿真候选保护值，不是设计源速度规格。

本地仍是MuJoCo/mujoco-warp3.11、RSL-RL5.5.1；alldog锁定3.14/3.14、RSL-RL5.4.2。机械与控制契约一致不等于完整运行栈复现。没有改驱动、安装依赖或迁入上游EPA补丁。原先已知圆柱MULTICCD接触限制仍在。

备份和本次来源/回放报告位于 `${WOLF_EVIDENCE_ROOT}/design_alignment_20261109/`；未修改或覆盖旧权重。

## 本次实际验证

- 源文件逐字/哈希核验通过；CPU编译 nq23、nv22、nu16、17个mesh，总质量33.6686kg。逐关节限位和轴、逐刚体质量/质心/惯量矩阵、轮几何与默认姿态前进符号均通过。
- 现有5项与新增11项合计16项pytest通过。新增测试覆盖四任务各自train/play控制契约。Ruff与改动文件的ty检查通过。
- 隔离Wolf运行环境下，4个已有本地未训练fixture严格加载，各1000控制步/4000物理步/20模拟秒，共4000控制步。全部在第1000步原生timeout，保留随机化与原终止项，观测/动作/奖励/执行器力矩有限，前后策略state_dict哈希相同。腿最大观测力矩60N·m，轮最大14.386446N·m，均未超名义60/17上限。显式禁用了learn、update、backward和optimizer.step。
- Pyright未完成：系统Node12.22.9无法解析已安装pyright脚本的可选链语法，报SyntaxError。未为此修改Node或安装包。全仓make check/test未执行，不能据本次focused检查声称所有Black/其他模块通过。
- 除来源注释与等价的mesh文件名类型收窄修复之外，现行Wolf运行代码/参数未改变。未提交、推送或启动训练。上述回放不证明训练收敛、稳定行走、真实旧策略兼容或实机安全。

## 后续可选实验

已另增[软件速度保护实验](SPEED_GUARD_EXPERIMENT.md)。它不改变上述原版profile，也不将空载转速当作刚性状态上限；须显式选择。详见独立说明与原版/实验并列数据。
