# Black 平地全向教师：前进、后退、转向与侧移

2026-11-10 新训练代码的源码归档。一个特权教师、一个 PPO 和一套权重，同时采样四类平地运动，并包含组合运动与停步。它不是四个独立教师，也不是已经通过步态验收的模型。

## 范围与基点

- 目标 fork 基点：[5junchen9/alldog_mjlab@2913d3f](https://github.com/5junchen9/alldog_mjlab/commit/2913d3f449f8eb5b0505156c460ad2f61e4c49fd)。本分支只新增本归档目录，不改 main、现有 Black/Wolf 或任何生产入口。
- 重建框架：[mujocolab/mjlab@45b1e263ec1c3c0d2c3211b458135d80bb7030df](https://github.com/mujocolab/mjlab/commit/45b1e263ec1c3c0d2c3211b458135d80bb7030df)。这是固定源码基点，不能用浮动 main 或本仓库的插件依赖代替。
- Black 模型与摩擦/延迟行为来源：[N-W-wolf/super-dog@1b9f15de46eb75f6f253582b66fd51536548b4d9](https://github.com/N-W-wolf/super-dog/tree/1b9f15de46eb75f6f253582b66fd51536548b4d9)。不使用 BlackW。

`overlay/` 保存实际新增的 11 个 Python 文件：机器人适配、Black Stock 配置、摩擦/延迟适配、特权教师与全向命令。Stock 的 Rough 工厂及摩擦/延迟对照注册是已有公共前置代码，不表示本次训练或验证了 Rough。

排除模型权重、checkpoint、训练日志、训练数据、网格、URDF、派生 OBJ、虚拟环境、机器路径启动器、旧 Black/Wolf 修改以及 HIM/DAgger 学生代码。遵守目标仓库约定，验证测试保留本地，不提交或改名藏入其它目录；哈希记录在 `SOURCE_MANIFEST.json`。

## 四类专项与混合采样

任务：`Mjlab-Velocity-Flat-Black-Teacher-Omni`。

| 运动类别 | 采样比例 | 指令 |
|---|---:|---|
| 前进 | 20% | 正 vx，其余为零 |
| 后退 | 20% | 负 vx，其余为零 |
| 转向 | 左右各 10% | 正/负 yaw，其余为零 |
| 侧移 | 左右各 10% | 正/负 vy，其余为零 |
| 组合运动 | 15% | 保留多轴速度采样 |
| 停步 | 5% | 三轴为零 |

指令每 3–8 秒重采样；vx/vy 范围 ±1 m/s，yaw 范围 ±0.5 rad/s。纯轴指令的绝对值至少为 0.1。关闭 heading 覆盖和速度范围扩展课程；`CommandCoverage/*` 记录实际送入策略的八类命令比例。

教师 actor/critic 都是 72 维单帧真值，关闭观测噪声，保留环境随机化与各自 running normalization。观测依次为：机身线速度、角速度、重力方向、关节位置误差、关节速度、上一步策略动作、三轴指令、足端高度、腾空时间、接触状态和接触力。接触力使用 `sign(F)*log(1+abs(F))`。

启用显式 PD 与 Black 电机摩擦 `0.35*tanh(3*dq)+0.1*dq`；减摩擦后饱和到 ±20 N·m。动作缩放 0.25 rad，KP/KD=40/1.2。20 ms 控制周期，5 ms 物理子步。每次 reset 按环境采样 0/1/2 控制步延迟，默认保留跨回合动作队列；策略动作观测仍是当前原始动作。全部 13 项平地奖励配置在 `black_stock/env_cfgs.py` 中可见。

## 重要的不兼容边界

本归档是 `src/mjlab/` 的独立源码 overlay，**不是** `src/alldog_mjlab/` 插件迁移。不要覆盖插件文件，不要接入已有任务入口，也不要把当前教师权重当作原插件可部署策略。

- 本教师使用 72 维特权输入；插件 Black PPO 是 45 维输入。
- 本适配的关节/动作顺序为 FL→FR→RR→RL；插件 policy 顺序为 FL→FR→RL→RR。
- 本教师沿用此固定 mjlab 源码的 Go1 PPO/runner 和奖励，而非插件的 Black 奖励/部署合同。
- 摩擦开启同时采用显式 PD，与关闭时的内置伺服不同；差异不能全部归因于摩擦。
- MuJoCo 与 IsaacGym 接触、自碰撞及控制实现不等价。没有实机标定或 sim2real 兼容性保证。

上述边界有意保留当前训练行为；任何后续插件迁移都需独立验证观测、动作、归一化和 joint mapping。

## 在新的独立目录中重建

下面操作只能用于新目录，不要在已有未提交工作区覆盖执行。

```bash
export ARCHIVE=/absolute/path/to/archives/black_flat_omni_20261110
git clone https://github.com/mujocolab/mjlab.git mjlab-black-omni
cd mjlab-black-omni
git checkout --detach 45b1e263ec1c3c0d2c3211b458135d80bb7030df
cp -a "$ARCHIVE/overlay/." .
```

本归档不是开箱即跑的模型包。先从有权使用的本地来源恢复 `ASSET_REQUIREMENTS.json` 所列 URDF、原始网格及派生 trunk.obj，并逐项校验 SHA256。网格缺失或版本不匹配时不能启动。trunk.STL 超过 MuJoCo STL 面数上限，派生 OBJ 保留全部 425043 个三角形；不可拿插件中另一个 trunk 网格替换。公开归档不提供资产字节，也不宣称已经确认其再分发权利。

依赖使用上述框架提交自身的 `pyproject.toml` 和 `uv.lock`。CPU 发布验证环境的精确版本记录在 `VERIFICATION.json`。现有训练环境另为 Python 3.13.15、Torch 2.14.1+cu126、RSL-RL 5.5.1、MuJoCo/MuJoCo-Warp 3.11.0、Warp 1.14.0、TensorDict 0.14.2、TensorBoard 2.20.0、setuptools 80.10.2、protobuf 6.33.6；不是目标插件使用的 3.14 后端。

可在独立环境按固定框架锁文件安装，并保留 TensorBoard 所需的 setuptools 版本：

```bash
uv sync --frozen --extra cu126 --group dev
uv pip install 'setuptools==80.10.2'
```

这里是重建指引，本次发布未下载或升级依赖、未验证全新安装。现有训练的 Torch 补丁版本与框架原锁文件不完全相同；需要复现实验运行时，应同时对齐上述训练版本及 CUDA 驱动，不能仅因基点相同就声称逐位复现。

恢复资产与依赖后，显式使用此新目录源码：

```bash
PYTHONPATH="$PWD/src" uv run --no-sync python -m mjlab.scripts.train \
  Mjlab-Velocity-Flat-Black-Teacher-Omni \
  --env.scene.num-envs 4096 --agent.max-iterations 3000 \
  --agent.run-name omni_4096_3000 --agent.seed 42
```

这是原训练设置的复现命令，本次归档发布没有另启正式训练。先在自己的硬件上确认容量，再执行大并行训练。

## 发布验证

- 515 个原版框架文件与固定基点逐字节一致。
- 完整 `make check` 通过：Ruff、ty、Pyright；Pyright 0 errors / 0 warnings，使用已有 Node 24.21.0。
- 9 项 CPU 回归通过，覆盖摩擦公式、饱和、延迟、局部 reset、队列保留/清理、IMU/fusion，以及 4096 个命令的八类覆盖与范围。
- Omni CPU 2 环境×24 步的一次 PPO 更新通过；奖励、观测、损失、参数均有限，参数实际更新；checkpoint、优化器、归一化和计数恢复通过，相同输入的恢复动作最大误差为 0；教师 ONNX 导出通过。
- 发布副本只修正一个可变属性覆写的类型声明并格式化，用 `cast` 保留同一配置对象；所有 11 个文件移除这类类型语法后与训练源码 AST 一致。原训练目录和其进程未改动。
- CPU PASS。CUDA 发布回归 SKIPPED，本次未占用训练 GPU。未运行完整项目 `make test`，未作训练收敛、各方向步态、摔倒/打滑或实机验收声明。

命令覆盖和一次 PPO 更新证明软件链路可运行，不等于四类运动已经学好。当前正在训练的运行结果需单独评估，未混入此源码发布报告。

## 许可与来源

MjLab 派生配置/实现遵循所附 `FRAMEWORK_LICENSE`（Apache-2.0）。Black 摩擦/延迟行为参考固定 super-dog 的 legged_gym 代码，相关 BSD-3-Clause 文本保留在 `BLACK_CONTROL_LICENSE`。super-dog 根仓库另有 CC BY-NC-SA 4.0，资产没有已确认的独立许可；本归档不重新授权这些资产，也不声明整体可无条件商用。
