# Wolf methods v2 源码回溯归档

这是 2026-11-09 的独立 **MjLab 源码 overlay**，不是对本仓库 `src/alldog_mjlab/` 插件的迁移，也不会改变插件主分支。归档保存实际源码，不只保存文件清单。

- Fork 基点：[5junchen9/alldog_mjlab 2913d3f](https://github.com/5junchen9/alldog_mjlab/commit/2913d3f449f8eb5b0505156c460ad2f61e4c49fd)。
- 重建框架：[mujocolab/mjlab 45b1e26](https://github.com/mujocolab/mjlab/commit/45b1e263ec1c3c0d2c3211b458135d80bb7030df)。不要把这些 `src/mjlab/` 文件直接覆盖到插件目录，也不要改用插件的 MjLab 1.6.0 依赖。
- 控制合同来自 alldog `2913d3f449f8eb5b0505156c460ad2f61e4c49fd`。机械资产历史基点为 `886015aef351f1e5517cd50e47628be35ec315ed`，详见 `overlay/docs/wolf/source_contract.json`。
- v1 原冻结快照保持不变。v2 在独立干净框架基线上整理 Wolf 文件，修复静态类型边界，不混入原工作区 Black 改动、个人注释或仿真补丁。

## 内容与排除项

`overlay/` 包含 Wolf PPO/HIM、方法模块、独立 BC/DAgger、机器人 XML/URDF、任务配置、三个入口脚本与文档。`framework.patch` 只包含 setuptools/TensorBoard 兼容锁定、Wolf changelog，以及已有的零速度范围 viewer 滑块修复。

`SOURCE_MANIFEST.json` 保存公开文件 SHA256、对应已验证源文件 SHA256、框架基点与本地测试文件 SHA256。文档中的个人机器路径已泛化，源码数值没有因此改变。

遵循 fork 的本地验证约定，`tests/` 源码不提交，也没有把测试改名塞进其它目录。127 项回归的测试文件保留在本地发布目录；公开归档提供其哈希和三个独立入口脚本。这个归档不包含权重、checkpoint、训练数据、STL 大文件、虚拟环境、机器启动器或私人日志。

## 重建到新的独立目录

将本归档路径设为 `ARCHIVE`。下面的命令只适用于新建目录，不要对有未提交修改的工作区直接覆盖。

```bash
export ARCHIVE=/absolute/path/to/archives/wolf_methods_20261109_v2
git clone https://github.com/mujocolab/mjlab.git mjlab-wolf-v2
cd mjlab-wolf-v2
git checkout --detach 45b1e263ec1c3c0d2c3211b458135d80bb7030df
git apply "$ARCHIVE/framework.patch"
cp -a "$ARCHIVE/overlay/." .
```

运行前，须从已保留的机械资产快照恢复 17 个原始 STL 到 `src/mjlab/asset_zoo/robots/wolf/xmls/assets/`，逐项校验 `source_contract.json` 中的 SHA256。URDF 的 `meshes` 链接指向 `../xmls/assets`。历史资产远端曾返回 404，不能声称仅靠当前公开 GitHub 链接就能恢复全部机械资产。

环境使用该框架的 `pyproject.toml` 和应用补丁后的 `uv.lock`。已验证的 GPU 运行时包括 Python 3.13、Torch 2.14.1+cu126、RSL-RL 5.5.1、MuJoCo/MuJoCo Warp 3.11.0、TensorDict 0.14.2。不要把插件主分支的锁文件与此框架混用；依赖版本与 CUDA 硬件相同也不保证训练逐位可复现。

恢复资产并准备兼容环境后，默认入口只打印计划：

```bash
uv run python scripts/train_wolf_methods.py --stage soft --num-envs 128 --rollout-steps 100 --seed 101
```

只有显式 `--train` 才启动正式训练。本文档不是启动训练的授权。`scripts/check_wolf_release.py` 是禁止 learn、optimizer update 与 backward 的有界仿真检查，不能用它证明训练有效。

## 本版本验证

- `make check` 完整通过：366 个文件无需重新格式化，Ruff、ty 通过，Pyright 0 errors / 0 warnings；使用已有 Node 24.21.0 与 Pyright 1.1.408。
- CPU 单元与合同回归：127 passed；测试文件哈希见 `SOURCE_MANIFEST.json`。
- Linux 与 Windows/WSL 的 CUDA 发布检查均通过。两端验证同一份 608 文件快照，清单 SHA256 为 `11e2df1a3bf5784c81e99149459ccbffed4ea695ae39b05525908e89405e45c7`；核心依赖版本一致。
- 两端均通过五组有界 profile：soft/full、hard/full、soft/no_him、soft/aux_off、hard/aux_off，每组 2 个环境、2 个控制步。强制 timeout、JIT 往返、未训练 checkpoint fixture 往返通过，actor/critic 权重保持不变。
- GPU 发布检查记录为 `training_started=false`、`optimizer_steps=0`、`backward_calls=0`。没有启动正式训练。
- 框架补丁在干净基点上重建的四个文件与已验证发布树逐字节相同；公开源码与发布树逐字节相同，只有文档机器路径已泛化。

完整训练、容量基准和完整 episode 的 DAgger 桥不属于本归档的发布通过声明；它们应按各自实验报告独立判断。

历史文档中的 17 条局部类型诊断、旧 Node 12 工具失败和早期测试数是当时记录，不代表 v2 最终状态。v2 使用已有 Node 24 运行 Pyright，没有通过关闭检查或新增 ignore 来隐藏错误。

## 能力边界

这些是论文方法的工程改编，不是完整论文复现。HIM、表征辅助学习和独立动作 DAgger 的职责不同；Methods 的在线地图表征教师不等于已经合格的动作专家。软件桥和未训练 fixture 通过不构成旧教师资格、已学步态、收敛、安全性或实机验收。正式 DAgger 仍要求独立的教师合同与目标域资格验证。

框架许可证见 `FRAMEWORK_LICENSE`，来源说明见 `overlay/docs/wolf/THIRD_PARTY_NOTICES.md`。
