# Wolf本地接入

接入目录：`${MJLAB_ROOT}`。

来源：<https://github.com/N-W-wolf/alldog_mjlab>，提交`2913d3f449f8eb5b0505156c460ad2f61e4c49fd`。
资产来源：<https://github.com/N-W-wolf/XJTURoboCon_quadruped_assets/tree/main/URDF/wolf>。

## 文件与行为

- `src/mjlab/asset_zoo/robots/wolf/`：上游MJCF、执行器、默认姿态。
- `src/mjlab/tasks/velocity/config/wolf/`：四个任务、观测、动作、奖励、随机化和runner。
- `src/mjlab/rl/wolf_him/`：上游HIM算法，供HIM任务使用。
- `THIRD_PARTY_NOTICES.md`：保留上游第三方声明。
- `UPSTREAM_STATE.md`：上游状态原文快照，其测试结果与路径不代表本机验证。

修改上游包导入路径，适配当前mjlab目录，整理格式，并为默认关节位置的zip增加strict=True。保留上游控制和训练参数；没有安装上游对MuJoCo3.14的依赖覆盖及EPA补丁。本机仍使用MuJoCo3.11，rough已通过8步短仿真，长时间接触稳定性仍需验证。

Wolf策略顺序FL→FR→RL→RR，每腿hip/thigh/calf/foot，共16维；腿位置控制，轮速度控制。PPO观测53维，HIM输入6×53维；物理步长0.005秒、策略周期0.02秒。详细参数以代码为准。

## 资产与权重

已从用户下载的`XJTURoboCon_quadruped_assets-main.zip`提取17个原始STL，存放于：

`${MJLAB_ROOT}/src/mjlab/asset_zoo/robots/wolf/xmls/assets/`

文件名为`base_link.STL`及FL/FR/RL/RR各自的`_Link1.STL`至`_Link4.STL`。原始URDF保存在机器人目录的`urdf/`，其meshes通过相对符号链接复用STL。使用基线自带的训练MJCF，未用原资产XML覆盖基线碰撞配置。两个源压缩包保留在下载目录。

本次模型编译结果：nq=23、nv=22、nu=16、nmesh=17，总质量33.66860kg。两个压缩包均未发现训练checkpoint或导出策略，需单独取得与本次配置匹配的权重。

## 命令

STL已经补齐，可运行以下预览和训练命令；本次仅执行短仿真验证，尚未启动训练。

```bash
cd ${MJLAB_ROOT}
uv run --no-sync list-envs
uv run --no-sync play wolf-flat --agent zero --num-envs 1 --viewer native
uv run --no-sync train wolf-flat --env.scene.num-envs 1024
```

HIM任务：将`wolf-flat`替换为`wolf-flat-him`。复杂地形任务为`wolf-rough`与`wolf-rough-him`。zero为零动作预览，不表示训练后的行走能力。

拿到与此配置匹配的权重后：

```bash
uv run --no-sync play wolf-flat --checkpoint-file /绝对路径/model_XXX.pt --num-envs 1 --viewer native
```

## 本机验证

- 四任务注册、train/play帮助入口：通过。list-envs打印完整列表，原入口用环境数作为退出码，返回35不代表注册失败。
- CPU配置检查：5项通过，覆盖四任务动作顺序、轮速符号、配置相互独立、HIM动态类路径和53维镜像还原。
- Ruff格式与检查：通过。
- ty类型检查：未通过，上游迁入代码存在TensorDict/可选张量/runner mixin等标注问题，尚未完成类型清理。
- CPU模型编译：通过。
- CPU与CUDA短仿真：四任务各1环境×8步零动作，观测和奖励有限，全部通过。PPO actor=(1,53)，critic=(1,56)；HIM actor=(1,6,53)，critic=(1,56)，estimator_velocity=(1,3)。本项使用play配置，未覆盖训练随机化。
- PPO/HIM更新、权重回放、长期稳定行走、实机：未验证。

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest tests/test_wolf_integration.py -q
```

关闭第三方pytest自动插件加载，避免系统ROS Humble插件混入Python3.13环境；不修改ROS设置。
