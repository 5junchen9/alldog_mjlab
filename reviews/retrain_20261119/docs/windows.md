# Windows训练空间

Windows训练运行于WSL2的Ubuntu22.04。统一目录为`D:\RLTrain`，在WSL内对应`/mnt/d/RLTrain`。

|位置|用途|
|---|---|
|`retrain`|与JC同一版本的精简训练源码、模型资产和经验文档|
|`.venv`|唯一训练依赖环境|
|`python`|环境使用的Python解释器|
|`tools`|运行环境所需的uv工具|
|`ENV_MIGRATION_RECEIPT.json`|环境迁移与路径修正回执|

2026-11-19迁移后，实际运行的是Python3.13.15、Torch2.14.1+cu126、MuJoCo3.11.0、NumPy2.5.1。Torch、MuJoCo、NumPy、Warp、TensorDict、RSL-RL、Tyro、YAML导入通过；解释器和Torch文件都来自新目录。JC与Windows的Torch/CUDA版本不同，实际GPU验证应分别记录。

环境原先位于`D:\Wolf\env_20261109`。迁移保留原文件，修正54处解释器软链、入口脚本、激活文件与环境配置。文本通过新文件原子替换，避免修改与uv缓存硬链接共享的文件。第一次移动uv自身目录时受到Windows文件占用限制，uv退出后单独移动成功，已完成的环境和解释器移动没有重复执行。

新源码同步和CPU验证结果另存回执。旧`D:\Black`、`D:\Wolf`仅在新源码验证、关键故障摘要提炼和运行依赖核对完成后删除。Pictures、网盘下载、WindowsDocuments与其他个人资料不属于此次训练清理。

运行命令从WSL进入项目目录，以明确的环境执行：

```bash
cd /mnt/d/RLTrain/retrain
VIRTUAL_ENV=/mnt/d/RLTrain/.venv /mnt/d/RLTrain/tools/bin/uv run --active --no-sync --offline python -m black.run --help
```

新一轮训练从随机初始化开始。历史学生r800、旧教师与旧动作头不作为恢复入口；历史NaN的关键事实与尚未定位的第一异常物理子步保留在`history.md`和`provenance.json`。

## 本轮完成状态

新源码353文件逐项哈希通过；完整CPU反例233项通过，补装ONNX Runtime1.30.0后剩余1项导出检查通过。两模型编译、Ruff、默认计划和依赖检查通过。旧D:\Black与D:\Wolf已经按精确清单删除，删除后新环境重新导入和从零模型构造通过。本轮Windows没有做GPU优化验收。

后续从D:\RLTrain\retrain工作，代码改动用Git记录；不要把原有旧头PT交给新入口。跨机可以载入同代码/资产的教师权重，目标机需重新验收；运行库不同不接续原优化器。

## 历史v2同步

2026-11-19中断改动补完后，Windows源码与JC运行源码/资产指纹一致，258项CPU检查通过。该次模型合同为v2；视觉学生使用50帧记忆段，GPU子步探针通过`--debug-first-fault`显式启用。Windows没有进行本轮GPU优化验证。JC与Win的Torch/CUDA版本不同，完整学习器不能跨运行库续接。详细记录见`evidence/review_completion_20261119/WINDOWS_CHECK.json`。

## 当前v3

本轮修复相机健康与命中分离、Wolf课程状态归一化、深度边界噪声及50/25帧重叠记忆。两机同步与当前检查以`evidence/boundary_review_20261119/`为准，v2的258项结果不能代替v3结果。Windows实际训练环境为WSL2，当前可见内存约15.33GiB；不采用复审文字里未经核对的62GiB假设。跨机只加载同代码/资产的权重，完整优化器恢复仍要求相同运行库。

本轮v3在Windows完成277项CPU和10组GPU检查：Black/Wolf各自的教师更新与独立恢复、学生4→8次Adam与独立恢复、相机空视野及合成记忆边界。GPU更新均为32环境有限窗口，4096优化和长期收敛未验证。验证用16份权重已清理，保留紧凑JSON结果。
