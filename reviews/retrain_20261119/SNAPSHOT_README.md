# 当前审查快照：Black/Wolf共享actor v2

本目录同步本地实现0a5003e及证据提交47c6f78；原版对照为本PR前一提交da6efa6。当前教师从随机参数开始，顺序为平地基础→轴向起停→综合六域→六专项分支。模型合同已升到v2。

先读docs/review_completion_20261119.md的14项差异，再按evidence/review_completion_20261119/SOURCE_INDEX.json审查代码；验证汇总为VALIDATION.json。学生使用50张有效新帧的固定记忆段，每次Adam前重算burn-in。默认只做控制步物理有限性检查，debug-first-fault显式启用逐子步探针。

JC与Windows各258项CPU检查通过；34组有限GPU检查及7组真实原始采集/复评分完成。4096仅4步、0优化；174条截短attempt全部未完成、成功0，没有策略资格。正式长训未启动，旧自然NaN根因和实机能力未确认。

公开快照保存代码、说明及紧凑摘要，未上传机器人XML、URDF、网格、权重或故障二进制包。运行需要本地团队资产，哈希见ASSET_REQUIREMENTS.json；完整原始轨迹和故障包保存在本地Git。相关资产不提供公开下载。Windows仅验收CPU，本轮GPU优化未验证。
