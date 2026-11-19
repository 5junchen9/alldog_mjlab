# Black/Wolf从零训练v3审查快照

当前主线：随机初始化→基础平地→轴向起停→综合六域→六个专项分支。两种机器人各有权重，内部教师与学生共享actor/HIM；完整特权状态用于critic和HIM监督。

v3将原始相机健康与量程命中分开，噪声在有效量程内夹紧；Wolf课程状态免在线归一化。学生最多缓存50帧，后25帧预热与逐渐交接；Black学生JIT新增warm_hidden，合同显式升级。

JC与Windows各277项CPU测试通过，JC32组、Windows10组有限GPU检查通过。232条截短原始试次均未完成，成功数0。4096仅做4控制步0优化；没有正式长训或实机能力结论。

详见docs/boundary_review_20261119.md、docs/black.md、docs/wolf.md和evidence/boundary_review_20261119/。上一轮review_completion_20261119证据归属v2，不能代替当前v3结果。

公开快照只包含源码、架构说明和紧凑验证摘要。ASSET_REQUIREMENTS.json列出本地必需资产指纹，不上传团队模型、权重、完整原始轨迹或故障二进制。完整仿真、make check和证据复核请使用本地私有资产仓库。此分支供审计，未合并main。
