# 上传建议与体积

发布目录：`archive/black-flat-success-20261111`。

- 全目录51个文件，总计1,173,400,352bytes（1.0928GiB，含两份发布清单）。
- 原始重建载荷1564文件，1,793,611,206bytes；不含新增可移植包装和复核输出。
- source压缩流63,768,121bytes，一片；evidence压缩流1,057,142,427bytes，16片；压缩无损，完整流及逐片SHA256在BUNDLES.json。
- 每片最多67,108,864bytes（64MiB），全部发布文件小于90MiB。选定模型4,885,263bytes；保留的起点3999为39,291,725bytes，对照2999为4,883,573bytes，均不需模型分片。

目标为用户已有`5junchen9/alldog_mjlab`。使用新分支`archive/black-flat-success-20261111`和本目录新路径，不覆盖main或旧归档。父会话通过已连接GitHub工具上传，本窗口没有提交、推送或访问账号凭证。

完整证据约1GiB，建议报告、许可、源码清单、脚本和3个模型作为分支文件；17个完整二进制分片可作为同一新Release的固定资产。若父会话选择全部作为分支文件，必须确认其工具单次请求/二进制上传能力并分批核验；不能把64MiB分片一次塞入普通文本文件接口。若工具只能上传小文件，保持“上传未完成”状态，交给具备二进制上传能力的流程，禁止只上传报告就宣称完整归档已发布。

Release方案不改变分片内容：下载后按原相对路径放回`bundles/`；分支README需补充实际Release tag、每片固定下载链接、字节数和SHA256。上传后逐文件远端核验，从远端下载到全新目录并运行SHA256SUMS→rebuild.py→cpu_policy_check.py→verify_raw.py。保留远端commit/tag/资产ID、下载哈希及核验输出，才算完成上传验收。本窗口只有本地全量重建验证，没有远端验收证据。

不要为了压体积删除原始NPZ、统计反证、失败启动审计、训练日志或normalizer/动作元数据。已剔除venv/cache及多余checkpoint。没有删除任何源文件；备份候选在black_delivery/CLEANUP_CANDIDATES.json。无验证等价副本的不可恢复删除逐项确认。
