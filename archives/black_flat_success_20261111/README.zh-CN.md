# Black 平地综合成功版 · 2026-11-11

选定 `model_33999.pt`，SHA256：
`55be4aff5a54232a45f27e6df567d0de49d53e4aeb9096c1e96c307b343749f7`。

原配对最终留出0/896跌倒；新增10个评估种子为1/4480跌倒，原Omni为80/4480。
唯一残余失败来自高速边界组合骤然反向。详见完整报告，不能宣称全场景零跌倒。
该模型是72维真值仿真教师，12维动作，尚无实机验收。

- 完整平地报告：`BLACK_FLAT_SUCCESS_REPORT.zh-CN.md`。
- 完整越野方案：`BLACK_ROUGH_TRAINING_PLAN.zh-CN.md`；仅方案，未启动越野训练。
- 可浏览代码：`code/`；实际可运行目录由下述命令重建，资产包含在源码包内。
- 冻结源码和资产：`bundles/source.tar.gz.part*`，可逆分片。
- 必要模型：最终33999、精确续训起点3999、原Omni2999；3999也采用可逆分片。
- 全部原始文件索引：`RECONSTRUCTED_FILES.json`；GitHub核心重建范围见
  `CORE_RESTORED_FILES.json`，原始轨迹与大体积日志留在JC，未删除。
- 逐场景指标、实际配置、验证、留出与10种子报告数据：`evidence_core.tar.gz`。

## 重建与检查

先完整下载本目录，保持分片相对路径；使用Python3.13以上与uv：

```sh
uv run --no-project --python 3.13 rebuild.py --destination /tmp/black-flat-success
```

脚本先逐片校验、重组源码/资产和模型，再逐项验证1288个原始冻结文件。
默认随后应用`publication_overlay/`：修正测试夹具类型、外部参考文件的扫描范围，
以及执行器配置类型收窄的静态检查注解；生产Python AST不变，模型字节不变。
`--exact-frozen`可重建未应用覆盖层的原始训练快照。
原始快照包含旧测试的类型诊断，不能把覆盖层通过结果归于未修改的快照。

`code/`提供应用覆盖层后的可浏览版本；模型训练和评估记录保持原样。
依赖和资产许可见`ASSETS_AND_LICENSES.zh-CN.md`、`uv.lock`及资产内来源记录。
完整`make check`在隔离发布副本通过，修正所涉及的67项CPU回归通过；
成熟teacherfree的77项CPU测试也通过，未运行全仓库`make test`、未建PR。
原始完整包和此轻量包的重建及actor/critic严格加载分别核验；详见audit日志。

配置已有运行环境后可进行CPU加载检查：

```sh
cd /tmp/black-flat-success
BLACK_UV="$(command -v uv)" BLACK_PYTHON=/path/to/existing/venv/bin/python \
  bash run_local.sh cpu_policy_check.py
```

入口默认不会启动训练。原`run.sh`保存了JC路径供来源审计；异机使用`run_local.sh`。

## 原始证据保留范围

GitHub核心包省略276个原始轨迹或大日志文件，完整原字节清单和SHA256仍随包发布。
完整1.09GiB归档在JC：
`/home/wujuncheng/Documents/Codex/2026-11-11/task/black_delivery/archive/black-flat-success-20261111`。
原始训练与全部评估证据仍在
`/home/wujuncheng/rl_train/black_mjlab_stock_20261110/logs/black_teacherfree/`。
`verify_raw.py`需要这些本地原始NPZ；核心包不假称包含它们。
已在本机独立复算全部40份NPZ和8960个trial，结果见audit文件。
完整原始数据没有验证等价远端副本，因此不在可删除范围内。
