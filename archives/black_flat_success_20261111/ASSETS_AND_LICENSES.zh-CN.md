# 资产、许可与依赖声明

本说明按冻结本地文件记录，不把整个归档统一改授一个许可证。

|内容|本地证据/来源|归档处理|
|---|---|---|
|Mjlab代码|`pyproject.toml`版本1.6.0、根`LICENSE` Apache-2.0；`preparation/mjlab_commit.txt`为原准备记录|保留全部原文与声明。当前源没有.git，本地记录不代替本窗口联网验证上游commit。|
|Black URDF、STL、D435外观/安装及控制出处|`N-W-wolf/super-dog@1b9f15de46eb75f6f253582b66fd51536548b4d9`；`preparation/black_source_manifest.json`|保留原URDF、mesh、来源清单。部分旧manifest的verified_local_copy=false是历史标记，本窗口另做训练冻结清单哈希核验，不篡改旧记录。|
|Black来源根许可|`src/mjlab/asset_zoo/robots/black_stock/source/LICENSE`，CC BY-NC-SA4.0，原版权标记Junfeng Long、Zirui Wang|保留署名、非商业和相同方式共享许可原文。Apache根声明不替代它。|
|legged_gym控制参考代码|Black source下`legged_gym/LICENSE`，BSD3条款，ETH Zurich/NVIDIA版权|保留单独许可及代码头。|
|derived/trunk.obj|由Black trunk.STL转换；`preparation/mesh_conversion.json`记录425043面、1275129顶点及三角形相同|转换用于绕开STL解码面数限制，无几何简化；保留衍生关系、来源许可、原STL和两份哈希。|
|Go1/G1/YAM等原树资产|完整src图含原任务注册导入与Go1碰撞配置，训练320文件清单也覆盖这些文件|原样保留原树。当前这些子树中未发现独立LICENSE文件，不据此推断其独立授权范围，也不宣称全部mesh都由Mjlab作者创作。若另做商业资产包，须逐项补齐来源授权核验。|
|模型权重和新增报告|用户工程产物，模型未附单独许可证|不替用户设定权重新许可；保留工程中已带的第三方声明。|

依赖版本以额外评估plan.runtime和checkpoint元数据中的实际运行版本为证据：Python3.13.15、torch2.14.1+cu126、MuJoCo3.11.0、mujoco-warp3.11.0、warp-lang1.14.0、rsl-rl-lib5.5.1、numpy2.5.1。完整源码pyproject和uv.lock一起保留，环境二进制/驱动不打包。没有联网安装依赖。

原动作和动力学定义保留：Black自身关节正负号/站姿、PD40/1.2、20N·m饱和、0.25动作比例、摩擦公式、0/1/2控制步延迟、0.005s物理步长和0.02s控制周期；保留质量惯性、自碰撞和D435几何。相机几何不是光学校准，模拟执行器不是实机参数辨识证据。

秘密排除采用明确文件白名单：未读取或打包用户环境配置、登录资料、Git凭证、浏览器数据、.env、.venv/cache。文本高置信模式扫描无发现，checkpoint infos另实读核对；`audit/SECRET_SCAN.json`给出扫描范围和限制。日志内原绝对路径、用户名、主机名和公开上游地址保留作来源追踪，不包含认证令牌。不能把模式扫描等同于对任意编码秘密的数学证明。
