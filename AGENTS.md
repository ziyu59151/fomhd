# FOMHD 源码协作规则

本仓库为 C++ 核心 + CUDA 后端。用户 2026-11-03 再次确认这一实现语言；Python/PyTorch 仅为辅助对照、输入处理和分析环境，不得因“成熟库优先”改写主求解器语言。开发约定见 [docs/development.md](docs/development.md)。

- 数学依据在同一用户的设计仓库 `fomhd-design`：DECISIONS、A0_IMPLEMENTATION_SPEC、A0_GLOBAL_DISCRETIZATION。此路径是当前工作区导航，不是编译/运行时依赖；缺设计文件时报告，不能凭空重建。
- 对应模块的完整实现级伪代码先行，未来在 docs/pseudocode 中与源代码和测试一起版本化。当前尚无完整伪代码或求解器，不用占位函数冒充实现。
- 模块接口明确单位、布局、阶段、读写/所有权、host/device 与失败语义。时间推进统一提交；诊断只读；共享物理通量和电流不重复构造。
- C++/CUDA 构建使用 CMake；依赖和版本必须显式。现有 pyproject.toml/uv.lock 只管理辅助环境。未安装的库不得默认为可用；首次构建后登记实际命令和版本。
- 普通用户操作；禁止 sudo、修改公用驱动/Toolkit/设备权限/Slurm 或其他用户文件。GPU 只在本人 Slurm 分配中运行。不得读取、输出或提交凭据。
- 主分支 master；任务分支 `codex/<task>`。benchmark 定稿后再建不可移动 tag 和 release/aN 维护分支。本次授权建立用户指定的 ziyu59151/fomhd 私有源码仓库及初始基线；后续远程写入按具体任务授权处理。
- Git 操作先验证根目录是本仓库；不在家目录父仓库暂存/提交。只提交明确列出的源码/配置/说明，不上传 .venv、build、运行结果、设计证据或文献。不 force push 或改写发布标签。
- 运行根为用户指定的 /home/lxzeng/runs；FOMHD 用 fomhd 子目录，算例名前置、时间后置。每次运行独立目录，保留实际输入、版本/二进制指纹、诊断和失败原文。
- 使用 skill-dispatch 选择 cpp-coding-standards、cpp-testing 与 NVIDIA 正确性排错技能；专门性能优化仍后置。C++/CUDA 常规实现不要求 NCU 计数器。
- 修改后运行作用域内检查并复核；当前没有求解器构建/测试目标，不报告空测试 PASS。不得重置、清理或覆盖无关用户工作。
