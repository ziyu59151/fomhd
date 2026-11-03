# FOMHD — C++ / CUDA

FOMHD 从 A0 开始以 **C++ 核心 + CUDA 后端**实现。优先标准库和成熟数值库，先正确性、数值 benchmark 与逐层算法验证，再考虑专门性能优化。完整实现级伪代码先于对应模块代码，并随代码版本共同维护。

本仓库目前保存工程约定和辅助环境配置，尚无求解器、完整伪代码或 CMake 构建目标。物理范围和数值合同以独立设计仓库的 DECISIONS（D045/D053–D057）、A0_IMPLEMENTATION_SPEC 和 A0_GLOBAL_DISCRETIZATION 为准；本机设计目录为 `/home/lxzeng/fomhd-design`，不是编译或运行依赖。

目录、语言/依赖、伪代码职责、版本管理和运行约定见 [开发说明](docs/development.md)。master 为主分支；达到相应 benchmark 基线后再建立 release/aN 维护分支和固定发布 tag。运行目录为 `/home/lxzeng/runs/fomhd/`，算例名前置、时间后置。

C++/CUDA 的构建由 CMake 管理；下述 pyproject/uv 只管理独立对照、输入生成和后处理使用的可选 Python 环境。保留已安装的 PyTorch，不把它作为 C++ 求解器必须经过的运行入口。

## 环境的几层

| 层 | 位置/工具 | 职责 |
|---|---|---|
| 服务器公共环境 | Linux、NVIDIA 驱动、Slurm、`/usr/local/cuda-13.3` | 管理硬件、分配 GPU、提供公共编译工具；由管理员维护 |
| 个人工具 | `~/.local/bin/uv`、`~/.local/share/uv/python/` | uv 管理个人 Python 解释器、包缓存和项目环境 |
| 设计项目 | `~/fomhd-design/.venv` | 文献、绘图、设计验证；有自己的 `pyproject.toml` 和 `uv.lock` |
| 源码项目的辅助环境 | 本目录的 `.venv` | 独立的 Python 3.12.16、PyTorch 2.14.1 + cu130、NumPy 2.5.4，用于对照和分析 |

uv 是管理工具，`.venv` 是它创建和维护的隔离环境，两者配合使用。项目依赖及来源记录在 `pyproject.toml`，精确解析结果记录在 `uv.lock`，Python 版本记录在 `.python-version`。`.venv` 是可重建的安装结果，不纳入版本管理。

PyTorch 及其依赖包带有 CUDA 13.0 用户态运行库。它们仍通过服务器的 NVIDIA 驱动访问 GPU，不会替换公共 CUDA 13.3 或驱动；预编译 PyTorch 算子不需要调用公共 `nvcc`。不要为此全局改写 `CUDA_HOME` 或 `LD_LIBRARY_PATH`。以后编译自定义 CUDA 扩展时，再单独确认编译器/库的兼容性。

## 辅助 Python 环境的使用与复现

以下命令检查辅助环境，不构建或运行 FOMHD 求解器。在本目录执行：

```bash
uv sync --locked
uv run --locked python -c 'import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda)'
```

`uv run` 自动选用项目 `.venv`，不必先手动激活。也可以使用 `.venv/bin/python`。在其他工作目录时使用 `uv run --project /home/lxzeng/projects/fomhd --locked python ...`。编辑器的 Python 解释器选择本目录的 `.venv/bin/python`。

GPU 程序必须在本人 Slurm 作业内运行。下面只是一个短时环境检查：

```bash
srun -p debug --immediate=10 --gres=gpu:a100:1 --cpus-per-task=2 --mem=2G --time=00:02:00 \
  .venv/bin/python -c 'import torch; x=torch.arange(16, device="cuda", dtype=torch.float64); assert x.sum().item()==120; print(torch.cuda.get_device_name())'
```

资源不能立即分配时停止本次检查，不修改设备权限。Python 环境隔离包依赖，并不隔离或授权硬件访问；GPU 的调度和访问仍由服务器控制。

## 当前验证与优化边界

数值 benchmark 需要参考解、误差/守恒指标和算例条件；基础性能测量记录规模、精度、设备、预热、重复次数和必要的 GPU 同步。这些工作不需要 NCU 性能计数器，也不要求达到加速比门槛。

`cuda-kernel-optimizer` 是后续明确要求优化时才启用的独立 skill，其 NCU 等严格门槛只适用于该优化流程。NVIDIA Compute Sanitizer 正确性排错能力继续保留。

环境冒烟通过只能证明指定设备上的这些库调用可运行，不能证明 FOMHD 模型、求解器、收敛性或生产性能。

2026-11-02 已通过 CPU 和 Slurm 分配的 RTX 5090（作业 `10626`）检查：FP64 矩阵乘法、归约、NumPy 互转、FFT、线性求解和基础计时。A100 因资源忙未立即分配，本次未验证。检查脚本与输出位于 `tmp/environment-check-20261102/`；GPU 检查只在本人 Slurm 分配内运行。
