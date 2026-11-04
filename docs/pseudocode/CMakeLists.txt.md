# 原生构建与依赖绑定

> Status: active; target `CMakeLists.txt` and required `cmake/` files. The native CPU and CUDA builds have now been executed; this pseudocode remains the build contract.  
> 依赖：[开发约定](../development.md)、[数值适配](src/numerics/dense.cpp.md)、[CUDA](src/cuda/backend.cu.md)、[I/O](src/io/npz.cpp.md)。最近同步：2026-11-04。

## A0-BUILD-01 — Targets

```text
cmake_minimum_required(VERSION 3.25)
project(FOMHD LANGUAGES C CXX)
require CXX_STANDARD=20,CXX_STANDARD_REQUIRED=ON,CXX_EXTENSIONS=OFF
option FOMHD_ENABLE_CUDA (explicit ON/OFF)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET fftw3)
pkg_check_modules(MKL REQUIRED IMPORTED_TARGET mkl-dynamic-lp64-seq)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
use pinned toml++,json,miniz sources from third_party
add_library(fomhd_numerics numerics/geometry sources)
target_link_libraries(fomhd_numerics PRIVATE PkgConfig::FFTW PkgConfig::MKL)
add_library(fomhd_core core/fluid/fields/boundary/time/diagnostics sources)
target_link_libraries(fomhd_core PUBLIC fomhd_numerics)
if CUDA:
  enable_language(CUDA); find_package(CUDAToolkit 13.0 REQUIRED)
  target_sources(fomhd_core PRIVATE src/cuda/backend.cu)
  define FOMHD_HAS_CUDA=1 on fomhd_core
  target_link_libraries(fomhd_core PRIVATE CUDA::cudart CUDA::cufft CUDA::cublas CUDA::cusolver)
  require CUDA_STANDARD=20, architectures=80;120, -lineinfo; no --use_fast_math
add_library(fomhd_miniz pinned C sources)
add_library(fomhd_codec src/io/npz.cpp src/io/hash.cpp)
target_link_libraries(fomhd_codec PRIVATE fomhd_miniz PUBLIC OpenSSL::Crypto)
add_library(fomhd_runtime src/io/config.cpp src/io/run.cpp)
target_link_libraries(fomhd_runtime PUBLIC fomhd_core fomhd_codec)
add_executable(fomhd app/fomhd.cpp); link fomhd_runtime
add_executable(fomhd_make_case tools/make_case.cpp); link fomhd_runtime
enable_testing(); add independent unit/integration/verification test executables from tests/
see tests/README.md for actual source-to-contract mapping
register individually named tests and CPU/CUDA labels, require nonzero test count
set CMAKE_EXPORT_COMPILE_COMMANDS=ON
if Python3 interpreter found:
  register NumPy interoperability, analyzer and real CLI tests
  run build_manifest.py after fomhd to retain the exact source snapshot,
    executable hash,compiler/build identity,library versions and design baseline hash
require that manifest for production CLI run creation
retain compile_commands.json,CMakeCache.txt and GPU job/device receipts with evidence
```

文件路径在实际 CMake 中写仓库根相对路径；本伪代码省略重复 `src/` 只为表示目标列表，不能把目录名当成实际源文件。Public interfaces are grouped in `include/fomhd/{types,geometry,numerics,operators,backend,physics,diagnostics,verification,io,npz,hash}.hpp`; they do not depend on app code. 不存在源文件时构建配置失败，不创建空目标充数。

## A0-BUILD-02 — NativeDependencies

| 用途 | 绑定与版本 | 当前证据/限制 |
|---|---|---|
| CPU FFT | FFTW 3.3.10；`fftw3.h`/pkg-config fftw3 | Built and tested; GPL-2.0-or-later; retain license obligations for distribution |
| CPU LU/SVD | 已装 MKL header 2020.0.4、包 2020.4.304-7；LP64 sequential LAPACKE | Built and tested; package and header versions are recorded separately |
| GPU 数值 | Toolkit 13.3；cuFFT 12.3.0.29、cuBLAS 13.6.0、cuSOLVER 12.2.6 | CUDA compilation and Slurm GPU execution verified; CMake minimum Toolkit is 13.0 |
| TOML | [toml++ v3.4.0](https://github.com/marzer/tomlplusplus/tree/v3.4.0) | MIT; bundled project-local source, built and tested |
| JSON | [nlohmann/json v3.12.0](https://github.com/nlohmann/json/tree/v3.12.0) | MIT；`parse`/SAX、`dump`，有限值由调用者校验 |
| ZIP/NPZ 容器 | [miniz 3.0.2](https://github.com/richgel999/miniz/tree/3.0.2) | MIT；reader/file_stat/extract_to_mem、writer/add_mem，明确无压缩输出 |
| SHA256 | 已装 OpenSSL 3.5.5，`OpenSSL::Crypto` / EVP | 只用于文件/配置内容指纹；无需外部 sha256sum 子进程 |

The build uses the checked-in `third_party/` source and system/user prefixes
resolved by pkg-config. Exact bundled contents and licenses are hashed in the
source snapshot; miniz also retains its download manifest. Configuration does
not fetch dependencies or implicitly borrow libraries from `/opt/miniconda3`.
Use the project uv Python explicitly for auxiliary tests and build manifests.

SHA 叶操作：`EVP_MD_CTX_new -> EVP_DigestInit_ex(EVP_sha256) -> EVP_DigestUpdate` 按固定大小字节块读取 -> `EVP_DigestFinal_ex -> EVP_MD_CTX_free`；new 检查非空，Init/Update/Final 返回值检查为 1，free 无返回值且由 RAII 保证执行。digest 长度必须 32，再编码小写 64 位 hex。输入内容 SHA 与语义配置 SHA 各用明确序列化数据，不哈希结构体未初始化 padding。

## A0-BUILD-03 — PresetsAndVerification

Current presets are `cpu-debug`, `cpu-release`, `cpu-sanitize`, and
`cuda-release`, each with its own build directory. A machine-local
`CMakeUserPresets.json` may supply compiler/dependency paths. Current CUDA
architectures 80 and 120 are explicit in CMake. Debug retains assertions and
symbols; Release retains scientific hard checks. Optimization flags do not
authorize a model or precision change.

Verification order used for A0: configure non-empty targets, build, run
`ctest --output-on-failure`, exercise input I/O and complete CLI runs, run
CUDA/Compute Sanitizer in the user Slurm allocation, and compare CPU/GPU
stages. Supplementary tests cover coupled/spatial MMS, source sensitivity,
linear correction, transactions, analyzer contracts and local theory.
Current commands and identities are retained in `tmp/a0-completion-20261104/`
and summarized in [A0 verification](../A0_VERIFICATION.md). CI, multi-GPU
execution, global/experimental validation and performance gates remain open.
