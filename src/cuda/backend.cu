#include "fomhd/backend.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cusolverDn.h>
#include <limits>
#include <string>
#include <utility>

namespace fomhd {
namespace {
using Z = cuDoubleComplex;
static_assert(sizeof(Z) == sizeof(Complex));
void check(cudaError_t status, const char *call) {
  if (status != cudaSuccess)
    throw NumericalError(std::string(call) + ": " + cudaGetErrorString(status));
}
template <class T> void library(T status, const char *call) {
  if (int(status) != 0)
    throw NumericalError(std::string(call) +
                         ": status=" + std::to_string(int(status)));
}
int dimension(std::size_t n) {
  if (n > INT_MAX)
    throw NumericalError("CUDA library dimension exceeds INT_MAX");
  return static_cast<int>(n);
}
unsigned blocks(std::size_t n) {
  const auto b = n / 256 + (n % 256 != 0);
  if (!b || b > INT_MAX)
    throw NumericalError("invalid CUDA launch size");
  return static_cast<unsigned>(b);
}
struct Memory {
  std::size_t cap{}, live{}, peak{};
};
template <class T> class Buffer {
public:
  Buffer(Memory &memory, std::size_t count)
      : memory_(memory),
        bytes_(checked_product(count, sizeof(T), "device buffer")) {
    if (bytes_ > memory_.cap - memory_.live)
      throw NumericalError("explicit device memory budget exceeded");
    if (bytes_)
      check(cudaMalloc(reinterpret_cast<void **>(&data_), bytes_),
            "cudaMalloc");
    memory_.live += bytes_;
    memory_.peak = std::max(memory_.peak, memory_.live);
  }
  ~Buffer() {
    if (data_)
      cudaFree(data_);
    memory_.live -= bytes_;
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  T *get() { return data_; }
  const T *get() const { return data_; }
  std::size_t bytes() const { return bytes_; }

private:
  Memory &memory_;
  std::size_t bytes_{};
  T *data_{};
};
struct Runtime {
  cudaStream_t stream{};
  cublasHandle_t blas{};
  cusolverDnHandle_t solver{};
  ~Runtime() {
    if (stream)
      cudaStreamSynchronize(stream);
    if (solver)
      cusolverDnDestroy(solver);
    if (blas)
      cublasDestroy(blas);
    if (stream)
      cudaStreamDestroy(stream);
  }
};
struct Plan {
  cufftHandle handle{};
  Plan() { library(cufftCreate(&handle), "cufftCreate"); }
  ~Plan() {
    if (handle)
      cufftDestroy(handle);
  }
  Plan(const Plan &) = delete;
  Plan &operator=(const Plan &) = delete;
};
__global__ void pad_kernel(const Z *input, Z *out, std::size_t E, int K,
                           int N) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(N) * E)
    return;
  const int bin = int(i / E), k = bin <= N / 2 ? bin : bin - N;
  out[i] = (k >= -K && k <= K) ? input[std::size_t(k + K) * E + i % E]
                               : make_cuDoubleComplex(0, 0);
}
__global__ void gather_kernel(const Z *input, Z *out, std::size_t E, int K,
                              int N) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(2 * K + 1) * E)
    return;
  int k = int(i / E) - K, bin = k < 0 ? k + N : k;
  Z value = input[std::size_t(bin) * E + i % E];
  out[i] = make_cuDoubleComplex(value.x / N, value.y / N);
}
__global__ void derivative_kernel(Z *values, std::size_t E, int N) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= std::size_t(N) * E)
    return;
  const int bin = int(i / E), k = bin <= N / 2 ? bin : bin - N;
  const double scale = double(k) / N;
  const Z v = values[i];
  values[i] = make_cuDoubleComplex(-scale * v.y, scale * v.x);
}
__global__ void axpy_kernel(const Z *a, const Z *b, Z *out, std::size_t n,
                            double alpha) {
  const std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] =
        make_cuDoubleComplex(a[i].x + alpha * b[i].x, a[i].y + alpha * b[i].y);
}

class CudaCompute final : public Compute {
public:
  CudaCompute(int ordinal, std::size_t cap) : memory_{cap} {
    if (!std::getenv("SLURM_JOB_ID"))
      throw NumericalError("CUDA execution requires a Slurm GPU allocation");
    if (cap == 0)
      throw NumericalError("device memory cap must be positive");
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (ordinal < 0 || ordinal >= count)
      throw NumericalError("device ordinal outside allocated visible devices");
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    check(cudaGetDeviceProperties(&device_, ordinal),
          "cudaGetDeviceProperties");
    check(cudaStreamCreateWithFlags(&runtime_.stream, cudaStreamNonBlocking),
          "cudaStreamCreateWithFlags");
    library(cublasCreate(&runtime_.blas), "cublasCreate");
    library(cublasSetStream(runtime_.blas, runtime_.stream), "cublasSetStream");
    library(cublasSetPointerMode(runtime_.blas, CUBLAS_POINTER_MODE_HOST),
            "cublasSetPointerMode");
    library(cusolverDnCreate(&runtime_.solver), "cusolverDnCreate");
    library(cusolverDnSetStream(runtime_.solver, runtime_.stream),
            "cusolverDnSetStream");
    // A fixed explicit BLAS workspace avoids an unbounded per-handle pool.
    blas_workspace_ = std::make_unique<Buffer<std::byte>>(memory_, 4U << 20);
    library(cublasSetWorkspace(runtime_.blas, blas_workspace_->get(),
                               blas_workspace_->bytes()),
            "cublasSetWorkspace");
  }
  ~CudaCompute() override {
    if (runtime_.stream)
      cudaStreamSynchronize(runtime_.stream);
  }
  std::string name() const override {
    return std::string("cuda: ") + device_.name;
  }
  void synchronize() override {
    check(cudaStreamSynchronize(runtime_.stream), "cudaStreamSynchronize");
  }
  Vec mv(const Matrix &a, const Vec &x) override {
    if (a.cols != x.size() || !is_finite(a) || !is_finite(x))
      throw NumericalError("invalid CUDA matrix-vector input");
    ++matrix_calls;
    Vec result(a.rows);
    if (!a.rows || !a.cols)
      return result;
    const int m = dimension(a.rows), n = dimension(a.cols);
    Buffer<Z> da(memory_, a.data.size()), dx(memory_, x.size()),
        dy(memory_, result.size());
    upload(da, a.data);
    upload(dx, x);
    const Z one{1, 0}, zero{0, 0};
    library(cublasZgemv(runtime_.blas, CUBLAS_OP_N, m, n, &one, da.get(), m,
                        dx.get(), 1, &zero, dy.get(), 1),
            "cublasZgemv");
    download(dy, result);
    return result;
  }
  Matrix mm(const Matrix &a, const Matrix &b) override {
    if (a.cols != b.rows || !is_finite(a) || !is_finite(b))
      throw NumericalError("invalid CUDA matrix-matrix input");
    ++matrix_calls;
    Matrix result(a.rows, b.cols);
    if (!a.rows || !a.cols || !b.cols)
      return result;
    const int m = dimension(a.rows), n = dimension(b.cols),
              k = dimension(a.cols);
    Buffer<Z> da(memory_, a.data.size()), db(memory_, b.data.size()),
        dy(memory_, result.data.size());
    upload(da, a.data);
    upload(db, b.data);
    const Z one{1, 0}, zero{0, 0};
    library(cublasZgemm(runtime_.blas, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one,
                        da.get(), m, db.get(), k, &zero, dy.get(), m),
            "cublasZgemm");
    download(dy, result.data);
    return result;
  }
  Vec axpy(const Vec &a, const Vec &b, double alpha) override {
    if (a.size() != b.size() || !is_finite(a) || !is_finite(b) ||
        !std::isfinite(alpha))
      throw NumericalError("invalid CUDA AXPY input");
    if (a.empty())
      return {};
    Buffer<Z> da(memory_, a.size()), db(memory_, b.size()),
        dy(memory_, a.size());
    upload(da, a);
    upload(db, b);
    axpy_kernel<<<blocks(a.size()), 256, 0, runtime_.stream>>>(
        da.get(), db.get(), dy.get(), a.size(), alpha);
    launch_check();
    Vec result(a.size());
    download(dy, result);
    return result;
  }
  Vec inverse_fft(const Vec &x, std::size_t E, int K, int N) override {
    return fft(x, E, K, N, 0);
  }
  Vec forward_fft(const Vec &x, std::size_t E, int K, int N) override {
    return fft(x, E, K, N, 1);
  }
  Vec derivative(const Vec &x, std::size_t E, int N) override {
    return fft(x, E, 0, N, 2);
  }
  LinearResult linear(const Matrix &a, const Vec &b, const RealVec &rows,
                      const RealVec &cols, double rank,
                      double tolerance) override {
    if (!a.rows || a.rows != a.cols || b.size() != a.rows ||
        rows.size() != a.rows || cols.size() != a.cols || !is_finite(a) ||
        !is_finite(b) || !(rank > 0 && rank < 1) || !(tolerance > 0) ||
        !std::isfinite(tolerance))
      throw NumericalError("invalid CUDA checked linear solve input");
    ++solve_calls;
    const int n = dimension(a.rows);
    for (double x : rows)
      if (!(x > 0) || !std::isfinite(x))
        throw NumericalError("invalid CUDA row scale");
    for (double x : cols)
      if (!(x > 0) || !std::isfinite(x))
        throw NumericalError("invalid CUDA column scale");
    Matrix scaled_matrix(n, n);
    Vec scaled_rhs(n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        scaled_matrix(i, j) = a(i, j) * cols[j] / rows[i];
    for (int i = 0; i < n; ++i)
      scaled_rhs[i] = b[i] / rows[i];
    if (!is_finite(scaled_matrix) || !is_finite(scaled_rhs))
      throw NumericalError("CUDA scaled system is not finite");
    LinearResult result;
    result.singular_values = svd(scaled_matrix);
    if (result.singular_values.front() == 0 ||
        result.singular_values.back() <= rank * result.singular_values.front())
      throw NumericalError("scaled CUDA system is singular or ill-conditioned");
    Vec y(n);
    {
      Buffer<Z> da(memory_, a.data.size()), db(memory_, b.size());
      Buffer<int> piv(memory_, n), info(memory_, 1);
      upload(da, scaled_matrix.data);
      upload(db, scaled_rhs);
      int work_count = 0;
      library(cusolverDnZgetrf_bufferSize(runtime_.solver, n, n, da.get(), n,
                                          &work_count),
              "cusolverDnZgetrf_bufferSize");
      if (work_count < 0)
        throw NumericalError("invalid LU workspace size");
      Buffer<Z> work(memory_, work_count);
      library(cusolverDnZgetrf(runtime_.solver, n, n, da.get(), n, work.get(),
                               piv.get(), info.get()),
              "cusolverDnZgetrf");
      check_info(info, "cuSOLVER LU factorization");
      library(cusolverDnZgetrs(runtime_.solver, CUBLAS_OP_N, n, 1, da.get(), n,
                               piv.get(), db.get(), n, info.get()),
              "cusolverDnZgetrs");
      check_info(info, "cuSOLVER LU solve");
      download(db, y);
      double previous_error = std::numeric_limits<double>::infinity();
      constexpr std::size_t max_refinement_steps = 5;
      for (;;) {
        const Vec sr = subtract(mv(scaled_matrix, y), scaled_rhs);
        result.scaled_residual = norm(sr);
        result.backward_error = 0;
        for (int i = 0; i < n; ++i) {
          long double scale = std::abs(scaled_rhs[i]);
          for (int j = 0; j < n; ++j)
            scale += static_cast<long double>(std::abs(scaled_matrix(i, j))) *
                     std::abs(y[j]);
          result.backward_error = std::max(
              result.backward_error,
              scale > 0 ? double(std::abs(sr[i]) / scale) : std::abs(sr[i]));
        }
        if (!std::isfinite(result.backward_error))
          throw NumericalError("CUDA refinement backward error is not finite");
        if (result.backward_error <= tolerance)
          break;
        if (result.refinement_steps == max_refinement_steps ||
            result.backward_error >= previous_error)
          throw NumericalError(
              "CUDA componentwise backward error exceeded its tolerance");
        previous_error = result.backward_error;
        upload(db, scaled(sr, -1.0));
        library(cusolverDnZgetrs(runtime_.solver, CUBLAS_OP_N, n, 1, da.get(), n,
                                 piv.get(), db.get(), n, info.get()),
                "cusolverDnZgetrs refinement");
        check_info(info, "cuSOLVER LU refinement solve");
        Vec correction(n);
        download(db, correction);
        Vec next = axpy(y, correction, 1.0);
        if (!is_finite(next) || next == y)
          throw NumericalError("CUDA refinement is non-finite or stagnated");
        y = std::move(next);
        ++result.refinement_steps;
      }
    }
    result.x.resize(n);
    for (int i = 0; i < n; ++i)
      result.x[i] = y[i] * cols[i];
    result.residual = subtract(mv(a, result.x), b);
    result.row_scale = rows;
    result.column_scale = cols;
    if (!is_finite(result.x) || !is_finite(result.residual) ||
        !std::isfinite(result.scaled_residual) ||
        !std::isfinite(result.backward_error) ||
        result.backward_error > tolerance)
      throw NumericalError(
          "CUDA componentwise backward error exceeded its tolerance");
    return result;
  }

private:
  Memory memory_;
  Runtime runtime_;
  std::unique_ptr<Buffer<std::byte>> blas_workspace_;
  cudaDeviceProp device_{};
  void launch_check() { check(cudaPeekAtLastError(), "CUDA kernel launch"); }
  template <class T, class U>
  void upload(Buffer<T> &d, const std::vector<U> &h) {
    if (d.bytes() != h.size() * sizeof(U))
      throw NumericalError("upload shape mismatch");
    check(cudaMemcpyAsync(d.get(), h.data(), d.bytes(), cudaMemcpyHostToDevice,
                          runtime_.stream),
          "upload");
    h2d_bytes += d.bytes();
    synchronize();
  }
  template <class T, class U>
  void download(const Buffer<T> &d, std::vector<U> &h) {
    if (d.bytes() != h.size() * sizeof(U))
      throw NumericalError("download shape mismatch");
    check(cudaMemcpyAsync(h.data(), d.get(), d.bytes(), cudaMemcpyDeviceToHost,
                          runtime_.stream),
          "download");
    d2h_bytes += d.bytes();
    synchronize();
  }
  void check_info(const Buffer<int> &info, const char *call) {
    std::vector<int> value(1);
    download(info, value);
    if (value[0] != 0)
      throw NumericalError(std::string(call) +
                           ": info=" + std::to_string(value[0]));
  }
  RealVec svd(const Matrix &a) {
    const int n = dimension(a.rows);
    int count = 0;
    library(cusolverDnZgesvd_bufferSize(runtime_.solver, n, n, &count),
            "cusolverDnZgesvd_bufferSize");
    if (count < 0)
      throw NumericalError("invalid SVD workspace size");
    Buffer<Z> da(memory_, a.data.size()), work(memory_, count),
        dummy_u(memory_, 1), dummy_v(memory_, 1);
    Buffer<double> values(memory_, n), rwork(memory_, std::max(1, n - 1));
    Buffer<int> info(memory_, 1);
    upload(da, a.data);
    // cuSOLVER's host-side SVD transfers include unused workspace entries.
    // Initialize the queried workspaces so no indeterminate bytes cross the
    // API.
    check(cudaMemsetAsync(work.get(), 0, work.bytes(), runtime_.stream),
          "initialize SVD workspace");
    check(cudaMemsetAsync(rwork.get(), 0, rwork.bytes(), runtime_.stream),
          "initialize SVD real workspace");
    library(cusolverDnZgesvd(runtime_.solver, 'N', 'N', n, n, da.get(), n,
                             values.get(), dummy_u.get(), 1, dummy_v.get(), 1,
                             work.get(), count, rwork.get(), info.get()),
            "cusolverDnZgesvd");
    check_info(info, "cuSOLVER SVD");
    RealVec result(n);
    download(values, result);
    for (int i = 0; i < n; ++i)
      if (!std::isfinite(result[i]) || result[i] < 0 ||
          (i && result[i] > result[i - 1]))
        throw NumericalError("invalid CUDA singular value spectrum");
    return result;
  }
  Vec fft(const Vec &x, std::size_t E, int K, int N, int kind) {
    (void)Fourier(K, N);
    if (!E || !is_finite(x))
      throw NumericalError("invalid CUDA Fourier input");
    const int entities = dimension(E);
    const auto full = checked_product(std::size_t(N), E, "CUDA FFT"),
               retained =
                   checked_product(std::size_t(2 * K + 1), E, "CUDA modes");
    if (x.size() != (kind == 0 ? retained : full))
      throw NumericalError("CUDA Fourier extent mismatch");
    ++fft_calls;
    Buffer<Z> input(memory_, x.size()), bins(memory_, full),
        output(memory_, kind == 1 ? retained : full);
    upload(input, x);
    Plan plan;
    library(cufftSetAutoAllocation(plan.handle, 0), "cufftSetAutoAllocation");
    int length = N, embed = N;
    std::size_t work_bytes = 0;
    library(cufftMakePlanMany(plan.handle, 1, &length, &embed, entities, 1,
                              &embed, entities, 1, CUFFT_Z2Z, entities,
                              &work_bytes),
            "cufftMakePlanMany");
    Buffer<std::byte> work(memory_, work_bytes);
    library(cufftSetWorkArea(plan.handle, work.get()), "cufftSetWorkArea");
    library(cufftSetStream(plan.handle, runtime_.stream), "cufftSetStream");
    if (kind == 0) {
      pad_kernel<<<blocks(full), 256, 0, runtime_.stream>>>(
          input.get(), bins.get(), E, K, N);
      launch_check();
      library(
          cufftExecZ2Z(plan.handle, bins.get(), output.get(), CUFFT_INVERSE),
          "cufftExecZ2Z inverse");
    } else {
      library(cufftExecZ2Z(plan.handle, input.get(), bins.get(), CUFFT_FORWARD),
              "cufftExecZ2Z forward");
      if (kind == 1) {
        gather_kernel<<<blocks(retained), 256, 0, runtime_.stream>>>(
            bins.get(), output.get(), E, K, N);
        launch_check();
      } else {
        derivative_kernel<<<blocks(full), 256, 0, runtime_.stream>>>(bins.get(),
                                                                     E, N);
        launch_check();
        library(
            cufftExecZ2Z(plan.handle, bins.get(), output.get(), CUFFT_INVERSE),
            "cufftExecZ2Z derivative");
      }
    }
    Vec result(kind == 1 ? retained : full);
    download(output, result);
    if (!is_finite(result))
      throw NumericalError("nonfinite CUDA Fourier result");
    return result;
  }
};
} // namespace
std::unique_ptr<Compute> make_cuda_compute(int ordinal, std::size_t cap) {
  return std::make_unique<CudaCompute>(ordinal, cap);
}
} // namespace fomhd
