#include "fomhd/backend.hpp"

namespace fomhd {
Vec Compute::mv(const Matrix &a, const Vec &x) {
  ++matrix_calls;
  return matvec(a, x);
}
Matrix Compute::mm(const Matrix &a, const Matrix &b) {
  ++matrix_calls;
  return matmul(a, b);
}
LinearResult Compute::linear(const Matrix &a, const Vec &b, const RealVec &rows,
                             const RealVec &cols, double rank,
                             double tolerance) {
  ++solve_calls;
  return checked_solve(a, b, rows, cols, rank, tolerance);
}
Vec Compute::inverse_fft(const Vec &x, std::size_t E, int K, int N) {
  ++fft_calls;
  return Fourier(K, N).inverse(x, E, false);
}
Vec Compute::forward_fft(const Vec &x, std::size_t E, int K, int N) {
  ++fft_calls;
  return Fourier(K, N).forward(x, E);
}
Vec Compute::derivative(const Vec &x, std::size_t E, int N) {
  ++fft_calls;
  return Fourier(0, N).derivative(x, E);
}
Vec Compute::axpy(const Vec &a, const Vec &b, double alpha) {
  return add(a, scaled(b, alpha));
}
} // namespace fomhd
#ifndef FOMHD_HAS_CUDA
namespace fomhd {
std::unique_ptr<Compute> make_cuda_compute(int, std::size_t) {
  throw NumericalError("this executable was built without CUDA support");
}
} // namespace fomhd
#endif
