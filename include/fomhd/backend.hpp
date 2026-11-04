#pragma once
#include "fomhd/numerics.hpp"
#include <memory>

namespace fomhd {
// A0 correctness backend: every host/device transfer is explicit and counted.
// Geometry and deterministic diagnostic reductions remain on the host.
class Compute {
public:
  virtual ~Compute() = default;
  virtual std::string name() const { return "cpu"; }
  virtual Vec mv(const Matrix &, const Vec &);
  virtual Matrix mm(const Matrix &, const Matrix &);
  virtual LinearResult linear(const Matrix &, const Vec &, const RealVec &,
                              const RealVec &, double rank_rtol,
                              double residual_rtol);
  virtual Vec inverse_fft(const Vec &, std::size_t entities, int K, int N);
  virtual Vec forward_fft(const Vec &, std::size_t entities, int K, int N);
  virtual Vec derivative(const Vec &, std::size_t entities, int N);
  virtual Vec axpy(const Vec &, const Vec &, double);
  virtual void synchronize() {}
  std::uint64_t h2d_bytes{}, d2h_bytes{}, matrix_calls{}, fft_calls{},
      solve_calls{};
};
std::unique_ptr<Compute> make_cuda_compute(int ordinal,
                                           std::size_t memory_limit);
} // namespace fomhd
