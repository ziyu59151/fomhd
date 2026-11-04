#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fomhd {

using Complex = std::complex<double>;
using Vec = std::vector<Complex>;
using RealVec = std::vector<double>;

class NumericalError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

struct MemoryBudget {
  std::string name{"allocation"};
  std::size_t limit_bytes{};
};

std::size_t checked_product(std::size_t a, std::size_t b,
                            const std::string &name);
std::size_t checked_sum(std::size_t a, std::size_t b, const std::string &name);
void require_memory(std::size_t bytes, const MemoryBudget &budget);

struct Matrix {
  std::size_t rows{};
  std::size_t cols{};
  Vec data;

  Matrix() = default;
  Matrix(std::size_t r, std::size_t c, Complex fill = Complex{});

  Complex &operator()(std::size_t r, std::size_t c);
  const Complex &operator()(std::size_t r, std::size_t c) const;
};

Vec matvec(const Matrix &matrix, const Vec &vector);
Matrix matmul(const Matrix &left, const Matrix &right);
Matrix adjoint(const Matrix &matrix);
Matrix identity(std::size_t size);
Matrix diagonal(const RealVec &values);
Matrix block_diagonal(const std::vector<Matrix> &blocks);
Matrix materialize(std::size_t nout, std::size_t nin,
                   const std::function<Vec(const Vec &)> &action);
Matrix materialize(std::size_t nout, std::size_t nin,
                   const std::function<Vec(const Vec &)> &action,
                   const MemoryBudget &budget);

Vec add(const Vec &left, const Vec &right);
Vec subtract(const Vec &left, const Vec &right);
Vec scaled(const Vec &vector, Complex factor);
double max_abs(const Vec &vector);
double max_abs(const Matrix &matrix);
double norm(const Vec &vector);
double weighted_norm(const Vec &vector, const RealVec &weights);
Complex weighted_dot(const Vec &left, const Vec &right, const RealVec &weights);

struct LU {
  Matrix original;
  Matrix factors;
  std::vector<int> pivots;
  bool valid{};
};

LU factor(const Matrix &matrix);
LU factor(const Matrix &matrix, const MemoryBudget &budget);
Vec solve(const LU &factorization, const Vec &rhs);
Matrix solve(const LU &factorization, const Matrix &rhs);
RealVec singular_values(const Matrix &matrix);

struct LinearResult {
  Vec x;
  Vec residual;
  RealVec singular_values;
  double scaled_residual{};
  double backward_error{};
  RealVec row_scale, column_scale;
  std::size_t refinement_steps{}; // Completed FP64 residual corrections, at most 5.
};

LinearResult checked_solve(const Matrix &matrix, const Vec &rhs,
                           const RealVec &row_scale, const RealVec &col_scale,
                           double rank_rtol, double residual_rtol = 1e-10);
std::vector<std::size_t> independent_rows(const Matrix &matrix,
                                          double rank_rtol);

class Fourier {
public:
  int K{};
  int N{};

  Fourier(int retained_k, int nphi);
  Vec inverse(const Vec &hat, std::size_t entities, bool physical_real = false,
              double tol = 1e-11) const;
  Vec forward(const Vec &samples, std::size_t entities) const;
  Vec derivative(const Vec &samples, std::size_t entities) const;
  double hermitian_defect(const Vec &hat, std::size_t entities) const;
};

bool is_finite(const Vec &vector);
bool is_finite(const Matrix &matrix);

} // namespace fomhd
