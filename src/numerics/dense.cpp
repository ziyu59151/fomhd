#include "fomhd/numerics.hpp"

#include <mkl_lapacke.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

namespace fomhd {
namespace {

constexpr std::size_t kComplexBytes = sizeof(Complex);

void validate_shape(std::size_t rows, std::size_t cols,
                    const std::string& name) {
    static_cast<void>(checked_product(rows, cols, name));
    if (rows > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max()) ||
        cols > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max())) {
        throw NumericalError(name + " exceeds LAPACK LP64 dimensions");
    }
}

std::vector<MKL_Complex16> to_mkl(const Vec& values) {
    std::vector<MKL_Complex16> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i].real = values[i].real();
        result[i].imag = values[i].imag();
    }
    return result;
}

Vec from_mkl(const std::vector<MKL_Complex16>& values) {
    Vec result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        result[i] = Complex{values[i].real, values[i].imag};
    }
    return result;
}

lapack_int checked_lapack_size(std::size_t value, const std::string& name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<lapack_int>::max())) {
        throw NumericalError(name + " exceeds LAPACK LP64 dimensions");
    }
    return static_cast<lapack_int>(value);
}

Matrix transpose_matrix(const Matrix& matrix) {
    Matrix result(matrix.cols, matrix.rows);
    for (std::size_t r = 0; r < matrix.rows; ++r) {
        for (std::size_t c = 0; c < matrix.cols; ++c) {
            result(c, r) = matrix(r, c);
        }
    }
    return result;
}

void require_square_finite(const Matrix& matrix) {
    if (matrix.rows != matrix.cols || matrix.rows == 0) {
        throw NumericalError("LU factorization requires a nonempty square matrix");
    }
    if (!is_finite(matrix)) {
        throw NumericalError("LU factorization input contains non-finite values");
    }
    validate_shape(matrix.rows, matrix.cols, "LU matrix");
}

double row_norm(const Matrix& matrix, std::size_t row) {
    double sum = 0.0;
    for (std::size_t c = 0; c < matrix.cols; ++c) {
        sum = std::hypot(sum, std::abs(matrix(row,c)));
    }
    if(!std::isfinite(sum))throw NumericalError("row norm is not representable");
    return sum;
}

}  // namespace

std::size_t checked_product(std::size_t a, std::size_t b,
                            const std::string& name) {
    if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
        throw NumericalError(name + " size product overflows");
    }
    return a * b;
}

std::size_t checked_sum(std::size_t a, std::size_t b,
                        const std::string& name) {
    if (b > std::numeric_limits<std::size_t>::max() - a) {
        throw NumericalError(name + " size sum overflows");
    }
    return a + b;
}

void require_memory(std::size_t bytes, const MemoryBudget& budget) {
    if (budget.limit_bytes != 0 && bytes > budget.limit_bytes) {
        throw NumericalError(budget.name + " memory budget exceeded: required=" +
                             std::to_string(bytes) + " limit=" +
                             std::to_string(budget.limit_bytes));
    }
}

Matrix::Matrix(std::size_t r, std::size_t c, Complex fill)
    : rows(r), cols(c), data(checked_product(r, c, "Matrix"), fill) {}

Complex& Matrix::operator()(std::size_t r, std::size_t c) {
    if (r >= rows || c >= cols) {
        throw NumericalError("Matrix index out of bounds");
    }
    return data[r + c * rows];
}

const Complex& Matrix::operator()(std::size_t r, std::size_t c) const {
    if (r >= rows || c >= cols) {
        throw NumericalError("Matrix index out of bounds");
    }
    return data[r + c * rows];
}

Vec matvec(const Matrix& matrix, const Vec& vector) {
    if (matrix.cols != vector.size()) {
        throw NumericalError("matrix-vector shape mismatch");
    }
    Vec result(matrix.rows, Complex{});
    for (std::size_t c = 0; c < matrix.cols; ++c) {
        const Complex value = vector[c];
        for (std::size_t r = 0; r < matrix.rows; ++r) {
            result[r] += matrix(r, c) * value;
        }
    }
    return result;
}

Matrix matmul(const Matrix& left, const Matrix& right) {
    if (left.cols != right.rows) {
        throw NumericalError("matrix-matrix shape mismatch");
    }
    Matrix result(left.rows, right.cols);
    for (std::size_t c = 0; c < right.cols; ++c) {
        for (std::size_t k = 0; k < left.cols; ++k) {
            const Complex value = right(k, c);
            for (std::size_t r = 0; r < left.rows; ++r) {
                result(r, c) += left(r, k) * value;
            }
        }
    }
    return result;
}

Matrix adjoint(const Matrix& matrix) {
    Matrix result(matrix.cols, matrix.rows);
    for (std::size_t r = 0; r < matrix.rows; ++r) {
        for (std::size_t c = 0; c < matrix.cols; ++c) {
            result(c, r) = std::conj(matrix(r, c));
        }
    }
    return result;
}

Matrix identity(std::size_t size) {
    Matrix result(size, size);
    for (std::size_t i = 0; i < size; ++i) {
        result(i, i) = Complex{1.0};
    }
    return result;
}

Matrix diagonal(const RealVec& values) {
    Matrix result(values.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            throw NumericalError("diagonal contains a non-finite value");
        }
        result(i, i) = values[i];
    }
    return result;
}

Matrix block_diagonal(const std::vector<Matrix>& blocks) {
    std::size_t rows = 0;
    std::size_t cols = 0;
    for (const Matrix& block : blocks) {
        rows = checked_sum(rows, block.rows, "block diagonal rows");
        cols = checked_sum(cols, block.cols, "block diagonal columns");
    }
    Matrix result(rows, cols);
    std::size_t row_offset = 0;
    std::size_t col_offset = 0;
    for (const Matrix& block : blocks) {
        for (std::size_t c = 0; c < block.cols; ++c) {
            for (std::size_t r = 0; r < block.rows; ++r) {
                result(row_offset + r, col_offset + c) = block(r, c);
            }
        }
        row_offset += block.rows;
        col_offset += block.cols;
    }
    return result;
}

Matrix materialize(std::size_t nout, std::size_t nin,
                   const std::function<Vec(const Vec&)>& action) {
    return materialize(nout, nin, action, MemoryBudget{});
}

Matrix materialize(std::size_t nout, std::size_t nin,
                   const std::function<Vec(const Vec&)>& action,
                   const MemoryBudget& budget) {
    const std::size_t elements = checked_product(nout, nin, "materialize");
    const std::size_t bytes =
        checked_product(elements, kComplexBytes, "materialize bytes");
    require_memory(bytes, budget);
    Matrix result(nout, nin);
    for (std::size_t c = 0; c < nin; ++c) {
        Vec unit(nin);
        unit[c] = Complex{1.0};
        const Vec column = action(unit);
        if (column.size() != nout || !is_finite(column)) {
            throw NumericalError("materialized action returned an invalid column");
        }
        for (std::size_t r = 0; r < nout; ++r) {
            result(r, c) = column[r];
        }
    }
    return result;
}

Vec add(const Vec& left, const Vec& right) {
    if (left.size() != right.size()) {
        throw NumericalError("vector addition shape mismatch");
    }
    Vec result(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        result[i] = left[i] + right[i];
    }
    return result;
}

Vec subtract(const Vec& left, const Vec& right) {
    if (left.size() != right.size()) {
        throw NumericalError("vector subtraction shape mismatch");
    }
    Vec result(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        result[i] = left[i] - right[i];
    }
    return result;
}

Vec scaled(const Vec& vector, Complex factor) {
    Vec result(vector.size());
    for (std::size_t i = 0; i < vector.size(); ++i) {
        result[i] = vector[i] * factor;
    }
    return result;
}

double max_abs(const Vec& vector) {
    double result = 0.0;
    for (const Complex& value : vector) {
        if(!std::isfinite(value.real())||!std::isfinite(value.imag()))
            throw NumericalError("maximum norm requires finite input");
        result = std::max(result, std::abs(value));
    }
    return result;
}

double max_abs(const Matrix& matrix) {
    return max_abs(matrix.data);
}

double norm(const Vec& vector) {
    if(!is_finite(vector))throw NumericalError("norm requires finite input");
    double result=0;
    for(const auto& z:vector)result=std::hypot(result,std::abs(z));
    if(!std::isfinite(result))throw NumericalError("norm is not representable");
    return result;
}

double weighted_norm(const Vec& vector,const RealVec& weights) {
    if(vector.size()!=weights.size()||!is_finite(vector))
        throw NumericalError("weighted norm shape or finite input failure");
    double result=0;
    for(std::size_t i=0;i<vector.size();++i) {
        if(!std::isfinite(weights[i])||weights[i]<0)
            throw NumericalError("weighted norm requires finite nonnegative weights");
        result=std::hypot(result,std::sqrt(weights[i])*std::abs(vector[i]));
    }
    if(!std::isfinite(result))throw NumericalError("weighted norm is not representable");
    return result;
}

Complex weighted_dot(const Vec& left, const Vec& right,
                     const RealVec& weights) {
    if (left.size() != right.size() || left.size() != weights.size()) {
        throw NumericalError("weighted dot shape mismatch");
    }
    Complex result{};
    for (std::size_t i = 0; i < left.size(); ++i) {
        if (!std::isfinite(weights[i]) || weights[i] < 0.0) {
            throw NumericalError("weighted dot requires finite nonnegative weights");
        }
        result += weights[i] * std::conj(left[i]) * right[i];
    }
    return result;
}

LU factor(const Matrix& matrix) {
    return factor(matrix, MemoryBudget{});
}

LU factor(const Matrix& matrix, const MemoryBudget& budget) {
    require_square_finite(matrix);
    const std::size_t matrix_bytes =
        checked_product(matrix.data.size(), kComplexBytes, "LU matrix bytes");
    const std::size_t pivot_bytes =
        checked_product(matrix.rows, sizeof(lapack_int), "LU pivot bytes");
    require_memory(checked_sum(matrix_bytes, pivot_bytes, "LU bytes"), budget);

    LU result;
    result.original = matrix;
    result.factors = matrix;
    result.pivots.resize(matrix.rows);
    auto factors_mkl = to_mkl(result.factors.data);
    const lapack_int dimension = checked_lapack_size(matrix.rows, "LU dimension");
    const lapack_int info =
        LAPACKE_zgetrf(LAPACK_COL_MAJOR, dimension, dimension,
                       factors_mkl.data(), dimension, result.pivots.data());
    if (info < 0) {
        throw NumericalError("LAPACKE_zgetrf rejected an argument: info=" +
                             std::to_string(info));
    }
    if (info > 0) {
        throw NumericalError("singular matrix in LU factorization at pivot " +
                             std::to_string(info));
    }
    result.factors.data = from_mkl(factors_mkl);
    if (!is_finite(result.factors)) {
        throw NumericalError("LU factors contain non-finite values");
    }
    result.valid = true;
    return result;
}

Vec solve(const LU& factorization, const Vec& rhs) {
    if (!factorization.valid) {
        throw NumericalError("invalid LU factorization");
    }
    if (rhs.size() != factorization.factors.rows || !is_finite(rhs)) {
        throw NumericalError("LU right-hand side shape or value is invalid");
    }
    auto rhs_mkl = to_mkl(rhs);
    auto factors_mkl = to_mkl(factorization.factors.data);
    const lapack_int dimension =
        checked_lapack_size(factorization.factors.rows, "LU solve dimension");
    const lapack_int info =
        LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', dimension, 1, factors_mkl.data(),
                       dimension, factorization.pivots.data(), rhs_mkl.data(),
                       dimension);
    if (info != 0) {
        throw NumericalError("LAPACKE_zgetrs failed: info=" +
                             std::to_string(info));
    }
    Vec result = from_mkl(rhs_mkl);
    if (!is_finite(result)) {
        throw NumericalError("LU solution contains non-finite values");
    }
    return result;
}

Matrix solve(const LU& factorization, const Matrix& rhs) {
    if (!factorization.valid || rhs.rows != factorization.factors.rows ||
        !is_finite(rhs)) {
        throw NumericalError("LU multiple right-hand side is invalid");
    }
    auto rhs_mkl = to_mkl(rhs.data);
    auto factors_mkl = to_mkl(factorization.factors.data);
    const lapack_int dimension =
        checked_lapack_size(factorization.factors.rows, "LU solve dimension");
    const lapack_int columns =
        checked_lapack_size(rhs.cols, "LU right-hand side count");
    const lapack_int lda = static_cast<lapack_int>(dimension);
    const lapack_int info =
        LAPACKE_zgetrs(LAPACK_COL_MAJOR, 'N', dimension, columns,
                       factors_mkl.data(), lda, factorization.pivots.data(),
                       rhs_mkl.data(), lda);
    if (info != 0) {
        throw NumericalError("LAPACKE_zgetrs batch failed: info=" +
                             std::to_string(info));
    }
    Matrix result(rhs.rows, rhs.cols);
    result.data = from_mkl(rhs_mkl);
    if (!is_finite(result)) {
        throw NumericalError("LU batch solution contains non-finite values");
    }
    return result;
}

RealVec singular_values(const Matrix& matrix) {
    std::size_t m = matrix.rows;
    std::size_t n = matrix.cols;
    if (std::min(m, n) == 0) {
        return {};
    }
    if (!is_finite(matrix)) {
        throw NumericalError("SVD input contains non-finite values");
    }
    validate_shape(m, n, "SVD matrix");
    Matrix working = m < n ? transpose_matrix(matrix) : matrix;
    if (m < n) {
        std::swap(m, n);
    }
    const lapack_int lm = checked_lapack_size(m, "SVD rows");
    const lapack_int ln = checked_lapack_size(n, "SVD columns");
    std::vector<double> values(n);
    std::vector<MKL_Complex16> dummy_u(1);
    std::vector<MKL_Complex16> dummy_vt(1);
    MKL_Complex16 workspace_query{};
    const lapack_int rwork_size =
        static_cast<lapack_int>(std::max<std::size_t>(1, 5 * n));
    std::vector<double> rwork(static_cast<std::size_t>(rwork_size));

    auto matrix_mkl = to_mkl(working.data);
    lapack_int info = LAPACKE_zgesvd_work(
        LAPACK_COL_MAJOR, 'N', 'N', lm, ln, matrix_mkl.data(), lm, values.data(),
        dummy_u.data(), 1, dummy_vt.data(), 1, &workspace_query, -1,
        rwork.data());
    if (info != 0 || !std::isfinite(workspace_query.real) ||
        workspace_query.real <= 0.0) {
        throw NumericalError("LAPACKE_zgesvd workspace query failed: info=" +
                             std::to_string(info));
    }
    const std::size_t work_entries =
        static_cast<std::size_t>(std::ceil(workspace_query.real));
    const lapack_int work_size = checked_lapack_size(work_entries, "SVD workspace");
    std::vector<MKL_Complex16> workspace(work_entries);
    matrix_mkl = to_mkl(working.data);
    info = LAPACKE_zgesvd_work(
        LAPACK_COL_MAJOR, 'N', 'N', lm, ln, matrix_mkl.data(), lm, values.data(),
        dummy_u.data(), 1, dummy_vt.data(), 1, workspace.data(), work_size,
        rwork.data());
    if (info != 0) {
        throw NumericalError("LAPACKE_zgesvd failed: info=" +
                             std::to_string(info));
    }
    for (double& value : values) {
        if (!std::isfinite(value) || value < -1e-12) {
            throw NumericalError("SVD returned an invalid singular value");
        }
        value = std::max(0.0, value);
    }
    for (std::size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[i - 1] * (1.0 + 1e-12) + 1e-14) {
            throw NumericalError("SVD singular values are not nonincreasing");
        }
    }
    return values;
}

LinearResult checked_solve(const Matrix& matrix, const Vec& rhs,
                           const RealVec& row_scale,
                           const RealVec& col_scale, double rank_rtol,
                           double residual_rtol) {
    require_square_finite(matrix);
    if (rhs.size() != matrix.rows || row_scale.size() != matrix.rows ||
        col_scale.size() != matrix.cols || !(rank_rtol > 0.0 && rank_rtol < 1.0) ||
        !(residual_rtol > 0.0) || !std::isfinite(residual_rtol) ||
        !is_finite(rhs)) {
        throw NumericalError("checked solve shape, threshold, or tolerance is invalid");
    }
    for (std::size_t i = 0; i < row_scale.size(); ++i) {
        if (!std::isfinite(row_scale[i]) || row_scale[i] <= 0.0) {
            throw NumericalError("row scale must be finite and positive");
        }
    }
    for (std::size_t i = 0; i < col_scale.size(); ++i) {
        if (!std::isfinite(col_scale[i]) || col_scale[i] <= 0.0) {
            throw NumericalError("column scale must be finite and positive");
        }
    }
    Matrix scaled_matrix(matrix.rows, matrix.cols);
    Vec scaled_rhs(rhs.size());
    for (std::size_t c = 0; c < matrix.cols; ++c) {
        for (std::size_t r = 0; r < matrix.rows; ++r) {
            scaled_matrix(r, c) =
                matrix(r, c) * col_scale[c] / row_scale[r];
        }
    }
    for (std::size_t r = 0; r < rhs.size(); ++r) {
        scaled_rhs[r] = rhs[r] / row_scale[r];
    }
    if (!is_finite(scaled_matrix) || !is_finite(scaled_rhs)) {
        throw NumericalError("scaled linear system contains non-finite values");
    }
    const RealVec values = singular_values(scaled_matrix);
    if (values.empty() || values.front() == 0.0 ||
        values.back() <= rank_rtol * values.front()) {
        throw NumericalError("scaled linear system is singular or ill-conditioned");
    }
    const LU factorization = factor(scaled_matrix);
    Vec y = solve(factorization, scaled_rhs);
    LinearResult result;
    double previous_error = std::numeric_limits<double>::infinity();
    constexpr std::size_t max_refinement_steps = 5;
    for (;;) {
        const Vec scaled_residual_vector =
            subtract(matvec(scaled_matrix, y), scaled_rhs);
        result.scaled_residual = norm(scaled_residual_vector);
        result.backward_error = 0.0;
        for (std::size_t r = 0; r < matrix.rows; ++r) {
            long double scale = std::abs(scaled_rhs[r]);
            for (std::size_t j = 0; j < matrix.cols; ++j)
                scale += static_cast<long double>(std::abs(scaled_matrix(r, j))) *
                         std::abs(y[j]);
            const double defect = std::abs(scaled_residual_vector[r]);
            result.backward_error = std::max(result.backward_error,
                scale > 0 ? static_cast<double>(defect / scale) : defect);
        }
        if (!std::isfinite(result.backward_error))
            throw NumericalError("linear refinement backward error is not finite");
        if (result.backward_error <= residual_rtol)
            break;
        if (result.refinement_steps == max_refinement_steps ||
            result.backward_error >= previous_error)
            throw NumericalError("scaled componentwise backward error exceeded its tolerance");
        previous_error = result.backward_error;
        const Vec correction = solve(factorization, scaled(scaled_residual_vector, -1.0));
        Vec next = add(y, correction);
        if (!is_finite(next) || next == y)
            throw NumericalError("linear refinement is non-finite or stagnated");
        y = std::move(next);
        ++result.refinement_steps;
    }
    result.x.resize(y.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
        result.x[i] = col_scale[i] * y[i];
    }
    result.residual = subtract(matvec(matrix, result.x), rhs);
    result.singular_values = values;
    result.row_scale=row_scale;result.column_scale=col_scale;
    if(!is_finite(result.x)||!is_finite(result.residual)||
       !std::isfinite(result.scaled_residual)||!std::isfinite(result.backward_error)||
       result.backward_error>residual_rtol)
        throw NumericalError("scaled componentwise backward error exceeded its tolerance");
    return result;
}

std::vector<std::size_t> independent_rows(const Matrix& matrix,
                                          double rank_rtol) {
    if (!(rank_rtol > 0.0 && rank_rtol < 1.0) || !is_finite(matrix)) {
        throw NumericalError("independent row input is invalid");
    }
    std::vector<std::size_t> selected;
    std::size_t current_rank = 0;
    for (std::size_t row = 0; row < matrix.rows; ++row) {
        const double magnitude = row_norm(matrix, row);
        if (magnitude == 0.0) {
            continue;
        }
        Matrix candidate(selected.size() + 1, matrix.cols);
        for (std::size_t r = 0; r < selected.size(); ++r) {
            const std::size_t source = selected[r];
            const double scale = row_norm(matrix, source);
            for (std::size_t c = 0; c < matrix.cols; ++c) {
                candidate(r, c) = matrix(source, c) / scale;
            }
        }
        for (std::size_t c = 0; c < matrix.cols; ++c) {
            candidate(selected.size(), c) = matrix(row, c) / magnitude;
        }
        const RealVec values = singular_values(candidate);
        std::size_t new_rank = 0;
        if (!values.empty() && values.front() > 0.0) {
            for (double value : values) {
                new_rank += value > rank_rtol * values.front() ? 1U : 0U;
            }
        }
        if (new_rank != current_rank && new_rank != current_rank + 1) {
            throw NumericalError("independent-row rank is ambiguous");
        }
        if (new_rank == current_rank + 1) {
            selected.push_back(row);
            current_rank = new_rank;
        }
    }
    return selected;
}

bool is_finite(const Vec& vector) {
    return std::all_of(vector.begin(), vector.end(), [](const Complex& value) {
        return std::isfinite(value.real()) && std::isfinite(value.imag());
    });
}

bool is_finite(const Matrix& matrix) {
    return is_finite(matrix.data);
}

}  // namespace fomhd
