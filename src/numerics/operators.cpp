#include "fomhd/operators.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fomhd {
namespace {

constexpr Complex kI{0.0, 1.0};

Matrix diagonal_inverse(const RealVec& values) {
    Matrix result(values.size(), values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!(values[i] > 0.0) || !std::isfinite(values[i])) {
            throw NumericalError("diagonal inverse requires positive finite weights");
        }
        result(i, i) = 1.0 / values[i];
    }
    return result;
}

RealVec reciprocal_positive(const RealVec& values, const char* name) {
    RealVec result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!(values[i] > 0.0) || !std::isfinite(values[i])) {
            throw NumericalError(std::string{name} +
                                 " requires positive finite weights");
        }
        result[i] = 1.0 / values[i];
    }
    return result;
}

Matrix right_solve(const Matrix& block, const LU& factor) {
    if (!factor.valid || factor.original.rows != factor.original.cols ||
        factor.original.rows != block.cols) {
        throw NumericalError("right solve factor or block shape mismatch");
    }
    return adjoint(solve(factor, adjoint(block)));
}

Matrix scale_rows(const Matrix& matrix, const RealVec& scale) {
    if (scale.size() != matrix.rows) {
        throw NumericalError("row scale length mismatch");
    }
    Matrix result(matrix.rows, matrix.cols);
    for (std::size_t c = 0; c < matrix.cols; ++c) {
        for (std::size_t r = 0; r < matrix.rows; ++r) {
            result(r, c) = scale[r] * matrix(r, c);
        }
    }
    return result;
}

Matrix scale_columns(const Matrix& matrix, const RealVec& scale) {
    if (scale.size() != matrix.cols) {
        throw NumericalError("column scale length mismatch");
    }
    Matrix result(matrix.rows, matrix.cols);
    for (std::size_t c = 0; c < matrix.cols; ++c) {
        for (std::size_t r = 0; r < matrix.rows; ++r) {
            result(r, c) = matrix(r, c) * scale[c];
        }
    }
    return result;
}

void set_block(Matrix& destination, std::size_t row_offset,
               std::size_t col_offset, const Matrix& block,
               Complex factor = Complex{1.0}) {
    if (row_offset + block.rows > destination.rows ||
        col_offset + block.cols > destination.cols) {
        throw NumericalError("block placement exceeds destination shape");
    }
    for (std::size_t c = 0; c < block.cols; ++c) {
        for (std::size_t r = 0; r < block.rows; ++r) {
            destination(row_offset + r, col_offset + c) +=
                factor * block(r, c);
        }
    }
}

Matrix graph_filter(const EntityTable& entities, bool radial_axis) {
    Matrix result = identity(entities.keys.size());
    for (std::size_t first = 0; first < entities.keys.size(); ++first) {
        if (entities.boundary[first] != 0) {
            continue;
        }
        const EntityKey key = entities.keys[first];
        const EntityKey next_key =
            radial_axis ? EntityKey{key.first + 1, key.second}
                        : EntityKey{key.first, key.second + 1};
        const auto iterator = entities.lookup.find(next_key);
        if (iterator == entities.lookup.end()) {
            continue;
        }
        const std::size_t second = iterator->second;
        if (entities.boundary[second] != 0) {
            continue;
        }
        result(first, first) += 1.0 / 24.0;
        result(second, second) += 1.0 / 24.0;
        result(first, second) -= 1.0 / 24.0;
        result(second, first) -= 1.0 / 24.0;
    }
    return result;
}

Vec apply_graph_filter(const EntityTable& entities, const Vec& values,
                       bool radial_axis) {
    if (values.size() != entities.keys.size()) {
        throw NumericalError("graph-filter input size mismatch");
    }
    Vec result = values;
    for (std::size_t first = 0; first < entities.keys.size(); ++first) {
        if (entities.boundary[first] != 0) {
            continue;
        }
        const EntityKey key = entities.keys[first];
        const auto add_neighbor = [&](bool forward) {
            EntityKey neighbor_key = key;
            if (radial_axis) {
                if (!forward && key.first == 0) {
                    return;
                }
                neighbor_key.first =
                    forward ? key.first + 1 : key.first - 1;
            } else {
                if (!forward && key.second == 0) {
                    return;
                }
                neighbor_key.second =
                    forward ? key.second + 1 : key.second - 1;
            }
            const auto iterator = entities.lookup.find(neighbor_key);
            if (iterator == entities.lookup.end()) {
                return;
            }
            const std::size_t second = iterator->second;
            if (entities.boundary[second] == 0) {
                result[first] += (values[first] - values[second]) / 24.0;
            }
        };
        add_neighbor(false);
        add_neighbor(true);
    }
    return result;
}

Vec matrix_block_column(const Matrix& matrix, std::size_t row_offset,
                        std::size_t count, std::size_t column) {
    if (row_offset + count > matrix.rows || column >= matrix.cols) {
        throw NumericalError("matrix block-column request is out of bounds");
    }
    Vec result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = matrix(row_offset + i, column);
    }
    return result;
}

Matrix make_g2(const Geometry& geometry, int mode, const Matrix& g_r,
               const Matrix& g_z) {
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t ne = nc + nr + nz;
    Matrix result(ne, nc);
    set_block(result, 0, 0, g_r);
    for (std::size_t cell = 0; cell < nc; ++cell) {
        result(nr + cell, cell) =
            kI * static_cast<double>(mode) / geometry.cells.R[cell];
    }
    set_block(result, nr + nc, 0, g_z);
    return result;
}

Matrix make_c2(const Geometry& geometry, int mode, const Matrix& g_r,
               const Matrix& g_z, const Matrix& b1r, const Matrix& b1z) {
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t nv = geometry.vertices.keys.size();
    const std::size_t ne = nc + nr + nz;
    const std::size_t nb = nz + nv + nr;
    Matrix result(nb, ne);

    set_block(result, 0, nr + nc, diagonal_inverse(geometry.z_faces.R),
              kI * static_cast<double>(mode));
    set_block(result, 0, nr, scale_columns(g_z, geometry.cells.R));
    for (std::size_t row = 0; row < nz; ++row) {
        for (std::size_t col = 0; col < nc; ++col) {
            result(row, nr + col) *= -1.0 / geometry.z_faces.R[row];
        }
    }
    set_block(result, nz, 0,
              scale_columns(adjoint(b1r), geometry.lr));
    for (std::size_t row = 0; row < nv; ++row) {
        for (std::size_t col = 0; col < nr; ++col) {
            result(nz + row, col) *= -1.0 / geometry.av[row];
        }
    }
    set_block(result, nz, nr + nc,
              scale_columns(adjoint(b1z), geometry.lz));
    for (std::size_t row = 0; row < nv; ++row) {
        for (std::size_t col = 0; col < nz; ++col) {
            result(nz + row, nr + nc + col) *= -1.0 / geometry.av[row];
        }
    }
    set_block(result, nz + nv, nr,
              scale_columns(g_r, geometry.cells.R));
    for (std::size_t row = 0; row < nr; ++row) {
        for (std::size_t col = 0; col < nc; ++col) {
            result(nz + nv + row, nr + col) /=
                geometry.Rbar_r[row];
        }
    }
    set_block(result, nz + nv, 0, diagonal_inverse(geometry.Rbar_r),
              -kI * static_cast<double>(mode));
    return result;
}

Matrix make_db2(const Geometry& geometry, int mode, const Matrix& b1r,
                const Matrix& b1z) {
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t nv = geometry.vertices.keys.size();
    Matrix result(nv, nz + nv + nr);
    RealVec lz_rz = geometry.lz;
    for (std::size_t i = 0; i < nz; ++i) {
        lz_rz[i] *= geometry.z_faces.R[i];
    }
    set_block(result, 0, 0,
              scale_columns(adjoint(b1z), lz_rz));
    for (std::size_t row = 0; row < nv; ++row) {
        for (std::size_t col = 0; col < nz; ++col) {
            result(row, col) /= geometry.wv[row];
        }
    }
    RealVec lr_rbar = geometry.lr;
    for (std::size_t i = 0; i < nr; ++i) {
        lr_rbar[i] *= geometry.Rbar_r[i];
    }
    set_block(result, 0, nz + nv,
              scale_columns(adjoint(b1r), lr_rbar),
              -1.0);
    for (std::size_t row = 0; row < nv; ++row) {
        for (std::size_t col = 0; col < nr; ++col) {
            result(row, nz + nv + col) /= geometry.wv[row];
        }
    }
    for (std::size_t vertex = 0; vertex < nv; ++vertex) {
        result(vertex, nz + vertex) +=
            kI * static_cast<double>(mode) * geometry.av[vertex] /
            geometry.wv[vertex];
    }
    return result;
}

Matrix make_t1(const Operators& operators) {
    const std::size_t nc = operators.Nc;
    const std::size_t nr = operators.PR_r.rows;
    const std::size_t nz = operators.PZ_z.rows;
    const std::size_t ne = nc + nr + nz;
    Matrix result(ne, ne);
    set_block(result, 0, 0, operators.PR_r);
    set_block(result, nr, nr, identity(nc));
    set_block(result, nr + nc, nr + nc, operators.PZ_z);
    (void)nz;
    return result;
}

Matrix make_t2(const Operators& operators, const RealVec& Rbar_r) {
    const std::size_t nc = operators.Nc;
    const std::size_t nr = operators.PR_r.rows;
    const std::size_t nz = operators.PZ_z.rows;
    const std::size_t nv = operators.PR_v.rows;
    const std::size_t nb = nz + nv + nr;
    Matrix result(nb, nb);
    set_block(result, 0, 0, operators.PZ_z);
    set_block(result, nz, nz, matmul(operators.PR_v, operators.PZ_v));
    set_block(result, nz + nv, nz + nv,
              matmul(diagonal_inverse(Rbar_r),
                     matmul(operators.PR_r, diagonal(Rbar_r))));
    (void)nz;
    (void)nc;
    return result;
}

Matrix make_t3(const Geometry& geometry, const Operators& operators) {
    return matmul(
        diagonal_inverse(geometry.Rbar_v),
        matmul(operators.PR_v,
               matmul(operators.PZ_v, diagonal(geometry.Rbar_v))));
}

struct FixedSpatialState {
    Operators operators;
    Matrix t1;
    Matrix t2;
    Matrix t3;
};

FixedSpatialState make_fixed_spatial_state(const Geometry& geometry) {
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t nv = geometry.vertices.keys.size();
    FixedSpatialState state;
    Operators& operators = state.operators;
    operators.Nc = nc;
    operators.Ne = nc + nr + nz;
    operators.Nb = nz + nv + nr;
    operators.M0 = geometry.M0;
    operators.M1 = geometry.M1;
    operators.MB = geometry.MB;
    operators.Rbar_r = geometry.Rbar_r;
    operators.Rbar_v = geometry.Rbar_v;

    operators.b2r = Matrix(nc, nr);
    operators.b2z = Matrix(nc, nz);
    operators.b1r = Matrix(nr, nv);
    operators.b1z = Matrix(nz, nv);
    for (std::size_t cell = 0; cell < nc; ++cell) {
        const EntityKey key = geometry.cells.keys[cell];
        const std::size_t r_low = geometry.r_faces.find(key.first, key.second);
        const std::size_t r_high =
            geometry.r_faces.find(key.first + 1, key.second);
        operators.b2r(cell, r_high) += 1.0;
        operators.b2r(cell, r_low) -= 1.0;
        const std::size_t z_low = geometry.z_faces.find(key.first, key.second);
        const std::size_t z_high =
            geometry.z_faces.find(key.first, key.second + 1);
        operators.b2z(cell, z_high) += 1.0;
        operators.b2z(cell, z_low) -= 1.0;
    }
    for (std::size_t face = 0; face < nr; ++face) {
        const EntityKey key = geometry.r_faces.keys[face];
        operators.b1r(face, geometry.vertices.find(key.first, key.second + 1)) +=
            1.0;
        operators.b1r(face, geometry.vertices.find(key.first, key.second)) -=
            1.0;
    }
    for (std::size_t face = 0; face < nz; ++face) {
        const EntityKey key = geometry.z_faces.keys[face];
        operators.b1z(face, geometry.vertices.find(key.first, key.second)) +=
            1.0;
        operators.b1z(face, geometry.vertices.find(key.first + 1, key.second)) -=
            1.0;
    }

    operators.PR_r = graph_filter(geometry.r_faces, true);
    operators.PZ_r = graph_filter(geometry.r_faces, false);
    operators.PR_z = graph_filter(geometry.z_faces, true);
    operators.PZ_z = graph_filter(geometry.z_faces, false);
    operators.PR_v = graph_filter(geometry.vertices, true);
    operators.PZ_v = graph_filter(geometry.vertices, false);
    operators.PR_r_factor = factor(operators.PR_r);
    operators.PZ_r_factor = factor(operators.PZ_r);
    operators.PR_z_factor = factor(operators.PR_z);
    operators.PZ_z_factor = factor(operators.PZ_z);
    operators.PR_v_factor = factor(operators.PR_v);
    operators.PZ_v_factor = factor(operators.PZ_v);

    state.t1 = make_t1(operators);
    state.t2 = make_t2(operators, geometry.Rbar_r);
    state.t3 = make_t3(geometry, operators);
    return state;
}

std::size_t estimate_spatial_bytes(const Geometry& geometry) {
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t nv = geometry.vertices.keys.size();
    const std::size_t ne = checked_sum(
        checked_sum(nc, nr, "spatial estimate"),
        nz, "spatial estimate");
    const std::size_t nb = checked_sum(
        checked_sum(nz, nv, "spatial estimate"),
        nr, "spatial estimate");
    const auto product = [](std::size_t left, std::size_t right) {
        return checked_product(left, right, "spatial estimate");
    };
    std::size_t elements = 0;
    const auto add = [&](std::size_t value) {
        elements = checked_sum(elements, value, "spatial estimate");
    };
    const auto add_product = [&](std::size_t left, std::size_t right,
                                 std::size_t copies) {
        add(checked_product(product(left, right), copies,
                            "spatial estimate copies"));
    };

    // Outputs and their temporary products: G, C, DB, and DJ.
    add_product(ne, nc, 4);
    add_product(nb, ne, 6);
    add_product(nv, nb, 6);
    add_product(nc, ne, 5);

    // Low-order blocks and five point-to-entity derivative blocks.
    add_product(nb, ne, 1);
    add_product(nv, nb, 1);
    add_product(ne, nc, 1);
    add_product(nc, nr, 2);
    add_product(nc, nz, 2);

    // Incidence matrices retained by the fixed state.
    add_product(nc, nr, 1);
    add_product(nc, nz, 1);
    add_product(nr, nv, 1);
    add_product(nz, nv, 1);

    // Each graph filter retains the matrix, LU original, and LU factors.
    add_product(nr, nr, 6);
    add_product(nz, nz, 6);
    add_product(nv, nv, 6);
    add(product(ne, ne));

    // T2 and T3 plus the batched right-solve workspaces.
    add_product(nb, nb, 2);

    const std::size_t dense_bytes =
        checked_product(elements, sizeof(Complex),
                        "spatial estimate bytes");
    return dense_bytes;
}

Matrix select_columns(const Matrix& source, std::size_t first,
                      std::size_t count);

SpatialOperators assemble_spatial_operators(
    const Geometry& geometry, int mode,
    const FixedSpatialState& fixed, bool verify_low_algebra,
    const OperatorOptions& options) {
    const Operators& operators = fixed.operators;
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t nv = geometry.vertices.keys.size();
    const std::size_t ne = nc + nr + nz;
    const std::size_t nb = nz + nv + nr;

    const Matrix g_r = matmul(diagonal_inverse(geometry.lr),
                              adjoint(operators.b2r));
    const Matrix negative_g_r = scale_rows(g_r, RealVec(nr, -1.0));
    const Matrix g_z = matmul(diagonal_inverse(geometry.lz),
                              adjoint(operators.b2z));
    const Matrix negative_g_z = scale_rows(g_z, RealVec(nz, -1.0));
    const Matrix g2 =
        make_g2(geometry, mode, negative_g_r, negative_g_z);
    const Matrix c2 =
        make_c2(geometry, mode, negative_g_r, negative_g_z,
                operators.b1r, operators.b1z);
    const Matrix db2 =
        make_db2(geometry, mode, operators.b1r, operators.b1z);
    if (verify_low_algebra) {
        const Matrix low_cg_matrix = matmul(c2, g2);
        const Matrix low_dbc_matrix = matmul(db2, c2);
        const double low_cg = max_abs(low_cg_matrix);
        const double low_dbc = max_abs(low_dbc_matrix);
        const double cg_bound =
            128 * std::numeric_limits<double>::epsilon() * ne *
            max_abs(c2) * max_abs(g2);
        const double dbc_bound =
            128 * std::numeric_limits<double>::epsilon() * nb *
            max_abs(db2) * max_abs(c2);
        if (low_cg > cg_bound || low_dbc > dbc_bound) {
            throw NumericalError(
                "low-order complex algebra failed: mode=" +
                std::to_string(mode) + " CG=" + std::to_string(low_cg) +
                " DBC=" + std::to_string(low_dbc));
        }
    }

    SpatialOperators result;
    result.mode = mode;
    result.Nc = nc;
    result.Ne = ne;
    result.Nb = nb;
    result.G = matmul(fixed.t1, g2);
    Matrix c2_with_inverse_transform(nb, ne);
    set_block(
        c2_with_inverse_transform, 0, 0,
        right_solve(select_columns(c2, 0, nr), operators.PR_r_factor));
    set_block(
        c2_with_inverse_transform, 0, nr,
        select_columns(c2, nr, nc));
    set_block(
        c2_with_inverse_transform, 0, nr + nc,
        right_solve(select_columns(c2, nr + nc, nz),
                    operators.PZ_z_factor));
    result.C =
        matmul(fixed.t2, c2_with_inverse_transform);

    const Matrix db2_with_inverse_transform =
        apply_t2_inverse_block(operators, db2, options);
    result.DB =
        matmul(fixed.t3, db2_with_inverse_transform);

    result.DJ = matmul(
        diagonal_inverse(geometry.M0),
        matmul(adjoint(result.G), diagonal(geometry.M1)));
    for (Complex& value : result.DJ.data) {
        value = -value;
    }
    return result;
}

Matrix select_rows(const Matrix& source, const std::vector<std::size_t>& rows) {
    Matrix result(rows.size(), source.cols);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (rows[r] >= source.rows) {
            throw NumericalError("selected matrix row is out of bounds");
        }
        for (std::size_t c = 0; c < source.cols; ++c) {
            result(r, c) = source(rows[r], c);
        }
    }
    return result;
}

Matrix select_columns(const Matrix& source, std::size_t first,
                      std::size_t count) {
    if (first + count > source.cols) {
        throw NumericalError("selected matrix columns are out of bounds");
    }
    Matrix result(source.rows, count);
    for (std::size_t c = 0; c < count; ++c) {
        for (std::size_t r = 0; r < source.rows; ++r) {
            result(r, c) = source(r, first + c);
        }
    }
    return result;
}

std::size_t estimate_operator_bytes(const Geometry& geometry, int K,
                                    bool estimate_only) {
    static_cast<void>(estimate_only);
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t nv = geometry.vertices.keys.size();
    const std::size_t ne = nc + nr + nz;
    const std::size_t nb = nz + nv + nr;
    const std::size_t modes = static_cast<std::size_t>(2 * K + 1);
    const auto square = [](std::size_t value) {
        return checked_product(value, value, "operator estimate");
    };
    std::size_t elements = 0;
    const auto add_elements = [&](std::size_t value) {
        elements = checked_sum(elements, value, "operator estimate");
    };
    add_elements(square(nr));
    add_elements(square(nz));
    add_elements(square(nv));
    add_elements(square(nv));
    add_elements(square(nc));
    add_elements(square(nc));
    add_elements(square(nr));
    add_elements(square(nz));
    add_elements(square(nc));
    add_elements(square(nz));
    add_elements(square(nc));
    add_elements(square(nv));
    add_elements(square(nv));
    add_elements(checked_product(ne, nc, "operator estimate"));
    add_elements(checked_product(nv, nb, "operator estimate"));
    add_elements(checked_product(ne, ne, "operator estimate"));
    add_elements(checked_product(ne, ne, "operator estimate"));
    add_elements(checked_product(ne, nb, "operator estimate"));
    add_elements(checked_product(nc, nr, "operator estimate"));
    add_elements(checked_product(nc, nz, "operator estimate"));
    add_elements(checked_product(geometry.walls.size(), ne, "operator estimate"));
    add_elements(checked_product(modes, checked_product(nc, ne, "operator estimate"),
                                 "operator estimate"));
    const std::size_t per_mode = checked_product(
        checked_product(modes,
                        checked_sum(
                            checked_sum(checked_product(ne, nc, "operator estimate"),
                                        checked_product(nb, ne, "operator estimate"),
                                        "operator estimate"),
                            checked_sum(checked_product(nv, nb, "operator estimate"),
                                        square(nc), "operator estimate"),
                            "operator estimate"),
                        "operator estimate"),
        2, "operator estimate");
    add_elements(per_mode);
    const std::size_t Bc_elements =
        checked_product(modes,
                        checked_product(nc,
                                        checked_product(modes,
                                                        checked_product(3, nc,
                                                                        "operator estimate"),
                                                        "operator estimate"),
                                        "operator estimate"),
                        "operator estimate");
    add_elements(Bc_elements);
    const std::size_t bytes =
        checked_product(elements, sizeof(Complex), "operator estimate bytes");
    return checked_sum(bytes, bytes / 2, "operator estimate with workspaces");
}

}  // namespace

Matrix build_gradient_operator(const Geometry& geometry, int mode,
                               OperatorOptions options) {
    if (!(options.rank_rtol > 0.0 && options.rank_rtol < 1.0)) {
        throw NumericalError("gradient operator rank tolerance is invalid");
    }
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t ne = nc + nr + nz;
    const std::size_t bytes = checked_product(
        checked_product(ne, nc, "gradient operator"),
        sizeof(Complex), "gradient operator bytes");
    require_memory(bytes,
                   MemoryBudget{"gradient_operator", options.memory_cap_bytes});

    Matrix b2r(nc, nr);
    Matrix b2z(nc, nz);
    for (std::size_t cell = 0; cell < nc; ++cell) {
        const EntityKey key = geometry.cells.keys[cell];
        const std::size_t r_low = geometry.r_faces.find(key.first, key.second);
        const std::size_t r_high =
            geometry.r_faces.find(key.first + 1, key.second);
        b2r(cell, r_high) += 1.0;
        b2r(cell, r_low) -= 1.0;
        const std::size_t z_low = geometry.z_faces.find(key.first, key.second);
        const std::size_t z_high =
            geometry.z_faces.find(key.first, key.second + 1);
        b2z(cell, z_high) += 1.0;
        b2z(cell, z_low) -= 1.0;
    }
    const Matrix g_r = scale_rows(
        matmul(diagonal_inverse(geometry.lr), adjoint(b2r)),
        RealVec(nr, -1.0));
    const Matrix g_z = scale_rows(
        matmul(diagonal_inverse(geometry.lz), adjoint(b2z)),
        RealVec(nz, -1.0));
    Matrix gradient = make_g2(geometry, mode, g_r, g_z);
    for (std::size_t column = 0; column < nc; ++column) {
        const Vec r = matrix_block_column(gradient, 0, nr, column);
        const Vec z = matrix_block_column(gradient, nr + nc, nz, column);
        const Vec filtered_r = apply_graph_filter(geometry.r_faces, r, true);
        const Vec filtered_z = apply_graph_filter(geometry.z_faces, z, false);
        for (std::size_t i = 0; i < nr; ++i) {
            gradient(i, column) = filtered_r[i];
        }
        for (std::size_t i = 0; i < nz; ++i) {
            gradient(nr + nc + i, column) = filtered_z[i];
        }
    }
    return gradient;
}

SpatialOperators build_spatial_operators(const Geometry& geometry, int mode,
                                         OperatorOptions options) {
    if (!(options.rank_rtol > 0.0 && options.rank_rtol < 1.0)) {
        throw NumericalError("spatial operator rank tolerance is invalid");
    }
    require_memory(estimate_spatial_bytes(geometry),
                   MemoryBudget{"spatial_operator_matrices",
                                options.memory_cap_bytes});
    const FixedSpatialState fixed = make_fixed_spatial_state(geometry);
    return assemble_spatial_operators(geometry, mode, fixed, false, options);
}

Matrix apply_t2_inverse_block(const Operators& operators,
                              const Matrix& block,
                              OperatorOptions options) {
    if (!(options.rank_rtol > 0.0 && options.rank_rtol < 1.0)) {
        throw NumericalError("T2 inverse rank tolerance is invalid");
    }
    const std::size_t nz = operators.PZ_z.rows;
    const std::size_t nv = operators.PR_v.rows;
    const std::size_t nr = operators.PR_r.rows;
    const std::size_t nb = nz + nv + nr;
    if (operators.Nb != nb || operators.MB.size() != nb ||
        operators.Rbar_r.size() != nr) {
        throw NumericalError("T2 inverse operator shape mismatch");
    }
    if (block.cols != nb) {
        throw NumericalError("T2 inverse block column count mismatch");
    }
    const std::size_t largest = std::max({nz, nv, nr});
    const std::size_t largest_block =
        checked_product(largest, block.rows,
                        "T2 inverse block workspace");
    // Retained block, adjoint/RHS/result copies, and LAPACK factor conversion.
    const std::size_t elements = checked_sum(
        checked_sum(block.data.size(), checked_product(largest_block, 6,
                                           "T2 inverse workspaces"),
                    "T2 inverse blocks"),
        checked_product(largest, largest, "T2 inverse factor conversion"),
        "T2 inverse block elements");
    const std::size_t bytes = checked_product(
        elements, sizeof(Complex), "T2 inverse bytes");
    require_memory(bytes,
                   MemoryBudget{"T2_inverse", options.memory_cap_bytes});

    Matrix transformed(block.rows, block.cols);
    set_block(
        transformed, 0, 0,
        right_solve(select_columns(block, 0, nz),
                    operators.PZ_z_factor));
    Matrix vertex =
        right_solve(select_columns(block, nz, nv),
                    operators.PZ_v_factor);
    vertex = right_solve(vertex, operators.PR_v_factor);
    set_block(transformed, 0, nz, vertex);
    Matrix radial =
        select_columns(block, nz + nv, nr);
    radial = scale_columns(
        radial,
        reciprocal_positive(operators.Rbar_r, "R-face mean radius"));
    radial = right_solve(radial, operators.PR_r_factor);
    radial = scale_columns(radial, operators.Rbar_r);
    set_block(transformed, 0, nz + nv, radial);
    return transformed;
}

Operators build_operators(const Geometry& geometry, int retained_k,
                          OperatorOptions options) {
    if (retained_k < 0 || !(options.rank_rtol > 0.0 && options.rank_rtol < 1.0)) {
        throw NumericalError("operator mode or rank tolerance is invalid");
    }
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t nv = geometry.vertices.keys.size();
    const std::size_t nb = nz + nv + nr;
    const std::size_t modes = static_cast<std::size_t>(2 * retained_k + 1);
    const std::size_t estimate =
        estimate_operator_bytes(geometry, retained_k, true);
    require_memory(estimate,
                   MemoryBudget{"operator_matrices", options.memory_cap_bytes});

    FixedSpatialState fixed = make_fixed_spatial_state(geometry);
    Operators& operators = fixed.operators;
    operators.K = retained_k;

    operators.G.reserve(modes);
    operators.C.reserve(modes);
    operators.DB.reserve(modes);
    operators.DJ.reserve(modes);
    operators.Cdag.reserve(modes);
    operators.Lphi.reserve(modes);
    operators.phi_factor.reserve(modes);
    for (int mode = -retained_k; mode <= retained_k; ++mode) {
        SpatialOperators spatial =
            assemble_spatial_operators(geometry, mode, fixed, true, options);
        const Matrix cdag =
            matmul(diagonal_inverse(geometry.M1),
                   matmul(adjoint(spatial.C), diagonal(geometry.MB)));
        Matrix lphi = matmul(spatial.DJ, spatial.G);
        for (Complex& value : lphi.data) {
            value = -value;
        }
        operators.G.push_back(std::move(spatial.G));
        operators.C.push_back(std::move(spatial.C));
        operators.DB.push_back(std::move(spatial.DB));
        operators.DJ.push_back(std::move(spatial.DJ));
        operators.Cdag.push_back(std::move(cdag));
        operators.Lphi.push_back(std::move(lphi));
        auto spectrum=singular_values(operators.Lphi.back());
        if(spectrum.empty()||spectrum.back()<=options.rank_rtol*spectrum.front())
            throw NumericalError("Poisson matrix is numerically rank deficient");
        operators.poisson_spectra.push_back(std::move(spectrum));
        operators.phi_factor.push_back(factor(operators.Lphi.back()));
    }

    Matrix i_r(nr, nc);
    for (std::size_t face = 0; face < nr; ++face) {
        const EntityKey key = geometry.r_faces.keys[face];
        if (geometry.r_faces.boundary[face] != 0) {
            const auto wall = std::find_if(
                geometry.walls.begin(), geometry.walls.end(),
                [&](const Wall& candidate) {
                    return candidate.component == WallComponent::r_face &&
                           candidate.face == face;
                });
            if (wall == geometry.walls.end()) {
                throw NumericalError("radial wall lookup failed");
            }
            i_r(face, wall->near_cell) = 1.5;
            i_r(face, wall->next_cell) = -0.5;
            continue;
        }
        const std::array<std::pair<long, long>, 4> cells{{
            {static_cast<long>(key.first) - 2, static_cast<long>(key.second)},
            {static_cast<long>(key.first) - 1, static_cast<long>(key.second)},
            {static_cast<long>(key.first), static_cast<long>(key.second)},
            {static_cast<long>(key.first) + 1, static_cast<long>(key.second)},
        }};
        bool high_order = true;
        std::array<std::size_t, 4> ids{};
        for (std::size_t i = 0; i < 4; ++i) {
            if (cells[i].first < 0 ||
                static_cast<std::size_t>(cells[i].first) >= geometry.R.size() ||
                !geometry.cells.contains(
                    static_cast<std::size_t>(cells[i].first), key.second)) {
                high_order = false;
                break;
            }
            ids[i] = geometry.cells.find(
                static_cast<std::size_t>(cells[i].first), key.second);
        }
        if (high_order) {
            constexpr std::array<double, 4> weights{-1.0 / 16.0, 9.0 / 16.0,
                                                    9.0 / 16.0, -1.0 / 16.0};
            for (std::size_t i = 0; i < 4; ++i) {
                i_r(face, ids[i]) = weights[i];
            }
        } else {
            i_r(face, geometry.cells.find(key.first, key.second)) = 0.5;
            i_r(face, geometry.cells.find(key.first - 1, key.second)) = 0.5;
        }
    }
    Matrix i_z(nz, nc);
    for (std::size_t face = 0; face < nz; ++face) {
        const EntityKey key = geometry.z_faces.keys[face];
        if (geometry.z_faces.boundary[face] != 0) {
            const auto wall = std::find_if(
                geometry.walls.begin(), geometry.walls.end(),
                [&](const Wall& candidate) {
                    return candidate.component == WallComponent::z_face &&
                           candidate.face == face;
                });
            if (wall == geometry.walls.end()) {
                throw NumericalError("axial wall lookup failed");
            }
            i_z(face, wall->near_cell) = 1.5;
            i_z(face, wall->next_cell) = -0.5;
            continue;
        }
        const std::array<std::pair<long, long>, 4> cells{{
            {static_cast<long>(key.first), static_cast<long>(key.second) - 2},
            {static_cast<long>(key.first), static_cast<long>(key.second) - 1},
            {static_cast<long>(key.first), static_cast<long>(key.second)},
            {static_cast<long>(key.first), static_cast<long>(key.second) + 1},
        }};
        bool high_order = true;
        std::array<std::size_t, 4> ids{};
        for (std::size_t i = 0; i < 4; ++i) {
            if (cells[i].second < 0 ||
                static_cast<std::size_t>(cells[i].second) >= geometry.Z.size() ||
                !geometry.cells.contains(
                    key.first, static_cast<std::size_t>(cells[i].second))) {
                high_order = false;
                break;
            }
            ids[i] = geometry.cells.find(
                key.first, static_cast<std::size_t>(cells[i].second));
        }
        if (high_order) {
            constexpr std::array<double, 4> weights{-1.0 / 16.0, 9.0 / 16.0,
                                                    9.0 / 16.0, -1.0 / 16.0};
            for (std::size_t i = 0; i < 4; ++i) {
                i_z(face, ids[i]) = weights[i];
            }
        } else {
            i_z(face, geometry.cells.find(key.first, key.second)) = 0.5;
            i_z(face, geometry.cells.find(key.first, key.second - 1)) = 0.5;
        }
    }
    operators.I = block_diagonal({i_r, identity(nc), i_z});
    RealVec node_weights(3 * nc);
    for (std::size_t component = 0; component < 3; ++component) {
        std::copy(geometry.M0.begin(), geometry.M0.end(),
                  node_weights.begin() +
                      static_cast<std::ptrdiff_t>(component * nc));
    }
    operators.H = matmul(diagonal_inverse(node_weights),
                         matmul(adjoint(operators.I), diagonal(geometry.M1)));
    operators.DR = select_columns(operators.DJ.front(), 0, nr);
    operators.DZ = select_columns(operators.DJ.front(), nr + nc, nz);
    for (Complex& value : operators.DR.data) {
        value = Complex{value.real(), 0.0};
    }
    for (Complex& value : operators.DZ.data) {
        value = Complex{value.real(), 0.0};
    }

    operators.T_full = Matrix(geometry.walls.size(), 3 * nc);
    for (std::size_t wall_id = 0; wall_id < geometry.walls.size(); ++wall_id) {
        const Wall& wall = geometry.walls[wall_id];
        if (wall.component == WallComponent::r_face) {
            for (std::size_t cell = 0; cell < nc; ++cell) {
                operators.T_full(wall_id, cell) =
                    static_cast<double>(wall.sign) * i_r(wall.face, cell);
            }
        } else {
            for (std::size_t cell = 0; cell < nc; ++cell) {
                operators.T_full(wall_id, 2 * nc + cell) =
                    static_cast<double>(wall.sign) * i_z(wall.face, cell);
            }
        }
    }
    operators.selected_wall_rows =
        independent_rows(operators.T_full, options.rank_rtol);
    operators.T = select_rows(operators.T_full, operators.selected_wall_rows);

    operators.QB = Matrix(3 * nc, nb);
    constexpr std::array<double,4> half_weights{-1./16,9./16,9./16,-1./16};
    for (std::size_t cell=0;cell<nc;++cell) {
        const auto key=geometry.cells.keys[cell];
        const auto active=[&](int di,int dj) {
            const auto i=static_cast<long>(key.first)+di;
            const auto j=static_cast<long>(key.second)+dj;
            return i>=0 && j>=0 && geometry.is_active(i,j);
        };
        // Validate the complete connected active support before writing a row.
        bool radial=true,axial=true,patch=true;
        for(int d=-1;d<=1;++d) {
            radial=radial&&active(0,d);
            axial=axial&&active(d,0);
            for(int e=-1;e<=1;++e)patch=patch&&active(d,e);
        }
        if(radial) {
            for(int d=0;d<4;++d)
                operators.QB(cell,geometry.z_faces.find(key.first,key.second+d-1))=half_weights[d];
        } else {
            operators.QB(cell,geometry.z_faces.find(key.first,key.second))=.5;
            operators.QB(cell,geometry.z_faces.find(key.first,key.second+1))=.5;
        }
        if(axial) {
            for(int d=0;d<4;++d)
                operators.QB(2*nc+cell,nz+nv+geometry.r_faces.find(key.first+d-1,key.second))=half_weights[d];
        } else {
            operators.QB(2*nc+cell,nz+nv+geometry.r_faces.find(key.first,key.second))=.5;
            operators.QB(2*nc+cell,nz+nv+geometry.r_faces.find(key.first+1,key.second))=.5;
        }
        if(patch) {
            for(int i=0;i<4;++i)for(int j=0;j<4;++j)
                operators.QB(nc+cell,nz+geometry.vertices.find(key.first+i-1,key.second+j-1))=half_weights[i]*half_weights[j];
        } else {
            for(int i=0;i<2;++i)for(int j=0;j<2;++j)
                operators.QB(nc+cell,nz+geometry.vertices.find(key.first+i,key.second+j))=.25;
        }
    }
    std::vector<Matrix> dj_blocks = operators.DJ;
    operators.Bc_full = block_diagonal(dj_blocks);
    operators.selected_constraint_rows =
        independent_rows(operators.Bc_full, options.rank_rtol);
    operators.Bc = select_rows(operators.Bc_full,
                               operators.selected_constraint_rows);

    Vec ell2(nb);
    for (std::size_t vertex = 0; vertex < nv; ++vertex) {
        ell2[nz + vertex] = geometry.av[vertex];
    }
    const LU adjoint_t2 = factor(adjoint(fixed.t2));
    const Vec ell_column = solve(adjoint_t2, [&] {
        Vec rhs(ell2.size());
        for (std::size_t i = 0; i < ell2.size(); ++i) {
            rhs[i] = std::conj(ell2[i]);
        }
        return rhs;
    }());
    operators.topology_flux_row = Matrix(1, nb);
    for (std::size_t i = 0; i < ell_column.size(); ++i) {
        operators.topology_flux_row(0, i) = std::conj(ell_column[i]);
    }
    if (operators.G.size() != modes || operators.C.size() != modes ||
        operators.DB.size() != modes || operators.Lphi.size() != modes) {
        throw NumericalError("operator mode count is inconsistent");
    }
    operators.setup_checks=validate_operators(geometry,operators);
    return std::move(fixed.operators);
}

Vec apply_t1(const Operators& operators, const Vec& x) {
    const std::size_t nc = operators.Nc;
    const std::size_t nr = operators.PR_r.rows;
    const std::size_t nz = operators.PZ_z.rows;
    if (x.size() != nc + nr + nz) {
        throw NumericalError("T1 input size mismatch");
    }
    Vec result(x.size());
    const Vec r = matvec(
        operators.PR_r,
        Vec(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(nr)));
    const Vec z = matvec(
        operators.PZ_z,
        Vec(x.begin() + static_cast<std::ptrdiff_t>(nr + nc), x.end()));
    std::copy(r.begin(), r.end(), result.begin());
    std::copy(x.begin() + static_cast<std::ptrdiff_t>(nr),
              x.begin() + static_cast<std::ptrdiff_t>(nr + nc),
              result.begin() + static_cast<std::ptrdiff_t>(nr));
    std::copy(z.begin(), z.end(),
              result.begin() + static_cast<std::ptrdiff_t>(nr + nc));
    (void)nz;
    return result;
}

Vec apply_t1_inverse(const Operators& operators, const Vec& x) {
    const std::size_t nc = operators.Nc;
    const std::size_t nr = operators.PR_r.rows;
    const std::size_t nz = operators.PZ_z.rows;
    if (x.size() != nc + nr + nz) {
        throw NumericalError("T1 inverse input size mismatch");
    }
    Vec result(x.size());
    const Vec solved_r =
        solve(operators.PR_r_factor,
              Vec(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(nr)));
    const Vec solved_z = solve(
        operators.PZ_z_factor,
        Vec(x.begin() + static_cast<std::ptrdiff_t>(nr + nc), x.end()));
    std::copy(solved_r.begin(), solved_r.end(), result.begin());
    std::copy(x.begin() + static_cast<std::ptrdiff_t>(nr),
              x.begin() + static_cast<std::ptrdiff_t>(nr + nc),
              result.begin() + static_cast<std::ptrdiff_t>(nr));
    std::copy(solved_z.begin(), solved_z.end(),
              result.begin() + static_cast<std::ptrdiff_t>(nr + nc));
    (void)nz;
    return result;
}

Vec apply_t2(const Operators& operators, const Vec& x) {
    const std::size_t nc = operators.Nc;
    const std::size_t nr = operators.PR_r.rows;
    const std::size_t nz = operators.PZ_z.rows;
    const std::size_t nv = operators.PR_v.rows;
    if (x.size() != nz + nv + nr) {
        throw NumericalError("T2 input size mismatch");
    }
    Vec result(x.size());
    const Vec r(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(nz));
    const Vec phi(x.begin() + static_cast<std::ptrdiff_t>(nz),
                  x.begin() + static_cast<std::ptrdiff_t>(nz + nv));
    Vec z(x.begin() + static_cast<std::ptrdiff_t>(nz + nv), x.end());
    const Vec transformed_r = matvec(operators.PZ_z, r);
    std::copy(transformed_r.begin(), transformed_r.end(), result.begin());
    const Vec transformed_phi =
        matvec(operators.PR_v, matvec(operators.PZ_v, phi));
    std::copy(transformed_phi.begin(), transformed_phi.end(),
              result.begin() + static_cast<std::ptrdiff_t>(nz));
    for (std::size_t i = 0; i < z.size(); ++i) {
        z[i] *= operators.Rbar_r[i];
    }
    z = matvec(operators.PR_r, z);
    for (std::size_t i = 0; i < z.size(); ++i) {
        z[i] /= operators.Rbar_r[i];
    }
    std::copy(z.begin(), z.end(),
              result.begin() + static_cast<std::ptrdiff_t>(nz + nv));
    (void)nz;
    (void)nc;
    return result;
}

Vec apply_t2_inverse(const Operators& operators, const Vec& x) {
    const std::size_t nc = operators.Nc;
    const std::size_t nr = operators.PR_r.rows;
    const std::size_t nz = operators.PZ_z.rows;
    const std::size_t nv = operators.PR_v.rows;
    if (x.size() != nz + nv + nr) {
        throw NumericalError("T2 inverse input size mismatch");
    }
    Vec result(x.size());
    const Vec r = solve(
        operators.PZ_z_factor,
        Vec(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(nz)));
    const Vec phi =
        solve(operators.PZ_v_factor,
              solve(operators.PR_v_factor,
                    Vec(x.begin() + static_cast<std::ptrdiff_t>(nz),
                        x.begin() + static_cast<std::ptrdiff_t>(nz + nv))));
    Vec z(x.begin() + static_cast<std::ptrdiff_t>(nz + nv), x.end());
    for (std::size_t i = 0; i < z.size(); ++i) {
        z[i] *= operators.Rbar_r[i];
    }
    z = solve(operators.PR_r_factor, z);
    for (std::size_t i = 0; i < z.size(); ++i) {
        z[i] /= operators.Rbar_r[i];
    }
    std::copy(r.begin(), r.end(), result.begin());
    std::copy(phi.begin(), phi.end(),
              result.begin() + static_cast<std::ptrdiff_t>(nz));
    std::copy(z.begin(), z.end(),
              result.begin() + static_cast<std::ptrdiff_t>(nz + nv));
    (void)nz;
    (void)nc;
    return result;
}

Vec apply_t3(const Operators& operators, const Vec& x) {
    if (x.size() != operators.PR_v.rows) {
        throw NumericalError("T3 input size mismatch");
    }
    Vec result = x;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] *= operators.Rbar_v[i];
    }
    result = matvec(operators.PZ_v, result);
    result = matvec(operators.PR_v, result);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] /= operators.Rbar_v[i];
    }
    return result;
}

Vec apply_t3_inverse(const Operators& operators, const Vec& x) {
    if (x.size() != operators.PR_v.rows) {
        throw NumericalError("T3 inverse input size mismatch");
    }
    Vec result = x;
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] *= operators.Rbar_v[i];
    }
    result = solve(operators.PR_v_factor, result);
    result = solve(operators.PZ_v_factor, result);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] /= operators.Rbar_v[i];
    }
    return result;
}

}  // namespace fomhd
