#pragma once

#include "fomhd/geometry.hpp"

#include <cstddef>
#include <vector>

namespace fomhd {

struct OperatorOptions {
  double rank_rtol{1e-10};
  std::size_t memory_cap_bytes{512U * 1024U * 1024U};
};

struct OperatorCheck {
  std::string name, unit;
  int mode{};
  double residual{}, scale{}, roundoff_bound{};
};
struct Operators {
  int K{};
  std::size_t Nc{};
  std::size_t Ne{};
  std::size_t Nb{};
  RealVec M0;
  RealVec M1;
  RealVec MB;
  RealVec Rbar_r;
  RealVec Rbar_v;

  Matrix b1r;
  Matrix b1z;
  Matrix b2r;
  Matrix b2z;

  Matrix PR_r;
  Matrix PZ_r;
  Matrix PR_z;
  Matrix PZ_z;
  Matrix PR_v;
  Matrix PZ_v;
  LU PR_r_factor;
  LU PZ_r_factor;
  LU PR_z_factor;
  LU PZ_z_factor;
  LU PR_v_factor;
  LU PZ_v_factor;

  std::vector<Matrix> G;
  std::vector<Matrix> C;
  std::vector<Matrix> DB;
  std::vector<Matrix> DJ;
  std::vector<Matrix> Cdag;
  std::vector<Matrix> Lphi;
  std::vector<LU> phi_factor;

  Matrix I;
  Matrix H;
  Matrix QB;
  Matrix DR;
  Matrix DZ;
  Matrix T_full;
  Matrix T;
  std::vector<std::size_t> selected_wall_rows;

  Matrix Bc_full;
  Matrix Bc;
  std::vector<std::size_t> selected_constraint_rows;
  Matrix topology_flux_row;
  std::vector<OperatorCheck> setup_checks;
  std::vector<RealVec> poisson_spectra;
};

// One Fourier mode on the RZ/Yee complex. Rows and columns follow the public
// component order documented by build_operators: G maps cell scalars to
// (R-face, cell, Z-face) edges, C maps edges to (Z-face, vertex, R-face)
// faces, DB maps faces to vertices, and DJ maps edges to cells. All actions are
// host-side FP64 complex coefficients at zero physical toroidal angle, with SI
// derivative units. Memory rejection follows OperatorOptions.
struct SpatialOperators {
  int mode{};
  std::size_t Nc{};
  std::size_t Ne{};
  std::size_t Nb{};
  Matrix G;
  Matrix C;
  Matrix DB;
  Matrix DJ;
};

Operators build_operators(const Geometry &geometry, int retained_k,
                          OperatorOptions options = {});
std::vector<OperatorCheck> validate_operators(const Geometry &,
                                              const Operators &);
SpatialOperators build_spatial_operators(const Geometry &geometry, int mode,
                                         OperatorOptions options = {});
// Applies the dimensionless T2^-1 action to block columns by batched right
// solves; no inverse matrix is formed. Columns are ordered as
// (Z-face, vertex, R-face), and irregular-mask factor order matches
// apply_t2_inverse. Memory rejection follows options.
Matrix apply_t2_inverse_block(const Operators &operators, const Matrix &block,
                              OperatorOptions options = {});
Matrix build_gradient_operator(const Geometry &geometry, int mode,
                               OperatorOptions options = {});

Vec apply_t1(const Operators &operators, const Vec &x);
Vec apply_t1_inverse(const Operators &operators, const Vec &x);
Vec apply_t2(const Operators &operators, const Vec &x);
Vec apply_t2_inverse(const Operators &operators, const Vec &x);
Vec apply_t3(const Operators &operators, const Vec &x);
Vec apply_t3_inverse(const Operators &operators, const Vec &x);

} // namespace fomhd
