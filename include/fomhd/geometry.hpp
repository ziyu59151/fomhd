#pragma once

#include "fomhd/numerics.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace fomhd {

struct Point2 {
  double R{};
  double Z{};
};

struct EntityKey {
  std::size_t first{};
  std::size_t second{};

  auto operator<=>(const EntityKey &) const = default;
};

struct EntityTable {
  std::string name;
  std::vector<EntityKey> keys;
  std::vector<std::size_t> dense_to_active;
  std::vector<std::size_t> active_to_dense;
  std::vector<double> R;
  std::vector<double> Z;
  std::vector<std::uint8_t> boundary;
  std::map<EntityKey, std::size_t> lookup;

  bool contains(std::size_t first, std::size_t second) const;
  std::size_t find(std::size_t first, std::size_t second) const;
  std::size_t dense_index(const EntityKey &key) const;
};

enum class WallComponent { r_face, z_face };

struct Wall {
  WallComponent component{};
  std::size_t face{};
  int sign{};
  std::size_t near_cell{};
  std::size_t next_cell{};
  std::array<std::size_t, 2> vertices{};
  double area_per_radian{};
};

struct GeometryOptions {
  std::size_t memory_cap_bytes{256U * 1024U * 1024U};
  double grid_rtol{1e-12};
};

struct GeometryDiagnostics {
  double hausdorff_lower_m{};
  double hausdorff_upper_m{};
  double signed_polygon_area{};
  std::size_t active_cells{};
  std::size_t wall_segments{};
};

struct Geometry {
  std::vector<double> R;
  std::vector<double> Z;
  std::vector<Point2> polygon_input;
  std::vector<Point2> polygon;
  std::vector<std::uint8_t> active;

  EntityTable cells;
  EntityTable r_faces;
  EntityTable z_faces;
  EntityTable vertices;

  std::vector<double> M0;
  std::vector<double> M1;
  std::vector<double> MB;
  std::vector<double> lr;
  std::vector<double> lz;
  std::vector<double> av;
  std::vector<double> wv;
  std::vector<double> Rbar_r;
  std::vector<double> Rbar_v;
  std::vector<double> cell_wall_distance;
  std::vector<Wall> walls;
  std::vector<Point2> wall_polyline;
  GeometryDiagnostics diagnostics;

  bool is_active(std::size_t i, std::size_t j) const;
  std::size_t cell_id(std::size_t i, std::size_t j) const;
  std::size_t r_face_id(std::size_t e, std::size_t j) const;
  std::size_t z_face_id(std::size_t i, std::size_t f) const;
  std::size_t vertex_id(std::size_t e, std::size_t f) const;
};

Geometry build_geometry(RealVec R, RealVec Z, std::vector<Point2> polygon,
                        GeometryOptions options = {});
Geometry build_rectangle(std::size_t nr, std::size_t nz, double r0, double z0,
                         double h_r, double h_z, GeometryOptions options = {});
Geometry build_stepped_disk(std::size_t nr, std::size_t nz, double r0,
                            double z0, double h_r, double h_z, double center_r,
                            double center_z, double radius,
                            std::size_t polygon_sides = 96,
                            GeometryOptions options = {});

double point_segment_distance(Point2 point, Point2 a, Point2 b);

} // namespace fomhd
