#include "fomhd/geometry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fomhd {
namespace {

constexpr std::size_t kInvalid = std::numeric_limits<std::size_t>::max();

bool finite_point(const Point2& point) {
    return std::isfinite(point.R) && std::isfinite(point.Z);
}

bool same_point(const Point2& left, const Point2& right) {
    return left.R == right.R && left.Z == right.Z;
}

long double orientation(Point2 a, Point2 b, Point2 c) {
    const long double ar = static_cast<long double>(a.R);
    const long double az = static_cast<long double>(a.Z);
    const long double br = static_cast<long double>(b.R);
    const long double bz = static_cast<long double>(b.Z);
    const long double cr = static_cast<long double>(c.R);
    const long double cz = static_cast<long double>(c.Z);
    // Axis-aligned exact equalities are certified ties for half-open predicates.
    if((a.R==b.R && c.R==a.R)||(a.Z==b.Z && c.Z==a.Z)||
       same_point(c,a)||same_point(c,b))return 0;
    const long double left=(br-ar)*(cz-az),right=(bz-az)*(cr-ar);
    const long double det=left-right;
    const long double bound=16*std::numeric_limits<long double>::epsilon()*
        ((std::abs(br)+std::abs(ar))*(std::abs(cz)+std::abs(az))+
         (std::abs(bz)+std::abs(az))*(std::abs(cr)+std::abs(ar)));
    if(!std::isfinite(det)||std::abs(det)<=bound)
        throw NumericalError("GeometryAmbiguous: orientation sign cannot be certified");
    return det;
}

bool within_bounds(Point2 point, Point2 a, Point2 b) {
    return std::min(a.R, b.R) <= point.R && point.R <= std::max(a.R, b.R) &&
           std::min(a.Z, b.Z) <= point.Z && point.Z <= std::max(a.Z, b.Z);
}

bool segments_intersect(Point2 a, Point2 b, Point2 c, Point2 d) {
    const long double o1 = orientation(a, b, c);
    const long double o2 = orientation(a, b, d);
    const long double o3 = orientation(c, d, a);
    const long double o4 = orientation(c, d, b);
    if (((o1 > 0.0L && o2 < 0.0L) || (o1 < 0.0L && o2 > 0.0L)) &&
        ((o3 > 0.0L && o4 < 0.0L) || (o3 < 0.0L && o4 > 0.0L))) {
        return true;
    }
    if (o1 == 0.0L && within_bounds(c, a, b)) {
        return true;
    }
    if (o2 == 0.0L && within_bounds(d, a, b)) {
        return true;
    }
    if (o3 == 0.0L && within_bounds(a, c, d)) {
        return true;
    }
    if (o4 == 0.0L && within_bounds(b, c, d)) {
        return true;
    }
    return false;
}

long double signed_area(const std::vector<Point2>& polygon) {
    long double area = 0.0L,scale=0.0L;
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const Point2& a = polygon[i];
        const Point2& b = polygon[(i + 1) % polygon.size()];
        const long double left=static_cast<long double>(a.R)*b.Z,right=static_cast<long double>(b.R)*a.Z;
        area+=left-right;scale+=std::abs(left)+std::abs(right);
    }
    if(std::abs(area)<=16*polygon.size()*std::numeric_limits<long double>::epsilon()*scale)
        throw NumericalError("GeometryAmbiguous: polygon area cannot be certified");
    return area / 2.0L;
}

bool inside_polygon(double R, double Z, const std::vector<Point2>& polygon) {
    bool inside = false;
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        const Point2& a = polygon[i];
        const Point2& b = polygon[(i + 1) % polygon.size()];
        if ((a.Z <= Z && Z < b.Z) || (b.Z <= Z && Z < a.Z)) {
            const auto side=orientation(a,b,{R,Z});
            // side/(b.Z-a.Z) has the sign of crossing-R, without division.
            if((b.Z>a.Z && side>0)||(b.Z<a.Z && side<0))inside=!inside;
        }
    }
    return inside;
}

void validate_polygon(std::vector<Point2>& polygon_input,
                      std::vector<Point2>& polygon) {
    if (polygon_input.size() < 3) {
        throw NumericalError("polygon requires at least three vertices");
    }
    for (const Point2& point : polygon_input) {
        if (!finite_point(point)) {
            throw NumericalError("polygon contains a non-finite coordinate");
        }
    }
    polygon = polygon_input;
    if (polygon.size() > 1 && same_point(polygon.front(), polygon.back())) {
        polygon.pop_back();
    }
    if (polygon.size() < 3) {
        throw NumericalError("polygon has fewer than three unique boundary vertices");
    }
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        for (std::size_t j = i + 1; j < polygon.size(); ++j) {
            if (same_point(polygon[i], polygon[j])) {
                throw NumericalError("polygon has a duplicate vertex");
            }
        }
        const Point2& a = polygon[i];
        const Point2& b = polygon[(i + 1) % polygon.size()];
        if (a.R == b.R && a.Z == b.Z) {
            throw NumericalError("polygon has a zero-length segment");
        }
    }
    const std::size_t count = polygon.size();
    for (std::size_t i = 0; i < count; ++i) {
        const Point2 a = polygon[i];
        const Point2 b = polygon[(i + 1) % count];
        for (std::size_t j = i + 1; j < count; ++j) {
            if (j == i || j == (i + 1) % count ||
                (j + 1) % count == i) {
                continue;
            }
            const Point2 c = polygon[j];
            const Point2 d = polygon[(j + 1) % count];
            if (segments_intersect(a, b, c, d)) {
                throw NumericalError("polygon is self-intersecting or self-touching");
            }
        }
    }
    const long double area = signed_area(polygon);
    if (area == 0.0L || !std::isfinite(area)) {
        throw NumericalError("polygon has zero or non-finite signed area");
    }
}

void validate_grid(const RealVec& R, const RealVec& Z, double rtol) {
    if(!std::isfinite(rtol)||rtol<=0||rtol>=1)throw NumericalError("invalid grid relative tolerance");
    if (R.size() < 2 || Z.size() < 2) {
        throw NumericalError("grid requires at least two nodes per direction");
    }
    for (double value : R) {
        if (!std::isfinite(value)) {
            throw NumericalError("R grid contains a non-finite value");
        }
    }
    for (double value : Z) {
        if (!std::isfinite(value)) {
            throw NumericalError("Z grid contains a non-finite value");
        }
    }
    const double h_r = R[1] - R[0];
    const double h_z = Z[1] - Z[0];
    if (!(h_r > 0.0 && h_z > 0.0) || R[0] - 0.5 * h_r <= 0.0) {
        throw NumericalError("grid spacing is non-positive or reaches R<=0");
    }
    for (std::size_t i = 1; i < R.size(); ++i) {
        if(!(R[i]>R[i-1]))throw NumericalError("R grid is not strictly increasing");
        const double expected = R[0] + static_cast<double>(i) * h_r;
        if (std::abs(R[i] - expected) > rtol * std::abs(h_r) +
            8*std::numeric_limits<double>::epsilon()*i*std::max(std::abs(R[0]),std::abs(R[i]))) {
            throw NumericalError("R grid is not uniform");
        }
    }
    for (std::size_t j = 1; j < Z.size(); ++j) {
        if(!(Z[j]>Z[j-1]))throw NumericalError("Z grid is not strictly increasing");
        const double expected = Z[0] + static_cast<double>(j) * h_z;
        if (std::abs(Z[j] - expected) > rtol * std::abs(h_z) +
            8*std::numeric_limits<double>::epsilon()*j*std::max(std::abs(Z[0]),std::abs(Z[j]))) {
            throw NumericalError("Z grid is not uniform");
        }
    }
}

std::vector<EntityKey> ordered_keys(const std::set<EntityKey>& source) {
    std::vector<EntityKey> keys(source.begin(), source.end());
    std::sort(keys.begin(), keys.end(), [](const EntityKey& left,
                                           const EntityKey& right) {
        return left.second < right.second ||
               (left.second == right.second && left.first < right.first);
    });
    return keys;
}

EntityTable make_table(const std::string& name, const std::set<EntityKey>& source,
                       std::size_t width, std::size_t height) {
    EntityTable table;
    table.name = name;
    table.keys = ordered_keys(source);
    const std::size_t dense_size =
        checked_product(width, height, name + " dense table");
    table.dense_to_active.assign(dense_size, kInvalid);
    table.active_to_dense.resize(table.keys.size());
    table.R.resize(table.keys.size());
    table.Z.resize(table.keys.size());
    table.boundary.assign(table.keys.size(), 0);
    for (std::size_t id = 0; id < table.keys.size(); ++id) {
        const EntityKey key = table.keys[id];
        if (key.first >= width || key.second >= height) {
            throw NumericalError(name + " key lies outside dense dimensions");
        }
        const std::size_t dense = key.first + width * key.second;
        if (table.dense_to_active[dense] != kInvalid) {
            throw NumericalError(name + " dense map is not one-to-one");
        }
        table.dense_to_active[dense] = id;
        table.active_to_dense[id] = dense;
        table.lookup.emplace(key, id);
    }
    return table;
}

std::size_t neighbor_count(const Geometry& geometry, std::size_t first,
                           std::size_t second, bool radial_face) {
    if (radial_face) {
        std::size_t count = 0;
        if (first != 0) {
            count += geometry.cells.contains(first - 1, second) ? 1U : 0U;
        }
        count += geometry.cells.contains(first, second) ? 1U : 0U;
        return count;
    }
    std::size_t count = 0;
    if (second != 0) {
        count += geometry.cells.contains(first, second - 1) ? 1U : 0U;
    }
    count += geometry.cells.contains(first, second) ? 1U : 0U;
    return count;
}

void validate_wall_support(const Geometry& geometry, Wall& wall) {
    const std::size_t i = wall.near_cell == kInvalid
                              ? kInvalid
                              : geometry.cells.keys[wall.near_cell].first;
    const std::size_t j = wall.near_cell == kInvalid
                              ? kInvalid
                              : geometry.cells.keys[wall.near_cell].second;
    if (wall.component == WallComponent::r_face) {
        const std::size_t next_i =
            wall.sign > 0 ? i - 1 : i + 1;
        if (!geometry.cells.contains(next_i, j)) {
            throw NumericalError("radial wall lacks two-point inward support");
        }
        wall.next_cell = geometry.cells.find(next_i, j);
    } else {
        const std::size_t next_j =
            wall.sign > 0 ? j - 1 : j + 1;
        if (!geometry.cells.contains(i, next_j)) {
            throw NumericalError("axial wall lacks two-point inward support");
        }
        wall.next_cell = geometry.cells.find(i, next_j);
    }
}

void enumerate_topology(Geometry& geometry) {
    std::set<EntityKey> cell_set;
    std::set<EntityKey> r_set;
    std::set<EntityKey> z_set;
    std::set<EntityKey> v_set;
    for (std::size_t j = 0; j < geometry.Z.size(); ++j) {
        for (std::size_t i = 0; i < geometry.R.size(); ++i) {
            if (!geometry.is_active(i, j)) {
                continue;
            }
            cell_set.insert({i, j});
            r_set.insert({i, j});
            r_set.insert({i + 1, j});
            z_set.insert({i, j});
            z_set.insert({i, j + 1});
            v_set.insert({i, j});
            v_set.insert({i + 1, j});
            v_set.insert({i, j + 1});
            v_set.insert({i + 1, j + 1});
        }
    }
    geometry.cells = make_table("cells", cell_set, geometry.R.size(),
                                geometry.Z.size());
    geometry.r_faces =
        make_table("r_faces", r_set, geometry.R.size() + 1, geometry.Z.size());
    geometry.z_faces = make_table("z_faces", z_set, geometry.R.size(),
                                  geometry.Z.size() + 1);
    geometry.vertices = make_table("vertices", v_set, geometry.R.size() + 1,
                                   geometry.Z.size() + 1);
    for (std::size_t id = 0; id < geometry.cells.keys.size(); ++id) {
        geometry.cells.R[id] = geometry.R[geometry.cells.keys[id].first];
        geometry.cells.Z[id] = geometry.Z[geometry.cells.keys[id].second];
    }
    const double h_r = geometry.R[1] - geometry.R[0];
    const double h_z = geometry.Z[1] - geometry.Z[0];
    for (std::size_t id = 0; id < geometry.r_faces.keys.size(); ++id) {
        geometry.r_faces.R[id] =
            geometry.R[0] +
            (static_cast<double>(geometry.r_faces.keys[id].first) - 0.5) * h_r;
        geometry.r_faces.Z[id] =
            geometry.Z[geometry.r_faces.keys[id].second];
    }
    for (std::size_t id = 0; id < geometry.z_faces.keys.size(); ++id) {
        geometry.z_faces.R[id] =
            geometry.R[geometry.z_faces.keys[id].first];
        geometry.z_faces.Z[id] =
            geometry.Z[0] +
            (static_cast<double>(geometry.z_faces.keys[id].second) - 0.5) * h_z;
    }
    for (std::size_t id = 0; id < geometry.vertices.keys.size(); ++id) {
        geometry.vertices.R[id] =
            geometry.R[0] +
            (static_cast<double>(geometry.vertices.keys[id].first) - 0.5) * h_r;
        geometry.vertices.Z[id] =
            geometry.Z[0] +
            (static_cast<double>(geometry.vertices.keys[id].second) - 0.5) * h_z;
    }

    for (std::size_t id = 0; id < geometry.r_faces.keys.size(); ++id) {
        const EntityKey key = geometry.r_faces.keys[id];
        const std::size_t count =
            neighbor_count(geometry, key.first, key.second, true);
        if (count != 1 && count != 2) {
            throw NumericalError("radial face does not have one or two cell neighbors");
        }
        geometry.r_faces.boundary[id] = count == 1 ? 1 : 0;
    }
    for (std::size_t id = 0; id < geometry.z_faces.keys.size(); ++id) {
        const EntityKey key = geometry.z_faces.keys[id];
        const std::size_t count =
            neighbor_count(geometry, key.first, key.second, false);
        if (count != 1 && count != 2) {
            throw NumericalError("axial face does not have one or two cell neighbors");
        }
        geometry.z_faces.boundary[id] = count == 1 ? 1 : 0;
    }
    for (std::size_t id = 0; id < geometry.vertices.keys.size(); ++id) {
        const EntityKey key = geometry.vertices.keys[id];
        std::array<bool, 4> active{};
        if (key.first > 0 && key.second > 0) {
            active[0] = geometry.cells.contains(key.first - 1, key.second - 1);
        }
        active[1] = geometry.cells.contains(key.first, key.second - 1);
        if (key.second > 0) {
            active[2] = geometry.cells.contains(key.first - 1, key.second);
        }
        active[3] = geometry.cells.contains(key.first, key.second);
        const std::size_t count =
            static_cast<std::size_t>(std::count(active.begin(), active.end(), true));
        if (count == 2 && !((active[0] && active[1]) ||
                            (active[0] && active[2]) ||
                            (active[1] && active[3]) ||
                            (active[2] && active[3]))) {
            throw NumericalError("vertices meet only diagonally");
        }
        geometry.vertices.boundary[id] = count < 4 ? 1 : 0;
    }

    const std::size_t first_cell = 0;
    std::vector<std::uint8_t> visited(geometry.cells.keys.size(), 0);
    std::vector<std::size_t> stack{first_cell};
    visited[first_cell] = 1;
    std::size_t reached = 0;
    while (!stack.empty()) {
        const std::size_t cell = stack.back();
        stack.pop_back();
        ++reached;
        const EntityKey key = geometry.cells.keys[cell];
        const std::array<std::pair<long, long>, 4> offsets{
            std::pair{-1, 0}, std::pair{1, 0}, std::pair{0, -1},
            std::pair{0, 1}};
        for (const auto& [di, dj] : offsets) {
            const long ni = static_cast<long>(key.first) + di;
            const long nj = static_cast<long>(key.second) + dj;
            if (ni < 0 || nj < 0 ||
                static_cast<std::size_t>(ni) >= geometry.R.size() ||
                static_cast<std::size_t>(nj) >= geometry.Z.size() ||
                !geometry.cells.contains(static_cast<std::size_t>(ni),
                                         static_cast<std::size_t>(nj))) {
                continue;
            }
            const std::size_t neighbor =
                geometry.cells.find(static_cast<std::size_t>(ni),
                                    static_cast<std::size_t>(nj));
            if (visited[neighbor] == 0) {
                visited[neighbor] = 1;
                stack.push_back(neighbor);
            }
        }
    }
    if (reached != geometry.cells.keys.size()) {
        throw NumericalError("active cell domain is disconnected");
    }

    for (std::size_t id = 0; id < geometry.r_faces.keys.size(); ++id) {
        if (geometry.r_faces.boundary[id] == 0) {
            continue;
        }
        const EntityKey key = geometry.r_faces.keys[id];
        const bool left = key.first > 0 && geometry.cells.contains(key.first - 1, key.second);
        const bool right = geometry.cells.contains(key.first, key.second);
        if (left == right) {
            throw NumericalError("radial wall face neighbor selection is ambiguous");
        }
        Wall wall;
        wall.component = WallComponent::r_face;
        wall.face = id;
        wall.sign = right ? -1 : 1;
        wall.near_cell =
            geometry.cells.find(right ? key.first : key.first - 1, key.second);
        wall.vertices = {geometry.vertices.find(key.first, key.second),
                         geometry.vertices.find(key.first, key.second + 1)};
        validate_wall_support(geometry, wall);
        geometry.cells.boundary[wall.near_cell] = 1;
        geometry.cells.boundary[wall.next_cell] = 1;
        geometry.walls.push_back(wall);
    }
    for (std::size_t id = 0; id < geometry.z_faces.keys.size(); ++id) {
        if (geometry.z_faces.boundary[id] == 0) {
            continue;
        }
        const EntityKey key = geometry.z_faces.keys[id];
        const bool lower =
            key.second > 0 && geometry.cells.contains(key.first, key.second - 1);
        const bool upper = geometry.cells.contains(key.first, key.second);
        if (lower == upper) {
            throw NumericalError("axial wall face neighbor selection is ambiguous");
        }
        Wall wall;
        wall.component = WallComponent::z_face;
        wall.face = id;
        wall.sign = upper ? -1 : 1;
        wall.near_cell =
            geometry.cells.find(key.first, upper ? key.second : key.second - 1);
        wall.vertices = {geometry.vertices.find(key.first, key.second),
                         geometry.vertices.find(key.first + 1, key.second)};
        validate_wall_support(geometry, wall);
        geometry.cells.boundary[wall.near_cell] = 1;
        geometry.cells.boundary[wall.next_cell] = 1;
        geometry.walls.push_back(wall);
    }
    std::sort(geometry.walls.begin(), geometry.walls.end(),
              [](const Wall& left, const Wall& right) {
                  if (left.component != right.component) {
                      return left.component < right.component;
                  }
                  return left.face < right.face;
              });
    if (geometry.walls.empty()) {
        throw NumericalError("active domain has no boundary wall");
    }

    std::vector<std::vector<std::size_t>> incidence(
        geometry.vertices.keys.size());
    for (std::size_t id = 0; id < geometry.walls.size(); ++id) {
        incidence[geometry.walls[id].vertices[0]].push_back(id);
        incidence[geometry.walls[id].vertices[1]].push_back(id);
    }
    for (const auto& segments : incidence) {
        if (!segments.empty() && segments.size() != 2) {
            throw NumericalError("wall graph is nonmanifold");
        }
    }
    std::vector<std::uint8_t> wall_visited(geometry.walls.size(), 0);
    std::vector<std::size_t> wall_stack{0};
    wall_visited[0] = 1;
    std::size_t wall_reached = 0;
    while (!wall_stack.empty()) {
        const std::size_t wall = wall_stack.back();
        wall_stack.pop_back();
        ++wall_reached;
        for (std::size_t vertex : geometry.walls[wall].vertices) {
            for (std::size_t next : incidence[vertex]) {
                if (wall_visited[next] == 0) {
                    wall_visited[next] = 1;
                    wall_stack.push_back(next);
                }
            }
        }
    }
    if (wall_reached != geometry.walls.size()) {
        throw NumericalError("wall graph is disconnected");
    }
    const long long euler =
        static_cast<long long>(geometry.vertices.keys.size()) -
        static_cast<long long>(geometry.r_faces.keys.size()) -
        static_cast<long long>(geometry.z_faces.keys.size()) +
        static_cast<long long>(geometry.cells.keys.size());
    if (euler != 1) {
        throw NumericalError("active domain does not have disk Euler characteristic");
    }

    std::size_t current_vertex = geometry.walls.front().vertices[0];
    std::size_t previous_wall = kInvalid;
    geometry.wall_polyline.push_back(
        Point2{geometry.vertices.R[current_vertex],
               geometry.vertices.Z[current_vertex]});
    for (std::size_t step = 0; step < geometry.walls.size(); ++step) {
        std::size_t next_wall = kInvalid;
        for (std::size_t candidate : incidence[current_vertex]) {
            if (candidate != previous_wall) {
                next_wall = candidate;
                break;
            }
        }
        if (next_wall == kInvalid) {
            throw NumericalError("wall cycle traversal failed");
        }
        const Wall& wall = geometry.walls[next_wall];
        current_vertex =
            wall.vertices[0] == current_vertex ? wall.vertices[1] : wall.vertices[0];
        geometry.wall_polyline.push_back(
            Point2{geometry.vertices.R[current_vertex],
                   geometry.vertices.Z[current_vertex]});
        previous_wall = next_wall;
    }
}

void compute_metrics(Geometry& geometry) {
    const double h_r = geometry.R[1] - geometry.R[0];
    const double h_z = geometry.Z[1] - geometry.Z[0];
    const std::size_t nc = geometry.cells.keys.size();
    const std::size_t nr = geometry.r_faces.keys.size();
    const std::size_t nz = geometry.z_faces.keys.size();
    const std::size_t nv = geometry.vertices.keys.size();
    geometry.M0.resize(nc);
    geometry.lr.resize(nr);
    geometry.lz.resize(nz);
    geometry.Rbar_r.resize(nr);
    geometry.av.assign(nv, 0.0);
    geometry.wv.assign(nv, 0.0);
    geometry.Rbar_v.resize(nv);
    std::vector<double> r_weight(nr);
    std::vector<double> c_weight(nc);
    std::vector<double> z_weight(nz);

    for (std::size_t cell = 0; cell < nc; ++cell) {
        const double R = geometry.cells.R[cell];
        geometry.M0[cell] = R * h_r * h_z;
        c_weight[cell] = geometry.M0[cell];
    }
    for (std::size_t face = 0; face < nr; ++face) {
        const EntityKey key = geometry.r_faces.keys[face];
        const std::size_t count = neighbor_count(geometry, key.first, key.second, true);
        geometry.lr[face] = count == 2 ? h_r : 0.5 * h_r;
        const double Rr = geometry.r_faces.R[face];
        if (count == 1) {
            const std::size_t cell =
                key.first > 0 && geometry.cells.contains(key.first - 1, key.second)
                    ? geometry.cells.find(key.first - 1, key.second)
                    : geometry.cells.find(key.first, key.second);
            geometry.Rbar_r[face] = 0.5 * (Rr + geometry.cells.R[cell]);
        } else {
            geometry.Rbar_r[face] = Rr;
        }
        r_weight[face] = Rr * h_z * geometry.lr[face];
    }
    for (std::size_t face = 0; face < nz; ++face) {
        const EntityKey key = geometry.z_faces.keys[face];
        const std::size_t count =
            neighbor_count(geometry, key.first, key.second, false);
        geometry.lz[face] = count == 2 ? h_z : 0.5 * h_z;
        const double Rz = geometry.z_faces.R[face];
        z_weight[face] = Rz * h_r * geometry.lz[face];
    }
    for (std::size_t cell = 0; cell < nc; ++cell) {
        const EntityKey key = geometry.cells.keys[cell];
        const double Rc = geometry.cells.R[cell];
        for (std::size_t de = 0; de < 2; ++de) {
            for (std::size_t df = 0; df < 2; ++df) {
                const std::size_t e = key.first + de;
                const std::size_t f = key.second + df;
                const std::size_t vertex = geometry.vertices.find(e, f);
                const double Rhalf =
                    geometry.R[0] + (static_cast<double>(e) - 0.5) * h_r;
                const double Rlo = std::min(Rc, Rhalf);
                const double Rhi = std::max(Rc, Rhalf);
                geometry.av[vertex] += h_r * h_z / 4.0;
                geometry.wv[vertex] +=
                    (Rhi * Rhi - Rlo * Rlo) * h_z / 4.0;
            }
        }
    }
    for (std::size_t vertex = 0; vertex < nv; ++vertex) {
        if (!(geometry.av[vertex] > 0.0) || !(geometry.wv[vertex] > 0.0)) {
            throw NumericalError("vertex area or cylindrical weight is non-positive");
        }
        geometry.Rbar_v[vertex] =
            geometry.wv[vertex] / geometry.av[vertex];
    }
    geometry.M1.reserve(nr + nc + nz);
    geometry.M1.insert(geometry.M1.end(), r_weight.begin(), r_weight.end());
    geometry.M1.insert(geometry.M1.end(), c_weight.begin(), c_weight.end());
    geometry.M1.insert(geometry.M1.end(), z_weight.begin(), z_weight.end());
    geometry.MB.reserve(nz + nv + nr);
    geometry.MB.insert(geometry.MB.end(), z_weight.begin(), z_weight.end());
    geometry.MB.insert(geometry.MB.end(), geometry.wv.begin(), geometry.wv.end());
    for(std::size_t f=0;f<nr;++f)
        geometry.MB.push_back(geometry.Rbar_r[f]*h_z*geometry.lr[f]);
    for (std::size_t wall_id = 0; wall_id < geometry.walls.size(); ++wall_id) {
        Wall& wall = geometry.walls[wall_id];
        if (wall.component == WallComponent::r_face) {
            wall.area_per_radian = geometry.r_faces.R[wall.face] * h_z;
        } else {
            wall.area_per_radian = geometry.z_faces.R[wall.face] * h_r;
        }
    }
    geometry.diagnostics.wall_segments = geometry.walls.size();
}

double segment_length(Point2 a, Point2 b) {
    return std::hypot(b.R - a.R, b.Z - a.Z);
}

double distance_to_segments(Point2 point,
                            const std::vector<std::pair<Point2, Point2>>& segments) {
    double result = std::numeric_limits<double>::infinity();
    for (const auto& [a, b] : segments) {
        result = std::min(result, point_segment_distance(point, a, b));
    }
    return result;
}

void compute_geometry_distance(Geometry& geometry) {
    const double h_r = geometry.R[1] - geometry.R[0];
    const double h_z = geometry.Z[1] - geometry.Z[0];
    const double eta = std::min(h_r, h_z) / 8.0;
    std::vector<std::pair<Point2, Point2>> wall_segments;
    wall_segments.reserve(geometry.walls.size());
    for (const Wall& wall : geometry.walls) {
        wall_segments.emplace_back(
            Point2{geometry.vertices.R[wall.vertices[0]],
                   geometry.vertices.Z[wall.vertices[0]]},
            Point2{geometry.vertices.R[wall.vertices[1]],
                   geometry.vertices.Z[wall.vertices[1]]});
    }
    std::vector<std::pair<Point2, Point2>> polygon_segments;
    polygon_segments.reserve(geometry.polygon.size());
    for (std::size_t i = 0; i < geometry.polygon.size(); ++i) {
        polygon_segments.emplace_back(
            geometry.polygon[i],
            geometry.polygon[(i + 1) % geometry.polygon.size()]);
    }
    double lower = 0.0;
    auto accumulate = [&](const std::vector<std::pair<Point2, Point2>>& source,
                          const std::vector<std::pair<Point2, Point2>>& target) {
        for (const auto& [a, b] : source) {
            const double length = segment_length(a, b);
            const std::size_t samples =
                std::max<std::size_t>(1U, static_cast<std::size_t>(
                    std::ceil(length / eta)));
            for (std::size_t sample = 0; sample <= samples; ++sample) {
                const double t =
                    static_cast<double>(sample) / static_cast<double>(samples);
                const Point2 point{a.R + t * (b.R - a.R),
                                   a.Z + t * (b.Z - a.Z)};
                lower =
                    std::max(lower, distance_to_segments(point, target));
            }
        }
    };
    accumulate(polygon_segments, wall_segments);
    accumulate(wall_segments, polygon_segments);
    geometry.diagnostics.hausdorff_lower_m = lower;
    geometry.diagnostics.hausdorff_upper_m = lower + eta / 2.0;
    geometry.diagnostics.active_cells = geometry.cells.keys.size();

    geometry.cell_wall_distance.resize(geometry.cells.keys.size());
    for (std::size_t cell = 0; cell < geometry.cells.keys.size(); ++cell) {
        geometry.cell_wall_distance[cell] = distance_to_segments(
            Point2{geometry.cells.R[cell], geometry.cells.Z[cell]},
            wall_segments);
    }
}

std::size_t estimate_geometry_bytes(std::size_t nr, std::size_t nz,
                                    std::size_t polygon_vertices) {
    const std::size_t grid = checked_product(nr, nz, "geometry estimate");
    const std::size_t dense_entities =
        checked_sum(grid, checked_product(checked_sum(nr, 1,
                                                      "geometry estimate"), nz,
                                          "geometry estimate"),
                    "geometry estimate");
    const std::size_t total_entities =
        checked_sum(dense_entities,
                    checked_product(nr, checked_sum(nz, 1,
                                                    "geometry estimate"),
                                    "geometry estimate"),
                    "geometry estimate");
    return checked_sum(
        checked_sum(checked_product(grid, 128, "geometry estimate"),
                    checked_product(total_entities, 96, "geometry estimate"),
                    "geometry estimate"),
        checked_product(polygon_vertices, 128, "geometry estimate"),
        "geometry estimate");
}

}  // namespace

bool EntityTable::contains(std::size_t first, std::size_t second) const {
    return lookup.find(EntityKey{first, second}) != lookup.end();
}

std::size_t EntityTable::find(std::size_t first, std::size_t second) const {
    const auto iterator = lookup.find(EntityKey{first, second});
    if (iterator == lookup.end()) {
        throw NumericalError(name + " entity is absent");
    }
    return iterator->second;
}

std::size_t EntityTable::dense_index(const EntityKey& key) const {
    const auto iterator = lookup.find(key);
    if (iterator == lookup.end()) {
        throw NumericalError(name + " entity is absent");
    }
    return active_to_dense[iterator->second];
}

bool Geometry::is_active(std::size_t i, std::size_t j) const {
    if (i >= R.size() || j >= Z.size()) {
        return false;
    }
    return active[i + R.size() * j] != 0;
}

std::size_t Geometry::cell_id(std::size_t i, std::size_t j) const {
    return cells.find(i, j);
}

std::size_t Geometry::r_face_id(std::size_t e, std::size_t j) const {
    return r_faces.find(e, j);
}

std::size_t Geometry::z_face_id(std::size_t i, std::size_t f) const {
    return z_faces.find(i, f);
}

std::size_t Geometry::vertex_id(std::size_t e, std::size_t f) const {
    return vertices.find(e, f);
}

Geometry build_geometry(RealVec R, RealVec Z, std::vector<Point2> polygon,
                        GeometryOptions options) {
    const std::size_t estimate =
        estimate_geometry_bytes(R.size(), Z.size(), polygon.size());
    require_memory(estimate,
                   MemoryBudget{"geometry", options.memory_cap_bytes});
    validate_grid(R, Z, options.grid_rtol);
    std::vector<Point2> canonical_polygon;
    validate_polygon(polygon, canonical_polygon);
    Geometry geometry;
    geometry.R = std::move(R);
    geometry.Z = std::move(Z);
    geometry.polygon_input = std::move(polygon);
    geometry.polygon = std::move(canonical_polygon);
    geometry.diagnostics.signed_polygon_area =
        static_cast<double>(signed_area(geometry.polygon));
    const double h_r = geometry.R[1] - geometry.R[0];
    const double h_z = geometry.Z[1] - geometry.Z[0];
    const double r_lo = geometry.R.front() - 0.5 * h_r;
    const double r_hi = geometry.R.back() + 0.5 * h_r;
    const double z_lo = geometry.Z.front() - 0.5 * h_z;
    const double z_hi = geometry.Z.back() + 0.5 * h_z;
    for (const Point2& point : geometry.polygon) {
        if (point.R < r_lo || point.R > r_hi || point.Z < z_lo ||
            point.Z > z_hi) {
            throw NumericalError("polygon lies outside the supplied grid rectangles");
        }
    }
    geometry.active.assign(
        checked_product(geometry.R.size(), geometry.Z.size(),
                        "active geometry mask"),
        0);
    std::size_t active_count = 0;
    for (std::size_t j = 0; j < geometry.Z.size(); ++j) {
        for (std::size_t i = 0; i < geometry.R.size(); ++i) {
            if (inside_polygon(geometry.R[i], geometry.Z[j], geometry.polygon)) {
                geometry.active[i + geometry.R.size() * j] = 1;
                ++active_count;
            }
        }
    }
    if (active_count == 0) {
        throw NumericalError("polygon masks no active cell centers");
    }
    enumerate_topology(geometry);
    compute_metrics(geometry);
    compute_geometry_distance(geometry);
    return geometry;
}

Geometry build_rectangle(std::size_t nr, std::size_t nz, double r0,
                         double z0, double h_r, double h_z,
                         GeometryOptions options) {
    if (nr < 2 || nz < 2) {
        throw NumericalError("rectangle requires at least 2 by 2 nodes");
    }
    RealVec R(nr);
    RealVec Z(nz);
    for (std::size_t i = 0; i < nr; ++i) {
        R[i] = r0 + static_cast<double>(i) * h_r;
    }
    for (std::size_t j = 0; j < nz; ++j) {
        Z[j] = z0 + static_cast<double>(j) * h_z;
    }
    const std::vector<Point2> polygon{{r0, z0},
                                      {R.back(), z0},
                                      {R.back(), Z.back()},
                                      {r0, Z.back()}};
    return build_geometry(std::move(R), std::move(Z), polygon, options);
}

Geometry build_stepped_disk(std::size_t nr, std::size_t nz, double r0,
                            double z0, double h_r, double h_z,
                            double center_r, double center_z, double radius,
                            std::size_t polygon_sides,
                            GeometryOptions options) {
    if (polygon_sides < 8 || !(radius > 0.0)) {
        throw NumericalError("stepped disk polygon input is invalid");
    }
    if (nr < 2 || nz < 2) {
        throw NumericalError("stepped disk grid is too small");
    }
    RealVec R(nr);
    RealVec Z(nz);
    for (std::size_t i = 0; i < nr; ++i) {
        R[i] = r0 + static_cast<double>(i) * h_r;
    }
    for (std::size_t j = 0; j < nz; ++j) {
        Z[j] = z0 + static_cast<double>(j) * h_z;
    }
    std::vector<Point2> polygon;
    polygon.reserve(polygon_sides);
    for (std::size_t side = 0; side < polygon_sides; ++side) {
        const double angle =
            2.0 * std::numbers::pi * static_cast<double>(side) /
            static_cast<double>(polygon_sides);
        polygon.push_back(
            Point2{center_r + radius * std::cos(angle),
                   center_z + radius * std::sin(angle)});
    }
    return build_geometry(std::move(R), std::move(Z), std::move(polygon),
                          options);
}

double point_segment_distance(Point2 point, Point2 a, Point2 b) {
    const double dr = b.R - a.R;
    const double dz = b.Z - a.Z;
    const double denominator = dr * dr + dz * dz;
    if (denominator == 0.0) {
        return std::hypot(point.R - a.R, point.Z - a.Z);
    }
    const double t =
        std::clamp(((point.R - a.R) * dr + (point.Z - a.Z) * dz) /
                       denominator,
                   0.0, 1.0);
    return std::hypot(point.R - (a.R + t * dr), point.Z - (a.Z + t * dz));
}

}  // namespace fomhd
