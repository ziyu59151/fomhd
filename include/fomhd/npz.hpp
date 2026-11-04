#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fomhd::io {

// Canonical data is little-endian, C-order, and has exactly the dtype in descr.
struct Array {
  std::string descr;
  std::vector<std::size_t> shape;
  std::vector<std::byte> data;
};

struct Limits {
  std::size_t max_input_bytes{256U * 1024U * 1024U};
  std::size_t max_expanded_bytes{512U * 1024U * 1024U};
  std::size_t max_entries{4096U};
  std::size_t max_header_bytes{1024U * 1024U};
};

using Keys = std::set<std::string, std::less<>>;

[[nodiscard]] Array from_doubles(std::span<const double> values,
                                 std::vector<std::size_t> shape = {});
[[nodiscard]] Array from_complex(std::span<const std::complex<double>> values,
                                 std::vector<std::size_t> shape = {});
[[nodiscard]] Array from_uint64(std::uint64_t value);
[[nodiscard]] Array from_double(double value);
[[nodiscard]] Array from_bool(bool value);
[[nodiscard]] Array from_utf8(std::string_view value);

[[nodiscard]] std::vector<double> to_doubles(const Array &array);
[[nodiscard]] std::vector<std::complex<double>> to_complex(const Array &array);
[[nodiscard]] std::uint64_t to_uint64(const Array &array);
[[nodiscard]] double to_double(const Array &array);
[[nodiscard]] bool to_bool(const Array &array);
[[nodiscard]] std::string to_utf8(const Array &array);

[[nodiscard]] Array parse_npy(std::span<const std::byte> bytes,
                              const Limits &limits = {});
[[nodiscard]] std::vector<std::byte> encode_npy(const Array &array);

[[nodiscard]] std::map<std::string, Array>
read_npz(const std::filesystem::path &path, const Limits &limits,
         std::optional<Keys> allowed_keys = std::nullopt);

class NpzWriteError : public std::runtime_error {
public:
  std::string phase;
  std::filesystem::path temporary;
  bool target_may_be_published;
  NpzWriteError(std::string p, std::filesystem::path t, bool published,
                const std::string &detail)
      : std::runtime_error("NPZ " + p + ": " + detail +
                           "; temporary=" + t.string()),
        phase(std::move(p)), temporary(std::move(t)),
        target_may_be_published(published) {}
};
// The caller owns exclusive creation, publication, directory fsync and failure
// retention.
void write_npz_temporary(const std::filesystem::path &path,
                         const std::map<std::string, Array> &arrays);

void write_npz(const std::filesystem::path &path,
               const std::map<std::string, Array> &arrays);

} // namespace fomhd::io
