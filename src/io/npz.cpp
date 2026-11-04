#include "fomhd/npz.hpp"

#include "miniz.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fomhd::io {
namespace {

constexpr std::array<unsigned char, 6> npy_magic{0x93, 'N', 'U', 'M', 'P', 'Y'};
constexpr std::size_t npy_v1_prefix{10};
constexpr std::size_t npy_v2_prefix{12};
constexpr std::size_t npy_alignment{64};
constexpr std::size_t npy_v1_max_header{
    std::numeric_limits<std::uint16_t>::max()};
constexpr std::size_t sha256_hex_size{64};

enum class ScalarKind {
  float64,
  complex128,
  uint64,
  boolean,
  unicode,
};

struct Dtype {
  ScalarKind kind{};
  std::size_t itemsize{};
  std::size_t unicode_length{};
  std::string canonical_descr;
  bool input_little_endian{true};
};

class FileDescriptor {
public:
  explicit FileDescriptor(int descriptor) noexcept : descriptor_{descriptor} {}

  ~FileDescriptor() {
    if (descriptor_ >= 0) {
      static_cast<void>(::close(descriptor_));
    }
  }

  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;

  [[nodiscard]] int get() const noexcept { return descriptor_; }

private:
  int descriptor_;
};

std::string system_error_text(std::string_view action) {
  return std::string(action) + ": " + std::system_category().message(errno);
}

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             std::string_view description) {
  if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::runtime_error(std::string(description) + " overflows size_t");
  }
  return left * right;
}

std::size_t checked_add(std::size_t left, std::size_t right,
                        std::string_view description) {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    throw std::runtime_error(std::string(description) + " overflows size_t");
  }
  return left + right;
}

std::size_t shape_count(const std::vector<std::size_t> &shape) {
  if (shape.empty()) {
    return 1U;
  }
  std::size_t count{1U};
  for (const std::size_t dimension : shape) {
    count = checked_multiply(count, dimension, "array element count");
  }
  return count;
}

std::size_t payload_size(const Array &array) {
  const std::size_t count{shape_count(array.shape)};
  if (array.descr == "<f8" || array.descr == "<u8") {
    return checked_multiply(count, 8U, "array payload size");
  }
  if (array.descr == "<c16") {
    return checked_multiply(count, 16U, "array payload size");
  }
  if (array.descr == "|b1") {
    return count;
  }
  if (array.descr.size() > 2U && array.descr.starts_with("<U")) {
    std::size_t length{};
    const std::string_view digits{array.descr.data() + 2,
                                  array.descr.size() - 2U};
    const auto parsed{
        std::from_chars(digits.data(), digits.data() + digits.size(), length)};
    if (parsed.ec != std::errc{} ||
        parsed.ptr != digits.data() + digits.size()) {
      throw std::runtime_error(
          "array has an invalid canonical Unicode descriptor");
    }
    const std::size_t itemsize{
        checked_multiply(length, 4U, "Unicode item size")};
    return checked_multiply(count, itemsize, "Unicode payload size");
  }
  throw std::runtime_error("array does not use a supported canonical dtype");
}

std::uint64_t read_unsigned(std::span<const std::byte> bytes,
                            bool little_endian) {
  std::uint64_t value{};
  if (little_endian) {
    for (std::size_t index{0}; index < bytes.size(); ++index) {
      value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
    }
  } else {
    for (const std::byte byte : bytes) {
      value = (value << 8U) | static_cast<std::uint64_t>(byte);
    }
  }
  return value;
}

void append_little_endian(std::vector<std::byte> &destination,
                          std::uint64_t value, std::size_t width) {
  for (std::size_t index{0}; index < width; ++index) {
    destination.push_back(
        static_cast<std::byte>((value >> (index * 8U)) & 0xffU));
  }
}

bool valid_utf8(std::span<const std::byte> bytes) {
  std::size_t index{0};
  while (index < bytes.size()) {
    const auto first{static_cast<unsigned int>(bytes[index])};
    if (first <= 0x7fU) {
      ++index;
      continue;
    }

    std::size_t width{};
    std::uint32_t codepoint{};
    if (first >= 0xc2U && first <= 0xdfU) {
      width = 2U;
      codepoint = first & 0x1fU;
    } else if (first >= 0xe0U && first <= 0xefU) {
      width = 3U;
      codepoint = first & 0x0fU;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      width = 4U;
      codepoint = first & 0x07U;
    } else {
      return false;
    }
    if (index + width > bytes.size()) {
      return false;
    }
    for (std::size_t offset{1}; offset < width; ++offset) {
      const auto continuation{static_cast<unsigned int>(bytes[index + offset])};
      if ((continuation & 0xc0U) != 0x80U) {
        return false;
      }
      codepoint = (codepoint << 6U) | (continuation & 0x3fU);
    }
    if ((width == 3U && codepoint < 0x800U) ||
        (width == 4U && codepoint < 0x10000U) ||
        (codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
        codepoint > 0x10ffffU) {
      return false;
    }
    index += width;
  }
  return true;
}

std::vector<std::uint32_t> decode_utf8(std::string_view text) {
  const auto bytes{std::as_bytes(std::span{text.data(), text.size()})};
  if (!valid_utf8(bytes)) {
    throw std::runtime_error("UTF-8 input is invalid");
  }

  std::vector<std::uint32_t> codepoints;
  std::size_t index{0};
  while (index < text.size()) {
    const auto first{
        static_cast<unsigned int>(static_cast<unsigned char>(text[index]))};
    if (first <= 0x7fU) {
      codepoints.push_back(static_cast<std::uint32_t>(first));
      ++index;
      continue;
    }
    const std::size_t width{first < 0xe0U ? 2U : (first < 0xf0U ? 3U : 4U)};
    std::uint32_t codepoint{
        first & (width == 2U ? 0x1fU : (width == 3U ? 0x0fU : 0x07U))};
    for (std::size_t offset{1}; offset < width; ++offset) {
      codepoint = (codepoint << 6U) |
                  (static_cast<unsigned int>(
                       static_cast<unsigned char>(text[index + offset])) &
                   0x3fU);
    }
    codepoints.push_back(codepoint);
    index += width;
  }
  return codepoints;
}

void append_utf8(std::string &destination, std::uint32_t codepoint) {
  if (codepoint <= 0x7fU) {
    destination.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ffU) {
    destination.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
    destination.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else if (codepoint <= 0xffffU) {
    if (codepoint >= 0xd800U && codepoint <= 0xdfffU) {
      throw std::runtime_error("Unicode surrogate cannot be encoded as UTF-8");
    }
    destination.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
    destination.push_back(
        static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    destination.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else if (codepoint <= 0x10ffffU) {
    destination.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
    destination.push_back(
        static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
    destination.push_back(
        static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    destination.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else {
    throw std::runtime_error("Unicode code point is out of range");
  }
}

Dtype parse_dtype(std::string_view text) {
  if (text.size() < 2U) {
    throw std::runtime_error("NPY dtype descriptor is too short");
  }
  const char endian{text.front()};
  const std::string_view kind{text.substr(1)};
  if (endian == '=') {
    throw std::runtime_error("NPY native-endian dtype is not portable");
  }

  if (kind == "f8") {
    if (endian != '<' && endian != '>') {
      throw std::runtime_error(
          "NPY float64 dtype requires explicit byte order");
    }
    return {ScalarKind::float64, 8U, 0U, "<f8", endian == '<'};
  }
  if (kind == "c16") {
    if (endian != '<' && endian != '>') {
      throw std::runtime_error(
          "NPY complex128 dtype requires explicit byte order");
    }
    return {ScalarKind::complex128, 16U, 0U, "<c16", endian == '<'};
  }
  if (kind == "u8") {
    if (endian != '<' && endian != '>') {
      throw std::runtime_error("NPY uint64 dtype requires explicit byte order");
    }
    return {ScalarKind::uint64, 8U, 0U, "<u8", endian == '<'};
  }
  if (kind == "b1") {
    if (endian != '|') {
      throw std::runtime_error(
          "NPY boolean dtype must use byte-order marker |");
    }
    return {ScalarKind::boolean, 1U, 0U, "|b1", true};
  }
  if (!kind.empty() && kind.front() == 'U') {
    if (endian != '<' && endian != '>') {
      throw std::runtime_error(
          "NPY Unicode dtype requires explicit byte order");
    }
    const std::string_view digits{kind.substr(1)};
    if (digits.empty()) {
      throw std::runtime_error("NPY Unicode dtype is missing its length");
    }
    std::size_t length{};
    const auto parsed{
        std::from_chars(digits.data(), digits.data() + digits.size(), length)};
    if (parsed.ec != std::errc{} ||
        parsed.ptr != digits.data() + digits.size()) {
      throw std::runtime_error("NPY Unicode dtype length is invalid");
    }
    if (length == 0)
      throw std::runtime_error(
          "zero-width Unicode is outside the metadata schema");
    const std::size_t itemsize{
        checked_multiply(length, 4U, "NPY Unicode item size")};
    return {ScalarKind::unicode, itemsize, length,
            "<U" + std::to_string(length), endian == '<'};
  }
  throw std::runtime_error("NPY dtype is unsupported");
}

class HeaderParser {
public:
  explicit HeaderParser(std::span<const std::byte> header) : header_{header} {}

  struct Result {
    std::string descr;
    std::vector<std::size_t> shape;
  };

  Result parse() {
    skip_space();
    expect('{');
    bool have_descr{false};
    bool have_fortran{false};
    bool have_shape{false};
    Result result;

    skip_space();
    while (peek() != '}') {
      const std::string key{parse_string()};
      skip_space();
      expect(':');
      skip_space();
      if (key == "descr") {
        if (have_descr) {
          throw std::runtime_error("duplicate NPY header key: descr");
        }
        result.descr = parse_string();
        have_descr = true;
      } else if (key == "fortran_order") {
        if (have_fortran) {
          throw std::runtime_error("duplicate NPY header key: fortran_order");
        }
        if (parse_bool()) {
          throw std::runtime_error("Fortran-order NPY arrays are unsupported");
        }
        have_fortran = true;
      } else if (key == "shape") {
        if (have_shape) {
          throw std::runtime_error("duplicate NPY header key: shape");
        }
        result.shape = parse_shape();
        have_shape = true;
      } else {
        throw std::runtime_error("unknown NPY header key: " + key);
      }

      skip_space();
      if (peek() == ',') {
        ++position_;
        skip_space();
        if (peek() == '}') {
          break;
        }
      } else {
        break;
      }
    }
    expect('}');
    skip_space();
    if (position_ != header_.size()) {
      throw std::runtime_error("trailing content in NPY header");
    }
    if (!have_descr || !have_fortran || !have_shape) {
      throw std::runtime_error("NPY header is missing a required key");
    }
    return result;
  }

private:
  void skip_space() {
    while (position_ < header_.size()) {
      const char value{static_cast<char>(header_[position_])};
      if (value != ' ' && value != '\t' && value != '\r' && value != '\n') {
        break;
      }
      ++position_;
    }
  }

  [[nodiscard]] char peek() const {
    return position_ < header_.size() ? static_cast<char>(header_[position_])
                                      : '\0';
  }

  void expect(char expected) {
    if (peek() != expected) {
      throw std::runtime_error(std::string("expected '") + expected +
                               "' in NPY header");
    }
    ++position_;
  }

  std::string parse_string() {
    const char quote{peek()};
    if (quote != '\'' && quote != '"') {
      throw std::runtime_error("expected quoted ASCII string in NPY header");
    }
    ++position_;
    std::string result;
    while (position_ < header_.size()) {
      const unsigned int value{static_cast<unsigned int>(header_[position_])};
      if (value == static_cast<unsigned int>(quote)) {
        ++position_;
        return result;
      }
      if (value < 0x20U || value > 0x7eU || value == '\\') {
        throw std::runtime_error(
            "unsupported character or escape in NPY header string");
      }
      result.push_back(static_cast<char>(value));
      ++position_;
    }
    throw std::runtime_error("unterminated NPY header string");
  }

  bool parse_bool() {
    if (consume("True")) {
      return true;
    }
    if (consume("False")) {
      return false;
    }
    throw std::runtime_error("expected boolean in NPY header");
  }

  bool consume(std::string_view text) {
    if (header_.size() - position_ < text.size()) {
      return false;
    }
    for (std::size_t index{0}; index < text.size(); ++index) {
      if (header_[position_ + index] != static_cast<std::byte>(text[index])) {
        return false;
      }
    }
    position_ += text.size();
    return true;
  }

  std::size_t parse_dimension() {
    const std::size_t start{position_};
    while (position_ < header_.size() && header_[position_] >= std::byte{'0'} &&
           header_[position_] <= std::byte{'9'}) {
      ++position_;
    }
    if (position_ == start) {
      throw std::runtime_error("expected unsigned dimension in NPY header");
    }
    if (position_ - start > 1 && header_[start] == std::byte{'0'})
      throw std::runtime_error("noncanonical leading zero in NPY dimension");
    std::size_t value{};
    const auto parsed{std::from_chars(
        reinterpret_cast<const char *>(header_.data() + start),
        reinterpret_cast<const char *>(header_.data() + position_), value)};
    if (parsed.ec != std::errc{}) {
      throw std::runtime_error("NPY dimension overflows size_t");
    }
    return value;
  }

  std::vector<std::size_t> parse_shape() {
    expect('(');
    skip_space();
    std::vector<std::size_t> shape;
    if (peek() == ')') {
      ++position_;
      return shape;
    }
    while (true) {
      shape.push_back(parse_dimension());
      if (shape.size() > 32)
        throw std::runtime_error("NPY rank exceeds 32");
      skip_space();
      if (peek() == ')') {
        if (shape.size() == 1)
          throw std::runtime_error(
              "singleton NPY shape requires a tuple comma");
        ++position_;
        return shape;
      }
      expect(',');
      skip_space();
      if (peek() == ')') {
        ++position_;
        return shape;
      }
    }
  }

  std::span<const std::byte> header_;
  std::size_t position_{0};
};

std::vector<std::byte> canonical_payload(std::span<const std::byte> payload,
                                         const Dtype &dtype,
                                         std::size_t count) {
  const std::size_t expected{
      checked_multiply(count, dtype.itemsize, "NPY payload size")};
  if (payload.size() != expected) {
    throw std::runtime_error(
        "NPY payload length does not match dtype and shape");
  }

  std::vector<std::byte> canonical;
  canonical.reserve(payload.size());
  if (dtype.kind == ScalarKind::boolean) {
    for (const std::byte value : payload) {
      if (value != std::byte{0} && value != std::byte{1}) {
        throw std::runtime_error("NPY boolean value is not 0 or 1");
      }
    }
    canonical.assign(payload.begin(), payload.end());
    return canonical;
  }

  for (std::size_t item{0}; item < count; ++item) {
    const std::size_t offset{
        checked_multiply(item, dtype.itemsize, "NPY item offset")};
    if (dtype.kind == ScalarKind::float64 || dtype.kind == ScalarKind::uint64) {
      const auto value{read_unsigned(payload.subspan(offset, 8U),
                                     dtype.input_little_endian)};
      append_little_endian(canonical, value, 8U);
    } else if (dtype.kind == ScalarKind::complex128) {
      const auto real{read_unsigned(payload.subspan(offset, 8U),
                                    dtype.input_little_endian)};
      const auto imaginary{read_unsigned(payload.subspan(offset + 8U, 8U),
                                         dtype.input_little_endian)};
      append_little_endian(canonical, real, 8U);
      append_little_endian(canonical, imaginary, 8U);
    } else {
      bool padding_started{false};
      for (std::size_t character{0}; character < dtype.unicode_length;
           ++character) {
        const std::size_t character_offset{
            offset +
            checked_multiply(character, 4U, "Unicode character offset")};
        const auto codepoint{read_unsigned(
            payload.subspan(character_offset, 4U), dtype.input_little_endian)};
        if ((codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
            codepoint > 0x10ffffU) {
          throw std::runtime_error("NPY Unicode code point is invalid");
        }
        if (padding_started && codepoint != 0U) {
          throw std::runtime_error(
              "NPY Unicode string contains an embedded NUL");
        }
        if (codepoint == 0U) {
          padding_started = true;
        }
        append_little_endian(canonical, codepoint, 4U);
      }
    }
  }
  return canonical;
}

std::string shape_text(const std::vector<std::size_t> &shape) {
  if (shape.empty()) {
    return "()";
  }
  std::string text{"("};
  for (std::size_t index{0}; index < shape.size(); ++index) {
    if (index != 0U) {
      text += ", ";
    }
    text += std::to_string(shape[index]);
  }
  text += ",)";
  return text;
}

struct EncodedHeader {
  std::string bytes;
  std::uint8_t version{};
};

EncodedHeader encode_header(const Array &array) {
  const std::string dictionary{
      "{'descr': '" + array.descr +
      "', 'fortran_order': False, 'shape': " + shape_text(array.shape) + ", }"};

  auto padded = [&](std::size_t prefix) {
    std::string header{dictionary};
    header.push_back('\n');
    while ((prefix + header.size()) % npy_alignment != 0U) {
      header.insert(header.end() - 1, ' ');
    }
    return header;
  };

  std::string header{padded(npy_v1_prefix)};
  std::uint8_t version{1};
  if (header.size() > npy_v1_max_header) {
    version = 2;
    header = padded(npy_v2_prefix);
    if (header.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("NPY header exceeds the v2 length field");
    }
  }
  return {std::move(header), version};
}

bool safe_key(std::string_view key) {
  if (key.empty() || key == "." || key == "..") {
    return false;
  }
  const auto bytes{std::as_bytes(std::span{key.data(), key.size()})};
  if (!valid_utf8(bytes)) {
    return false;
  }
  for (const unsigned char value : key) {
    if (value == '/' || value == '\\' || value == '\0' || value < 0x20U) {
      return false;
    }
  }
  return true;
}

std::vector<std::byte> read_input_file(const std::filesystem::path &path,
                                       const Limits &limits) {
  FileDescriptor file{::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
  if (file.get() < 0) {
    throw std::runtime_error(system_error_text("cannot open NPZ input"));
  }
  struct stat metadata{};
  if (::fstat(file.get(), &metadata) != 0) {
    throw std::runtime_error(system_error_text("cannot stat NPZ input"));
  }
  if (!S_ISREG(metadata.st_mode)) {
    throw std::runtime_error("NPZ input is not a regular file");
  }
  if (metadata.st_size < 0 ||
      static_cast<std::uintmax_t>(metadata.st_size) > limits.max_input_bytes ||
      static_cast<std::uintmax_t>(metadata.st_size) >
          std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("NPZ input exceeds the configured byte limit");
  }

  const auto size{static_cast<std::size_t>(metadata.st_size)};
  std::vector<std::byte> bytes(size);
  std::size_t position{0};
  while (position < size) {
    const ssize_t count{
        ::read(file.get(), bytes.data() + position, size - position)};
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(system_error_text("NPZ input read failed"));
    }
    if (count == 0) {
      throw std::runtime_error("NPZ input changed while being read");
    }
    position += static_cast<std::size_t>(count);
  }
  std::byte extra{};
  while (true) {
    const ssize_t extra_count{::read(file.get(), &extra, 1U)};
    if (extra_count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(system_error_text("NPZ input read failed"));
    }
    if (extra_count != 0) {
      throw std::runtime_error("NPZ input changed while being read");
    }
    break;
  }
  return bytes;
}

class ZipReader {
public:
  ZipReader() { mz_zip_zero_struct(&archive_); }

  ~ZipReader() {
    if (active_) {
      static_cast<void>(mz_zip_reader_end(&archive_));
    }
  }

  ZipReader(const ZipReader &) = delete;
  ZipReader &operator=(const ZipReader &) = delete;

  void initialize(std::span<const std::byte> bytes) {
    if (!mz_zip_reader_init_mem(&archive_, bytes.data(), bytes.size(), 0U)) {
      throw std::runtime_error(
          std::string("cannot initialize ZIP reader: ") +
          mz_zip_get_error_string(mz_zip_get_last_error(&archive_)));
    }
    active_ = true;
  }

  [[nodiscard]] mz_zip_archive &archive() noexcept { return archive_; }

  void finish() {
    if (active_ && !mz_zip_reader_end(&archive_)) {
      active_ = false;
      throw std::runtime_error("ZIP reader finalization failed");
    }
    active_ = false;
  }

private:
  mz_zip_archive archive_{};
  bool active_{false};
};

std::string zip_entry_name(mz_zip_archive &archive, mz_uint index,
                           const Limits &limits) {
  const mz_uint needed{
      mz_zip_reader_get_filename(&archive, index, nullptr, 0U)};
  constexpr std::size_t max_zip_name_with_nul{
      static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max()) + 1U};
  if (needed == 0U || needed > std::numeric_limits<std::size_t>::max() ||
      static_cast<std::size_t>(needed) > max_zip_name_with_nul ||
      static_cast<std::size_t>(needed) > limits.max_header_bytes) {
    throw std::runtime_error("ZIP entry name is invalid or too long");
  }
  std::vector<char> buffer(needed);
  const mz_uint written{
      mz_zip_reader_get_filename(&archive, index, buffer.data(), needed)};
  if (written != needed || buffer.size() != needed || buffer.back() != '\0') {
    throw std::runtime_error("cannot read complete ZIP entry name");
  }
  if (std::find(buffer.begin(), buffer.end() - 1, '\0') != buffer.end() - 1) {
    throw std::runtime_error("ZIP entry name contains NUL");
  }
  return std::string(buffer.begin(), buffer.end() - 1);
}

bool arrays_equal(const Array &left, const Array &right) {
  return left.descr == right.descr && left.shape == right.shape &&
         left.data == right.data;
}

std::filesystem::path temporary_path(const std::filesystem::path &destination) {
  static std::atomic<std::uint64_t> sequence{0};
  const auto parent{destination.parent_path().empty()
                        ? std::filesystem::path{"."}
                        : destination.parent_path()};
  for (unsigned int attempt{0}; attempt < 100U; ++attempt) {
    const auto serial = sequence.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::filesystem::path candidate{
        parent / (destination.filename().string() + ".tmp." +
                  std::to_string(::getpid()) + "." + std::to_string(serial))};
    FileDescriptor file{
        ::open(candidate.c_str(),
               O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600)};
    if (file.get() >= 0) {
      return candidate;
    }
    if (errno != EEXIST) {
      throw std::runtime_error(
          system_error_text("cannot create NPZ temporary file"));
    }
  }
  throw std::runtime_error("cannot allocate a unique NPZ temporary file");
}

void fsync_file(const std::filesystem::path &path) {
  FileDescriptor file{::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW)};
  if (file.get() < 0) {
    throw std::runtime_error(system_error_text("cannot reopen completed NPZ"));
  }
  if (::fsync(file.get()) != 0) {
    throw std::runtime_error(system_error_text("cannot fsync completed NPZ"));
  }
}

void write_archive(const std::filesystem::path &path,
                   const std::map<std::string, Array> &arrays) {
  mz_zip_archive archive{};
  if (!mz_zip_writer_init_file(&archive, path.c_str(), 0U)) {
    throw std::runtime_error("cannot initialize NPZ writer");
  }
  bool active{true};
  try {
    for (const auto &[key, array] : arrays) {
      const std::vector<std::byte> encoded{encode_npy(array)};
      const std::string entry_name{key + ".npy"};
      if (!mz_zip_writer_add_mem(&archive, entry_name.c_str(), encoded.data(),
                                 encoded.size(), MZ_NO_COMPRESSION)) {
        throw std::runtime_error("cannot add NPZ entry: " + key);
      }
    }
    if (!mz_zip_writer_finalize_archive(&archive)) {
      throw std::runtime_error("cannot finalize NPZ archive");
    }
    if (!mz_zip_writer_end(&archive)) {
      active = false;
      throw std::runtime_error("cannot close NPZ archive");
    }
    active = false;
  } catch (...) {
    if (active) {
      static_cast<void>(mz_zip_writer_end(&archive));
    }
    throw;
  }
}

} // namespace

Array from_doubles(std::span<const double> values,
                   std::vector<std::size_t> shape) {
  if (shape_count(shape) != values.size()) {
    throw std::runtime_error("double values do not match the requested shape");
  }
  Array result{"<f8", std::move(shape), {}};
  result.data.reserve(
      checked_multiply(values.size(), 8U, "double payload size"));
  for (const double value : values) {
    append_little_endian(result.data, std::bit_cast<std::uint64_t>(value), 8U);
  }
  return result;
}

Array from_complex(std::span<const std::complex<double>> values,
                   std::vector<std::size_t> shape) {
  if (shape_count(shape) != values.size()) {
    throw std::runtime_error("complex values do not match the requested shape");
  }
  Array result{"<c16", std::move(shape), {}};
  result.data.reserve(
      checked_multiply(values.size(), 16U, "complex payload size"));
  for (const std::complex<double> value : values) {
    append_little_endian(result.data,
                         std::bit_cast<std::uint64_t>(value.real()), 8U);
    append_little_endian(result.data,
                         std::bit_cast<std::uint64_t>(value.imag()), 8U);
  }
  return result;
}

Array from_uint64(std::uint64_t value) {
  Array result{"<u8", {}, {}};
  append_little_endian(result.data, value, 8U);
  return result;
}

Array from_double(double value) {
  const std::array<double, 1> values{value};
  return from_doubles(values);
}

Array from_bool(bool value) {
  return {"|b1", {}, {static_cast<std::byte>(value)}};
}

Array from_utf8(std::string_view value) {
  const std::vector<std::uint32_t> codepoints{decode_utf8(value)};
  if (std::find(codepoints.begin(), codepoints.end(), 0U) != codepoints.end()) {
    throw std::runtime_error("UTF-8 metadata string contains an embedded NUL");
  }
  const std::size_t length{std::max<std::size_t>(1U, codepoints.size())};
  Array result{"<U" + std::to_string(length), {}, {}};
  result.data.reserve(checked_multiply(length, 4U, "Unicode payload size"));
  for (const std::uint32_t codepoint : codepoints) {
    append_little_endian(result.data, codepoint, 4U);
  }
  while (result.data.size() < length * 4U) {
    result.data.push_back(std::byte{0});
  }
  return result;
}

std::vector<double> to_doubles(const Array &array) {
  if (array.descr != "<f8") {
    throw std::runtime_error("array is not canonical float64");
  }
  const std::size_t count{shape_count(array.shape)};
  if (array.data.size() != checked_multiply(count, 8U, "double payload size")) {
    throw std::runtime_error("float64 payload length mismatch");
  }
  std::vector<double> values;
  values.reserve(count);
  for (std::size_t index{0}; index < count; ++index) {
    const auto bits{
        read_unsigned(std::span{array.data}.subspan(index * 8U, 8U), true)};
    values.push_back(std::bit_cast<double>(bits));
  }
  return values;
}

std::vector<std::complex<double>> to_complex(const Array &array) {
  if (array.descr != "<c16") {
    throw std::runtime_error("array is not canonical complex128");
  }
  const std::size_t count{shape_count(array.shape)};
  if (array.data.size() !=
      checked_multiply(count, 16U, "complex payload size")) {
    throw std::runtime_error("complex128 payload length mismatch");
  }
  std::vector<std::complex<double>> values;
  values.reserve(count);
  for (std::size_t index{0}; index < count; ++index) {
    const auto real{std::bit_cast<double>(
        read_unsigned(std::span{array.data}.subspan(index * 16U, 8U), true))};
    const auto imaginary{std::bit_cast<double>(read_unsigned(
        std::span{array.data}.subspan(index * 16U + 8U, 8U), true))};
    values.emplace_back(real, imaginary);
  }
  return values;
}

std::uint64_t to_uint64(const Array &array) {
  if (array.descr != "<u8" || !array.shape.empty() || array.data.size() != 8U) {
    throw std::runtime_error("array is not a canonical uint64 scalar");
  }
  return read_unsigned(array.data, true);
}

double to_double(const Array &array) {
  if (array.descr != "<f8" || !array.shape.empty() || array.data.size() != 8U) {
    throw std::runtime_error("array is not a canonical float64 scalar");
  }
  return std::bit_cast<double>(read_unsigned(array.data, true));
}

bool to_bool(const Array &array) {
  if (array.descr != "|b1" || !array.shape.empty() || array.data.size() != 1U ||
      (array.data.front() != std::byte{0} &&
       array.data.front() != std::byte{1})) {
    throw std::runtime_error("array is not a canonical boolean scalar");
  }
  return array.data.front() == std::byte{1};
}

std::string to_utf8(const Array &array) {
  if (!array.descr.starts_with("<U") || !array.shape.empty()) {
    throw std::runtime_error("array is not a canonical Unicode scalar");
  }
  std::size_t length{};
  const std::string_view digits{array.descr.data() + 2,
                                array.descr.size() - 2U};
  const auto parsed{
      std::from_chars(digits.data(), digits.data() + digits.size(), length)};
  if (parsed.ec != std::errc{} || parsed.ptr != digits.data() + digits.size() ||
      array.data.size() !=
          checked_multiply(length, 4U, "Unicode payload size")) {
    throw std::runtime_error("Unicode scalar payload length mismatch");
  }

  std::string result;
  bool padding_started{false};
  for (std::size_t index{0}; index < length; ++index) {
    const auto codepoint{
        read_unsigned(std::span{array.data}.subspan(index * 4U, 4U), true)};
    if ((codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
        codepoint > 0x10ffffU) {
      throw std::runtime_error("Unicode scalar contains an invalid code point");
    }
    if (padding_started && codepoint != 0U) {
      throw std::runtime_error("Unicode scalar contains an embedded NUL");
    }
    if (codepoint == 0U) {
      padding_started = true;
      continue;
    }
    append_utf8(result, codepoint);
  }
  return result;
}

Array parse_npy(std::span<const std::byte> bytes, const Limits &limits) {
  if (bytes.size() > limits.max_input_bytes)
    throw std::runtime_error("NPY input exceeds the configured byte limit");
  if (bytes.size() < npy_v1_prefix) {
    throw std::runtime_error("NPY input is shorter than its magic and prefix");
  }
  for (std::size_t index{0}; index < npy_magic.size(); ++index) {
    if (static_cast<unsigned char>(bytes[index]) != npy_magic[index]) {
      throw std::runtime_error("NPY magic is invalid");
    }
  }
  const auto major{static_cast<unsigned int>(bytes[6])};
  const auto minor{static_cast<unsigned int>(bytes[7])};
  if (minor != 0U || (major != 1U && major != 2U && major != 3U)) {
    throw std::runtime_error("NPY version is unsupported");
  }
  const std::size_t prefix{major == 1U ? npy_v1_prefix : npy_v2_prefix};
  if (bytes.size() < prefix) {
    throw std::runtime_error("NPY input is shorter than its header prefix");
  }

  const std::uint64_t raw_header_length{
      major == 1U ? read_unsigned(bytes.subspan(8U, 2U), true)
                  : read_unsigned(bytes.subspan(8U, 4U), true)};
  if (raw_header_length > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("NPY header length exceeds this platform");
  }
  const auto header_length{static_cast<std::size_t>(raw_header_length)};
  if (header_length > limits.max_header_bytes) {
    throw std::runtime_error("NPY header exceeds the configured byte limit");
  }
  if (header_length > bytes.size() - prefix) {
    throw std::runtime_error("NPY header is truncated");
  }
  const auto header{bytes.subspan(prefix, header_length)};
  if (header.empty() || header.back() != std::byte{'\n'}) {
    throw std::runtime_error("NPY header is not newline terminated");
  }
  if (major == 3U && !valid_utf8(header)) {
    throw std::runtime_error("NPY v3 header is not valid UTF-8");
  }
  if (major != 3U) {
    for (const std::byte value : header) {
      if (static_cast<unsigned char>(value) > 0x7fU) {
        throw std::runtime_error("NPY v1/v2 header is not ASCII");
      }
    }
  }

  HeaderParser parser{header};
  HeaderParser::Result parsed{parser.parse()};
  const Dtype dtype{parse_dtype(parsed.descr)};
  const std::size_t count{shape_count(parsed.shape)};
  const auto payload{bytes.subspan(prefix + header_length)};
  if (payload.size() > limits.max_expanded_bytes)
    throw std::runtime_error("NPY payload exceeds the configured byte limit");
  Array array{dtype.canonical_descr, std::move(parsed.shape),
              canonical_payload(payload, dtype, count)};
  return array;
}

std::vector<std::byte> encode_npy(const Array &array) {
  const Dtype dtype{parse_dtype(array.descr)};
  if (dtype.canonical_descr != array.descr) {
    throw std::runtime_error(
        "encode_npy requires canonical little-endian dtype");
  }
  const std::size_t expected_size{payload_size(array)};
  if (array.data.size() != expected_size) {
    throw std::runtime_error(
        "array payload length does not match dtype and shape");
  }
  if (dtype.kind == ScalarKind::boolean) {
    for (const std::byte value : array.data) {
      if (value != std::byte{0} && value != std::byte{1}) {
        throw std::runtime_error("boolean array value is not 0 or 1");
      }
    }
  }
  if (dtype.kind == ScalarKind::unicode) {
    const std::size_t count{shape_count(array.shape)};
    for (std::size_t item{0}; item < count; ++item) {
      bool padding_started{false};
      for (std::size_t character{0}; character < dtype.unicode_length;
           ++character) {
        const std::size_t offset{item * dtype.itemsize + character * 4U};
        const auto codepoint{
            read_unsigned(std::span{array.data}.subspan(offset, 4U), true)};
        if ((codepoint >= 0xd800U && codepoint <= 0xdfffU) ||
            codepoint > 0x10ffffU) {
          throw std::runtime_error(
              "Unicode array contains an invalid code point");
        }
        if (padding_started && codepoint != 0U) {
          throw std::runtime_error("Unicode array contains an embedded NUL");
        }
        padding_started = codepoint == 0U;
      }
    }
  }

  const auto [header, version] = encode_header(array);
  std::vector<std::byte> result;
  result.reserve(
      checked_add(checked_add(npy_v1_prefix, header.size(), "NPY encoded size"),
                  array.data.size(), "NPY encoded size"));
  result.push_back(std::byte{0x93});
  for (const unsigned char value : std::string_view{"NUMPY"}) {
    result.push_back(static_cast<std::byte>(value));
  }
  result.push_back(std::byte{version});
  result.push_back(std::byte{0});
  if (version == 1U) {
    append_little_endian(result, header.size(), 2U);
  } else {
    append_little_endian(result, header.size(), 4U);
  }
  const auto header_bytes{
      std::as_bytes(std::span{header.data(), header.size()})};
  result.insert(result.end(), header_bytes.begin(), header_bytes.end());
  result.insert(result.end(), array.data.begin(), array.data.end());
  return result;
}

std::map<std::string, Array> read_npz(const std::filesystem::path &path,
                                      const Limits &limits,
                                      std::optional<Keys> allowed_keys) {
  const std::vector<std::byte> input{read_input_file(path, limits)};
  ZipReader reader;
  reader.initialize(input);
  mz_zip_archive &archive{reader.archive()};
  const mz_uint entry_count{mz_zip_reader_get_num_files(&archive)};
  if (entry_count > limits.max_entries) {
    throw std::runtime_error("NPZ entry count exceeds the configured limit");
  }

  std::map<std::string, Array> arrays;
  std::set<std::string> seen;
  std::size_t expanded_bytes{0};
  for (mz_uint index{0}; index < entry_count; ++index) {
    mz_zip_archive_file_stat stat{};
    if (!mz_zip_reader_file_stat(&archive, index, &stat)) {
      throw std::runtime_error("cannot read NPZ entry metadata");
    }
    if (stat.m_is_directory) {
      throw std::runtime_error("directory entries are not allowed in NPZ");
    }
    if (stat.m_is_encrypted ||
        mz_zip_reader_is_file_encrypted(&archive, index)) {
      throw std::runtime_error("encrypted entries are not allowed in NPZ");
    }
    if (stat.m_method != MZ_NO_COMPRESSION && stat.m_method != MZ_DEFLATED) {
      throw std::runtime_error("unsupported ZIP compression method");
    }
    if (!stat.m_is_supported ||
        !mz_zip_reader_is_file_supported(&archive, index)) {
      throw std::runtime_error("unsupported ZIP entry");
    }
    if (stat.m_uncomp_size > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("NPZ entry is too large for this platform");
    }
    const auto entry_size{static_cast<std::size_t>(stat.m_uncomp_size)};
    expanded_bytes =
        checked_add(expanded_bytes, entry_size, "NPZ expanded byte count");
    if (expanded_bytes > limits.max_expanded_bytes) {
      throw std::runtime_error(
          "NPZ expanded data exceeds the configured byte limit");
    }

    const std::string name{zip_entry_name(archive, index, limits)};
    if (!name.ends_with(".npy") || name.size() == 4U) {
      throw std::runtime_error("NPZ entry does not end in .npy");
    }
    const std::string key{name.substr(0, name.size() - 4U)};
    if (!safe_key(key)) {
      throw std::runtime_error("NPZ entry has an unsafe key or path");
    }
    if (!seen.insert(name).second) {
      throw std::runtime_error("duplicate NPZ entry name");
    }
    if (!arrays.emplace(key, Array{}).second) {
      throw std::runtime_error("duplicate NPZ key");
    }
    if (allowed_keys.has_value() && !allowed_keys->contains(key)) {
      throw std::runtime_error("NPZ entry is not present in the key whitelist");
    }

    std::vector<std::byte> encoded(entry_size == 0U ? 1U : entry_size);
    if (!mz_zip_reader_extract_to_mem(&archive, index, encoded.data(),
                                      entry_size, 0U)) {
      throw std::runtime_error(
          std::string("NPZ entry extraction or CRC check failed: ") +
          mz_zip_get_error_string(mz_zip_get_last_error(&archive)));
    }
    Limits entry_limits = limits;
    entry_limits.max_input_bytes = limits.max_expanded_bytes;
    arrays.at(key) =
        parse_npy(std::span{encoded.data(), entry_size}, entry_limits);
  }
  reader.finish();
  return arrays;
}

void write_npz_temporary(const std::filesystem::path &path,
                         const std::map<std::string, Array> &arrays) {
  const auto status = std::filesystem::symlink_status(path);
  if (!std::filesystem::is_regular_file(status) ||
      std::filesystem::file_size(path) != 0)
    throw std::runtime_error(
        "NPZ temp writer requires a caller-owned empty regular file");
  for (const auto &[key, array] : arrays) {
    if (!safe_key(key))
      throw std::runtime_error("invalid NPZ key");
    (void)encode_npy(array);
  }
  write_archive(path, arrays);
  fsync_file(path);

  std::size_t encoded_total{0};
  for (const auto &[key, array] : arrays) {
    static_cast<void>(key);
    encoded_total = checked_add(encoded_total, encode_npy(array).size(),
                                "NPZ verification byte count");
  }
  Limits verification_limits{};
  verification_limits.max_input_bytes =
      static_cast<std::size_t>(std::filesystem::file_size(path));
  verification_limits.max_expanded_bytes = encoded_total;
  verification_limits.max_entries = arrays.size();
  verification_limits.max_header_bytes =
      std::numeric_limits<std::uint32_t>::max();
  const auto verified{read_npz(path, verification_limits)};
  if (verified.size() != arrays.size()) {
    throw std::runtime_error("NPZ readback key count mismatch");
  }
  for (const auto &[key, array] : arrays) {
    const auto found{verified.find(key)};
    if (found == verified.end() || !arrays_equal(found->second, array)) {
      throw std::runtime_error("NPZ readback content mismatch for key: " + key);
    }
  }
}

void write_npz(const std::filesystem::path &path,
               const std::map<std::string, Array> &arrays) {
  if (path.empty())
    throw std::runtime_error("NPZ output path is empty");
  if (std::filesystem::symlink_status(path).type() !=
      std::filesystem::file_type::not_found)
    throw std::runtime_error("refusing to overwrite an existing NPZ file");
  const auto temporary = temporary_path(path);
  bool published = false;
  std::string phase = "writing";
  try {
    write_npz_temporary(temporary, arrays);
    phase = "publishing";
    if (::link(temporary.c_str(), path.c_str()) != 0)
      throw std::runtime_error(system_error_text("cannot publish NPZ output"));
    published = true;
    phase = "published_not_confirmed_durable";
    if (::unlink(temporary.c_str()) != 0)
      throw std::runtime_error(
          system_error_text("cannot unlink published NPZ temp"));
    const auto parent = path.parent_path().empty() ? std::filesystem::path{"."}
                                                   : path.parent_path();
    FileDescriptor directory{::open(
        parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW)};
    if (directory.get() < 0 || ::fsync(directory.get()) != 0)
      throw std::runtime_error(
          system_error_text("cannot fsync NPZ parent directory"));
  } catch (const std::exception &e) {
    // Preserve unpublished bytes and the first failure; never claim rollback
    // after publication.
    throw NpzWriteError(phase, temporary, published, e.what());
  }
}

} // namespace fomhd::io
