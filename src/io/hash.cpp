#include "fomhd/hash.hpp"

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <openssl/evp.h>

namespace fomhd::io {
namespace {

constexpr std::size_t digest_size{32};
constexpr std::size_t read_buffer_size{64U * 1024U};

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

using DigestContext = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>;

DigestContext make_context() {
  DigestContext context{EVP_MD_CTX_new(), &EVP_MD_CTX_free};
  if (!context) {
    throw std::runtime_error("SHA-256 context allocation failed");
  }
  if (EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1) {
    throw std::runtime_error("SHA-256 initialization failed");
  }
  return context;
}

void update(DigestContext &context, std::span<const std::byte> bytes) {
  if (!bytes.empty() &&
      EVP_DigestUpdate(context.get(), bytes.data(), bytes.size()) != 1) {
    throw std::runtime_error("SHA-256 update failed");
  }
}

std::string finish(DigestContext &context) {
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length{};
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1) {
    throw std::runtime_error("SHA-256 finalization failed");
  }
  if (length != digest_size) {
    throw std::runtime_error("SHA-256 produced an unexpected digest size");
  }

  constexpr char hexadecimal[]{"0123456789abcdef"};
  std::string result(digest_size * 2U, '0');
  for (std::size_t index{0}; index < digest_size; ++index) {
    result[index * 2U] = hexadecimal[digest[index] >> 4U];
    result[index * 2U + 1U] = hexadecimal[digest[index] & 0x0fU];
  }
  return result;
}

std::string system_error_text(std::string_view action) {
  return std::string(action) + ": " + std::system_category().message(errno);
}

} // namespace

std::string sha256(std::span<const std::byte> bytes) {
  DigestContext context{make_context()};
  update(context, bytes);
  return finish(context);
}

std::string sha256(std::string_view text) {
  return sha256(std::as_bytes(std::span{text.data(), text.size()}));
}

namespace {
std::string hash_open_file(int descriptor) {
  FileDescriptor file{descriptor};
  if (file.get() < 0) {
    throw std::runtime_error(system_error_text("cannot open file for SHA-256"));
  }

  struct stat metadata{};
  if (::fstat(file.get(), &metadata) != 0) {
    throw std::runtime_error(system_error_text("cannot stat file for SHA-256"));
  }
  if (!S_ISREG(metadata.st_mode)) {
    throw std::runtime_error("SHA-256 input is not a regular file");
  }

  DigestContext context{make_context()};
  std::array<std::byte, read_buffer_size> buffer{};
  while (true) {
    const ssize_t count{::read(file.get(), buffer.data(), buffer.size())};
    if (count < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error(system_error_text("SHA-256 file read failed"));
    }
    if (count == 0) {
      break;
    }
    update(context, std::span{buffer.data(), static_cast<std::size_t>(count)});
  }
  return finish(context);
}
} // namespace

std::string sha256_file(const std::filesystem::path &path) {
  return hash_open_file(
      ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
}

std::string sha256_running_executable() {
  // This kernel-owned magic link identifies the loaded image even if renamed.
  return hash_open_file(::open("/proc/self/exe", O_RDONLY | O_CLOEXEC));
}

} // namespace fomhd::io
