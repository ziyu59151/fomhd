#pragma once

#include <cstddef>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>

namespace fomhd::io {

[[nodiscard]] std::string sha256(std::span<const std::byte> bytes);
[[nodiscard]] std::string sha256(std::string_view text);
[[nodiscard]] std::string sha256_file(const std::filesystem::path &path);
[[nodiscard]] std::string sha256_running_executable();

} // namespace fomhd::io
