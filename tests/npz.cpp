#include "fomhd/hash.hpp"
#include "fomhd/npz.hpp"

#include "miniz.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

using fomhd::io::Array;
using fomhd::io::Limits;
using fomhd::io::Keys;
using Bytes = std::vector<std::byte>;

class TestFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Context {
public:
    void require(bool condition, std::string_view message) const {
        if (!condition) {
            throw TestFailure(std::string(message));
        }
    }

    template <typename Left, typename Right>
    void equal(const Left& left, const Right& right, std::string_view message) const {
        if (!(left == right)) {
            throw TestFailure(std::string(message));
        }
    }

    template <typename Callable>
    void throws(Callable&& callable, std::string_view message) const {
        try {
            std::forward<Callable>(callable)();
        } catch (const std::exception&) {
            return;
        }
        throw TestFailure(std::string(message));
    }
};

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        std::array<char, 64> pattern{};
        const std::string base{"/tmp/fomhd-npz-test-XXXXXX"};
        std::copy(base.begin(), base.end(), pattern.begin());
        char* created{mkdtemp(pattern.data())};
        if (created == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        path_ = created;
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    std::filesystem::path path_;
};

void append_bytes(Bytes& destination, std::span<const std::byte> source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

void append_byte(Bytes& destination, unsigned int value) {
    destination.push_back(static_cast<std::byte>(value & 0xffU));
}

void append_le16(Bytes& destination, std::uint16_t value) {
    append_byte(destination, value);
    append_byte(destination, value >> 8U);
}

void append_le32(Bytes& destination, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        append_byte(destination, value >> shift);
    }
}

template <typename Value>
Bytes value_bytes(Value value) {
    std::array<std::byte, sizeof(Value)> bytes{};
    std::memcpy(bytes.data(), &value, sizeof(Value));
    return {bytes.begin(), bytes.end()};
}

Bytes uint32_bytes(std::uint32_t value) {
    Bytes bytes;
    append_le32(bytes, value);
    return bytes;
}

Bytes make_npy(std::string_view dictionary, std::span<const std::byte> payload,
               std::uint8_t major = 1) {
    Bytes result;
    append_byte(result, 0x93);
    append_bytes(result, std::as_bytes(std::span{"NUMPY", 5}));
    append_byte(result, major);
    append_byte(result, 0);

    std::string header{dictionary};
    const std::size_t prefix_size{major == 1 ? 10U : 12U};
    header.push_back('\n');
    while ((prefix_size + header.size()) % 64U != 0U) {
        header.insert(header.end() - 1, ' ');
    }

    if (major == 1) {
        if (header.size() > 0xffffU) {
            throw std::runtime_error("test header is too large for NPY v1");
        }
        append_le16(result, static_cast<std::uint16_t>(header.size()));
    } else {
        append_le32(result, static_cast<std::uint32_t>(header.size()));
    }
    append_bytes(result, std::as_bytes(std::span{header.data(), header.size()}));
    append_bytes(result, payload);
    return result;
}

std::string npy_dictionary(std::string_view descr, std::string_view shape,
                           bool fortran = false) {
    const std::string shape_tuple{
        shape.empty() ? "()" : "(" + std::string(shape) + ",)"};
    return "{'descr': '" + std::string(descr) + "', 'fortran_order': " +
           (fortran ? "True" : "False") + ", 'shape': " + shape_tuple + ", }";
}

void write_bytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("cannot open test output");
    }
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    if (!output) {
        throw std::runtime_error("cannot write test output");
    }
}

Bytes read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open test input");
    }
    Bytes bytes;
    std::array<char, 4096> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count{static_cast<std::size_t>(input.gcount())};
        for (std::size_t index{0}; index < count; ++index) {
            bytes.push_back(static_cast<std::byte>(
                static_cast<unsigned char>(buffer[index])));
        }
    }
    return bytes;
}

void write_text(const std::filesystem::path& path, std::string_view text) {
    write_bytes(path, std::as_bytes(std::span{text.data(), text.size()}));
}

struct ZipEntry {
    std::string name;
    Bytes data;
};

void make_zip(const std::filesystem::path& path, std::span<const ZipEntry> entries) {
    mz_zip_archive archive{};
    if (!mz_zip_writer_init_file(&archive, path.c_str(), 0)) {
        throw std::runtime_error("cannot initialize test zip");
    }
    try {
        for (const ZipEntry& entry : entries) {
            const std::byte empty_data{};
            const void* data{entry.data.empty()
                                 ? static_cast<const void*>(&empty_data)
                                 : static_cast<const void*>(entry.data.data())};
            if (!mz_zip_writer_add_mem(&archive, entry.name.c_str(), data,
                                       entry.data.size(), MZ_NO_COMPRESSION)) {
                throw std::runtime_error("cannot add test zip entry");
            }
        }
        if (!mz_zip_writer_finalize_archive(&archive)) {
            throw std::runtime_error("cannot finalize test zip");
        }
    } catch (...) {
        static_cast<void>(mz_zip_writer_end(&archive));
        throw;
    }
    if (!mz_zip_writer_end(&archive)) {
        throw std::runtime_error("cannot close test zip");
    }
}

void set_zip_field(const std::filesystem::path& path, std::uint32_t signature,
                   std::size_t field_offset, std::uint16_t value) {
    Bytes bytes{read_bytes(path)};
    Bytes signature_bytes{uint32_bytes(signature)};
    bool found{false};
    for (std::size_t index{0}; index + field_offset + 2U <= bytes.size(); ++index) {
        if (std::equal(signature_bytes.begin(), signature_bytes.end(),
                       bytes.begin() + static_cast<std::ptrdiff_t>(index))) {
            found = true;
            bytes[index + field_offset] =
                static_cast<std::byte>(static_cast<unsigned int>(value) & 0xffU);
            bytes[index + field_offset + 1U] =
                static_cast<std::byte>(static_cast<unsigned int>(value) >> 8U);
        }
    }
    if (!found) {
        throw std::runtime_error("test zip signature not found");
    }
    write_bytes(path, bytes);
}

void set_zip_flag(const std::filesystem::path& path, std::uint16_t flag) {
    Bytes bytes{read_bytes(path)};
    const Bytes local_signature{uint32_bytes(0x04034b50U)};
    const Bytes central_signature{uint32_bytes(0x02014b50U)};
    bool found{false};
    for (std::size_t index{0}; index + 10U <= bytes.size(); ++index) {
        std::size_t offset{};
        if (std::equal(local_signature.begin(), local_signature.end(),
                       bytes.begin() + static_cast<std::ptrdiff_t>(index))) {
            offset = 6U;
        } else if (std::equal(central_signature.begin(), central_signature.end(),
                              bytes.begin() + static_cast<std::ptrdiff_t>(index))) {
            offset = 8U;
        } else {
            continue;
        }
        found = true;
        const auto current{static_cast<std::uint16_t>(bytes[index + offset]) |
                           static_cast<std::uint16_t>(bytes[index + offset + 1U]) << 8U};
        const std::uint16_t updated{static_cast<std::uint16_t>(current | flag)};
        bytes[index + offset] = static_cast<std::byte>(updated & 0xffU);
        bytes[index + offset + 1U] = static_cast<std::byte>(updated >> 8U);
    }
    if (!found) {
        throw std::runtime_error("test zip headers not found");
    }
    write_bytes(path, bytes);
}

void corrupt_payload(const std::filesystem::path& path,
                     std::span<const std::byte> payload) {
    Bytes bytes{read_bytes(path)};
    const auto found{std::search(bytes.begin(), bytes.end(), payload.begin(), payload.end())};
    if (found == bytes.end()) {
        throw std::runtime_error("test payload not found");
    }
    const auto offset{static_cast<std::size_t>(std::distance(bytes.begin(), found))};
    bytes[offset] ^= std::byte{0x01};
    write_bytes(path, bytes);
}

Array double_array() {
    const std::array<double, 4> values{1.25, -2.5, 0.0, 8.75};
    return fomhd::io::from_doubles(values, {2, 2});
}

Array complex_array() {
    const std::array<std::complex<double>, 3> values{
        std::complex<double>{-1.5, -2.25},
        std::complex<double>{-0.0, 4.0},
        std::complex<double>{9.0, -0.0},
    };
    return fomhd::io::from_complex(values, {3});
}

void test_helpers(const Context& context) {
    const Array doubles{double_array()};
    context.equal(doubles.descr, std::string{"<f8"}, "double dtype mismatch");
    context.equal(doubles.shape, std::vector<std::size_t>{2, 2}, "double shape mismatch");
    const std::vector<double> restored{fomhd::io::to_doubles(doubles)};
    context.equal(restored.size(), std::size_t{4}, "double value count mismatch");
    context.equal(restored[1], -2.5, "double value mismatch");

    const Array complex_values{complex_array()};
    const std::vector<std::complex<double>> complex_restored{
        fomhd::io::to_complex(complex_values)};
    context.equal(complex_restored[0], std::complex<double>{-1.5, -2.25},
                  "complex value mismatch");
    context.require(std::signbit(complex_restored[1].real()),
                    "negative zero real component was not preserved");

    context.equal(fomhd::io::to_uint64(fomhd::io::from_uint64(UINT64_MAX)), UINT64_MAX,
                  "uint64 scalar mismatch");
    context.equal(fomhd::io::to_double(fomhd::io::from_double(-3.25)), -3.25,
                  "double scalar mismatch");
    context.equal(fomhd::io::to_bool(fomhd::io::from_bool(true)), true,
                  "boolean scalar mismatch");

    const std::string text{"A0 \xE4\xB8\xAD\xE6\x96\x87"};
    const Array unicode{fomhd::io::from_utf8(text)};
    context.equal(unicode.descr, std::string{"<U5"}, "UTF-8 codepoint count mismatch");
    context.equal(fomhd::io::to_utf8(unicode), text, "Unicode helper roundtrip mismatch");

    context.throws([] {
        const std::array<double, 2> values{1.0, 2.0};
        static_cast<void>(fomhd::io::from_doubles(values, {3}));
    }, "shape/value count mismatch was accepted");
    context.throws([&] {
        Array wrong{unicode};
        wrong.descr = "<U4";
        static_cast<void>(fomhd::io::to_utf8(wrong));
    }, "Unicode payload length mismatch was accepted");
}

void test_npy_versions(const Context& context) {
    const Bytes payload{std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
                        std::byte{0}, std::byte{0}, std::byte{0xf8}, std::byte{0x3f}};
    const std::string dictionary{npy_dictionary("<f8", "")};
    for (const std::uint8_t version : {std::uint8_t{1}, std::uint8_t{2},
                                       std::uint8_t{3}}) {
        const Bytes encoded{make_npy(dictionary, payload, version)};
        const Array array{fomhd::io::parse_npy(encoded)};
        context.equal(array.descr, std::string{"<f8"}, "NPY version dtype mismatch");
        context.equal(array.shape.empty(), true, "NPY scalar shape mismatch");
        context.equal(fomhd::io::to_double(array), 1.5, "NPY scalar value mismatch");
    }

    const std::array<std::byte, 16> big_endian{
        std::byte{0x40}, std::byte{0x08}, std::byte{0}, std::byte{0},
        std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
        std::byte{0xbf}, std::byte{0xf8}, std::byte{0}, std::byte{0},
        std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
    };
    const Array opposite{fomhd::io::parse_npy(
        make_npy(npy_dictionary(">f8", "2"), big_endian))};
    context.equal(opposite.descr, std::string{"<f8"}, "opposite-endian dtype not canonical");
    const std::vector<double> values{fomhd::io::to_doubles(opposite)};
    context.equal(values[0], 3.0, "opposite-endian first value mismatch");
    context.equal(values[1], -1.5, "opposite-endian second value mismatch");
}

void test_npy_rejections(const Context& context) {
    const Bytes scalar_payload{std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
                               std::byte{0}, std::byte{0}, std::byte{0xf8}, std::byte{0x3f}};
    context.throws([] { static_cast<void>(fomhd::io::parse_npy({})); },
                   "empty NPY was accepted");

    Bytes truncated_magic{make_npy(npy_dictionary("<f8", ""), scalar_payload)};
    truncated_magic.resize(5);
    context.throws([&] { static_cast<void>(fomhd::io::parse_npy(truncated_magic)); },
                   "truncated NPY magic was accepted");

    Bytes truncated_header{make_npy(npy_dictionary("<f8", ""), scalar_payload)};
    truncated_header.resize(11);
    context.throws([&] { static_cast<void>(fomhd::io::parse_npy(truncated_header)); },
                   "truncated NPY header was accepted");

    Bytes truncated_payload{make_npy(npy_dictionary("<f8", ""), scalar_payload)};
    truncated_payload.pop_back();
    context.throws([&] { static_cast<void>(fomhd::io::parse_npy(truncated_payload)); },
                   "truncated NPY payload was accepted");

    const Bytes trailing{make_npy(npy_dictionary("<f8", ""),
                                  std::span<const std::byte>{scalar_payload}.first(7))};
    context.throws([&] { static_cast<void>(fomhd::io::parse_npy(trailing)); },
                   "NPY payload underflow was accepted");

    const std::string expression{
        "{'descr': '<f8', 'fortran_order': False, 'shape': (), } + __import__('os')"};
    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(make_npy(expression, scalar_payload)));
    }, "NPY header expression was accepted");

    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(
            make_npy(npy_dictionary("|O", ""), scalar_payload)));
    }, "NPY object dtype was accepted");

    const std::string duplicate_key{
        "{'descr': '<f8', 'descr': '<f8', 'fortran_order': False, 'shape': (), }"};
    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(make_npy(duplicate_key, scalar_payload)));
    }, "duplicate NPY header key was accepted");

    const std::string extra_key{
        "{'descr': '<f8', 'fortran_order': False, 'shape': (), 'extra': 1, }"};
    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(make_npy(extra_key, scalar_payload)));
    }, "unknown NPY header key was accepted");

    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(
            make_npy(npy_dictionary("<f8", "", true), scalar_payload)));
    }, "Fortran-order NPY was accepted");

    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(
            make_npy(npy_dictionary("<f8", "-1"), scalar_payload)));
    }, "negative NPY dimension was accepted");

    const std::string overflow_shape{"18446744073709551615"};
    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(make_npy(
            npy_dictionary("<f8", overflow_shape), scalar_payload)));
    }, "overflowing NPY shape was accepted");
}

void test_unicode_validation(const Context& context) {
    context.throws([] {
        const std::string embedded{"A\0B", 3};
        static_cast<void>(fomhd::io::from_utf8(embedded));
    }, "UTF-8 metadata string with embedded NUL was accepted");
    context.throws([] {
        const std::string invalid{"\xC0\xAF"};
        static_cast<void>(fomhd::io::from_utf8(invalid));
    }, "overlong UTF-8 was accepted");
    context.throws([] {
        const std::string surrogate{"\xED\xA0\x80"};
        static_cast<void>(fomhd::io::from_utf8(surrogate));
    }, "UTF-8 surrogate was accepted");

    const Bytes surrogate_utf32{std::byte{0x00}, std::byte{0xD8}, std::byte{0x00},
                                std::byte{0x00}};
    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(make_npy(
            npy_dictionary("<U1", ""), surrogate_utf32)));
    }, "UTF-32 surrogate was accepted");

    const Bytes embedded_nul{
        std::byte{'A'}, std::byte{0}, std::byte{0}, std::byte{0},
        std::byte{0}, std::byte{0}, std::byte{0}, std::byte{0},
        std::byte{'B'}, std::byte{0}, std::byte{0}, std::byte{0},
    };
    context.throws([&] {
        static_cast<void>(fomhd::io::parse_npy(make_npy(
            npy_dictionary("<U3", ""), embedded_nul)));
    }, "embedded Unicode NUL was accepted");
}

void test_npz_roundtrip(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "roundtrip.npz"};
    const std::map<std::string, Array> arrays{
        {"complex", complex_array()},
        {"doubles", double_array()},
        {"empty", fomhd::io::from_doubles(std::vector<double>{}, {2, 0, 3})},
        {"flag", fomhd::io::from_bool(false)},
        {"label", fomhd::io::from_utf8("\xE4\xB8\xAD\xE6\x96\x87")},
        {"offset", fomhd::io::from_double(-7.5)},
        {"step", fomhd::io::from_uint64(42)},
    };
    fomhd::io::write_npz(path, arrays);
    const auto restored{fomhd::io::read_npz(path, Limits{})};
    context.equal(restored.size(), arrays.size(), "NPZ key count mismatch");
    context.equal(restored.at("doubles").data, arrays.at("doubles").data,
                  "NPZ double payload mismatch");
    context.equal(fomhd::io::to_complex(restored.at("complex"))[1],
                  std::complex<double>{-0.0, 4.0}, "NPZ complex payload mismatch");
    context.equal(restored.at("empty").shape,
                  std::vector<std::size_t>{2, 0, 3}, "zero-size shape mismatch");
    context.equal(fomhd::io::to_utf8(restored.at("label")),
                  std::string{"\xE4\xB8\xAD\xE6\x96\x87"}, "NPZ Unicode mismatch");
    context.equal(fomhd::io::to_uint64(restored.at("step")), std::uint64_t{42},
                  "NPZ uint64 mismatch");
}

void test_npz_allowlist(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "allowlist.npz"};
    fomhd::io::write_npz(path, {{"allowed", fomhd::io::from_double(1.0)}});
    const auto allowed{fomhd::io::read_npz(path, Limits{}, Keys{"allowed"})};
    context.equal(allowed.size(), std::size_t{1}, "allowlist result count mismatch");
    const std::filesystem::path extra{temporary.path() / "allowlist-extra.npz"};
    fomhd::io::write_npz(extra, {{"allowed", fomhd::io::from_double(1.0)},
                                 {"rejected", fomhd::io::from_double(2.0)}});
    context.throws([&] {
        static_cast<void>(
            fomhd::io::read_npz(extra, Limits{}, Keys{"allowed"}));
    }, "allowlisted archive with an unexpected entry was accepted");
}

void test_npz_path_rejection(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "path.npz"};
    const Bytes npy{make_npy(npy_dictionary("<f8", ""), value_bytes(1.0))};
    make_zip(path, std::vector<ZipEntry>{{"../escape.npy", npy}});
    context.throws([&] { static_cast<void>(fomhd::io::read_npz(path, Limits{})); },
                   "ZIP path traversal entry was accepted");
}

void test_npz_directory_rejection(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "directory.npz"};
    make_zip(path, std::vector<ZipEntry>{{"folder/", Bytes{}}});
    context.throws([&] { static_cast<void>(fomhd::io::read_npz(path, Limits{})); },
                   "ZIP directory entry was accepted");
}

void test_npz_unknown_entry(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "unknown.npz"};
    const Bytes npy{make_npy(npy_dictionary("<f8", ""), value_bytes(1.0))};
    make_zip(path, std::vector<ZipEntry>{{"unknown.txt", npy}});
    context.throws([&] { static_cast<void>(fomhd::io::read_npz(path, Limits{})); },
                   "non-NPY ZIP entry was accepted");
}

void test_npz_duplicate_entry(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "duplicate.npz"};
    const Bytes npy{make_npy(npy_dictionary("<f8", ""), value_bytes(1.0))};
    make_zip(path, std::vector<ZipEntry>{{"same.npy", npy}, {"same.npy", npy}});
    context.throws([&] { static_cast<void>(fomhd::io::read_npz(path, Limits{})); },
                   "duplicate ZIP entry was accepted");
}

void test_npz_encrypted_entry(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "encrypted.npz"};
    const Bytes npy{make_npy(npy_dictionary("<f8", ""), value_bytes(1.0))};
    make_zip(path, std::vector<ZipEntry>{{"value.npy", npy}});
    set_zip_flag(path, 0x0001U);
    context.throws([&] { static_cast<void>(fomhd::io::read_npz(path, Limits{})); },
                   "encrypted ZIP entry was accepted");
}

void test_npz_unsupported_method(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "method.npz"};
    const Bytes npy{make_npy(npy_dictionary("<f8", ""), value_bytes(1.0))};
    make_zip(path, std::vector<ZipEntry>{{"value.npy", npy}});
    set_zip_field(path, 0x04034b50U, 8U, 99U);
    set_zip_field(path, 0x02014b50U, 10U, 99U);
    context.throws([&] { static_cast<void>(fomhd::io::read_npz(path, Limits{})); },
                   "unsupported ZIP compression method was accepted");
}

void test_npz_crc_corruption(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "crc.npz"};
    const double value{123.456};
    const Array array{fomhd::io::from_double(value)};
    fomhd::io::write_npz(path, {{"value", array}});
    corrupt_payload(path, value_bytes(value));
    context.throws([&] { static_cast<void>(fomhd::io::read_npz(path, Limits{})); },
                   "ZIP CRC corruption was accepted");
}

void test_npz_limits(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "limits.npz"};
    fomhd::io::write_npz(path, {{"value", double_array()}});
    const auto file_size{static_cast<std::size_t>(std::filesystem::file_size(path))};
    context.throws([&] {
        Limits limits{};
        limits.max_input_bytes = file_size - 1U;
        static_cast<void>(fomhd::io::read_npz(path, limits));
    }, "input byte limit was not enforced");
    context.throws([&] {
        Limits limits{};
        limits.max_expanded_bytes = 8U;
        static_cast<void>(fomhd::io::read_npz(path, limits));
    }, "expanded byte limit was not enforced");
    context.throws([&] {
        Limits limits{};
        limits.max_entries = 0U;
        static_cast<void>(fomhd::io::read_npz(path, limits));
    }, "entry limit was not enforced");
    context.throws([&] {
        Limits limits{};
        limits.max_header_bytes = 16U;
        static_cast<void>(fomhd::io::read_npz(path, limits));
    }, "NPY header limit was not enforced");
}

void test_npz_no_overwrite(const Context& context) {
    TemporaryDirectory temporary;
    const std::filesystem::path path{temporary.path() / "existing.npz"};
    const std::map<std::string, Array> original{{"value", fomhd::io::from_double(1.0)}};
    fomhd::io::write_npz(path, original);
    const Bytes before{read_bytes(path)};
    context.throws([&] {
        fomhd::io::write_npz(path, {{"value", fomhd::io::from_double(2.0)}});
    }, "existing NPZ was overwritten");
    context.equal(read_bytes(path), before, "existing NPZ changed after refusal");
}

void test_sha256(const Context& context) {
    TemporaryDirectory temporary;
    context.equal(fomhd::io::sha256(std::string_view{}),
                  std::string{"e3b0c44298fc1c149afbf4c8996fb924"
                              "27ae41e4649b934ca495991b7852b855"},
                  "empty SHA-256 mismatch");
    context.equal(fomhd::io::sha256(std::string_view{"abc"}),
                  std::string{"ba7816bf8f01cfea414140de5dae2223"
                              "b00361a396177a9cb410ff61f20015ad"},
                  "text SHA-256 mismatch");
    const std::filesystem::path path{temporary.path() / "hash-input.txt"};
    write_text(path, "abc");
    context.equal(fomhd::io::sha256_file(path),
                  std::string{"ba7816bf8f01cfea414140de5dae2223"
                              "b00361a396177a9cb410ff61f20015ad"},
                  "file SHA-256 mismatch");
    context.throws([&] {
        static_cast<void>(fomhd::io::sha256_file(temporary.path() / "missing"));
    }, "missing SHA-256 file was accepted");
}

void test_parser_limits(const Context& c) {
    auto a=fomhd::io::from_double(2.0);auto bytes=fomhd::io::encode_npy(a);
    Limits limits;limits.max_input_bytes=bytes.size()-1;
    c.throws([&]{(void)fomhd::io::parse_npy(bytes,limits);},"public NPY input limit ignored");
    limits=Limits{};limits.max_expanded_bytes=4;
    c.throws([&]{(void)fomhd::io::parse_npy(bytes,limits);},"public NPY payload limit ignored");
    c.throws([&]{(void)fomhd::io::parse_npy(make_npy(npy_dictionary("<U0","2"),{}));},"zero-width Unicode accepted");
    c.throws([&]{(void)fomhd::io::parse_npy(make_npy("{'descr':'<f8','fortran_order':False,'shape':(1)}",value_bytes(2.0)));},"non-tuple singleton accepted");
    c.throws([&]{(void)fomhd::io::parse_npy(make_npy(npy_dictionary("<f8","01"),value_bytes(2.0)));},"noncanonical dimension accepted");
}

struct TestCase {
    std::string_view name;
    std::function<void(const Context&)> run;
};

std::vector<TestCase> test_cases() {
    return {
        {"parser_limits", test_parser_limits},
        {"helpers", test_helpers},
        {"npy_versions", test_npy_versions},
        {"npy_rejections", test_npy_rejections},
        {"unicode_validation", test_unicode_validation},
        {"npz_roundtrip", test_npz_roundtrip},
        {"npz_allowlist", test_npz_allowlist},
        {"npz_path_rejection", test_npz_path_rejection},
        {"npz_directory_rejection", test_npz_directory_rejection},
        {"npz_unknown_entry", test_npz_unknown_entry},
        {"npz_duplicate_entry", test_npz_duplicate_entry},
        {"npz_encrypted_entry", test_npz_encrypted_entry},
        {"npz_unsupported_method", test_npz_unsupported_method},
        {"npz_crc_corruption", test_npz_crc_corruption},
        {"npz_limits", test_npz_limits},
        {"npz_no_overwrite", test_npz_no_overwrite},
        {"sha256", test_sha256},
    };
}

}  // namespace

int main(int argc, char** argv) {
    std::optional<std::string_view> selected{};
    if (argc == 3 && std::string_view{argv[1]} == "--run") {
        selected = argv[2];
    } else if (argc != 1) {
        std::cerr << "usage: npz_tests [--run TEST_NAME]\n";
        return 2;
    }

    const Context context;
    std::size_t failures{0};
    std::size_t executed{0};
    for (const TestCase& test : test_cases()) {
        if (selected.has_value() && *selected != test.name) {
            continue;
        }
        ++executed;
        std::cout << "[RUN] " << test.name << '\n';
        try {
            test.run(context);
            std::cout << "[PASS] " << test.name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "[FAIL] " << test.name << ": " << error.what() << '\n';
        }
    }
    if (executed == 0U) {
        std::cerr << "no test matched the requested name\n";
        return 2;
    }
    std::cout << executed << " test(s), " << failures << " failure(s)\n";
    return failures == 0U ? EXIT_SUCCESS : EXIT_FAILURE;
}
