#include "fomhd/hash.hpp"
#include "fomhd/npz.hpp"

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <ranges>
#include <string>
#include <vector>

namespace {

std::string shape_text(const std::vector<std::size_t>& shape) {
    std::string text;
    for (const std::size_t dimension : shape) {
        if (!text.empty()) {
            text.push_back(',');
        }
        text += std::to_string(dimension);
    }
    return text;
}

int read_probe(const std::filesystem::path& path) {
    const auto arrays{fomhd::io::read_npz(path, fomhd::io::Limits{})};
    for (const auto& [key, array] : arrays) {
        std::cout << key << '\t' << array.descr << '\t' << shape_text(array.shape) << '\t'
                  << fomhd::io::sha256(array.data) << '\n';
    }
    return EXIT_SUCCESS;
}

int write_probe(const std::filesystem::path& path) {
    const std::vector<double> doubles{1.25, -2.5, 0.0, 8.75};
    const std::vector<std::complex<double>> complex{
        {-1.5, -2.25},
        {-0.0, 4.0},
        {9.0, -0.0},
    };
    fomhd::io::write_npz(
        path,
        {
            {"complex", fomhd::io::from_complex(complex, {3})},
            {"doubles", fomhd::io::from_doubles(doubles, {2, 2})},
            {"empty", fomhd::io::from_doubles(std::vector<double>{}, {2, 0, 3})},
            {"flag", fomhd::io::from_bool(true)},
            {"label", fomhd::io::from_utf8("\xE4\xB8\xAD\xE6\x96\x87")},
            {"offset", fomhd::io::from_double(-7.5)},
            {"step", fomhd::io::from_uint64(UINT64_MAX)},
        });
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: npz_probe read|write PATH\n";
            return 2;
        }
        const std::string command{argv[1]};
        const std::filesystem::path path{argv[2]};
        if (command == "read") {
            return read_probe(path);
        }
        if (command == "write") {
            return write_probe(path);
        }
        std::cerr << "unknown probe command\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "npz_probe: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
