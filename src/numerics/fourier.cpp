#include "fomhd/numerics.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

namespace fomhd {
namespace {

class FftwBuffer {
public:
    explicit FftwBuffer(std::size_t count)
        : data_(fftw_alloc_complex(count)) {
        if (data_ == nullptr) {
            throw NumericalError("FFTW allocation failed");
        }
    }

    ~FftwBuffer() { fftw_free(data_); }

    FftwBuffer(const FftwBuffer&) = delete;
    FftwBuffer& operator=(const FftwBuffer&) = delete;

    fftw_complex* data() { return data_; }
    const fftw_complex* data() const { return data_; }

private:
    fftw_complex* data_{};
};

class FftwPlan {
public:
    explicit FftwPlan(fftw_plan plan) : plan_(plan) {
        if (plan_ == nullptr) {
            throw NumericalError("FFTW plan creation failed");
        }
    }

    ~FftwPlan() { fftw_destroy_plan(plan_); }

    FftwPlan(const FftwPlan&) = delete;
    FftwPlan& operator=(const FftwPlan&) = delete;

    void execute() const { fftw_execute(plan_); }

private:
    fftw_plan plan_{};
};

std::size_t checked_fftw_batch(std::size_t value, const std::string& name) {
    if (value > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw NumericalError(name + " exceeds FFTW int dimensions");
    }
    return value;
}

void copy_to_fftw(const Vec& source, fftw_complex* destination) {
    for (std::size_t i = 0; i < source.size(); ++i) {
        destination[i][0] = source[i].real();
        destination[i][1] = source[i].imag();
    }
}

Vec copy_from_fftw(const fftw_complex* source, std::size_t count) {
    Vec result(count);
    for (std::size_t i = 0; i < count; ++i) {
        result[i] = Complex{source[i][0], source[i][1]};
    }
    return result;
}

void validate_entities(int N, std::size_t entities) {
    if (entities == 0) {
        throw NumericalError("Fourier batch must contain at least one entity");
    }
    checked_fftw_batch(entities, "Fourier entity count");
    static_cast<void>(checked_product(static_cast<std::size_t>(N), entities,
                                      "Fourier storage"));
}

std::size_t bin_index(int mode, int N) {
    int value = mode % N;
    if (value < 0) {
        value += N;
    }
    return static_cast<std::size_t>(value);
}

Vec execute_transform(const Vec& input, int N, int entities, int sign) {
    const std::size_t count =
        static_cast<std::size_t>(N) * static_cast<std::size_t>(entities);
    FftwBuffer input_buffer(count);
    FftwBuffer output_buffer(count);
    copy_to_fftw(input, input_buffer.data());
    int length = N;
    int howmany = entities;
    int in_embed = N;
    int out_embed = N;
    int in_stride = entities;
    int out_stride = entities;
    int input_distance = 1;
    int output_distance = 1;
    FftwPlan plan(fftw_plan_many_dft(
        1, &length, howmany, input_buffer.data(), &in_embed, in_stride,
        input_distance, output_buffer.data(), &out_embed, out_stride,
        output_distance, sign, FFTW_ESTIMATE));
    plan.execute();
    return copy_from_fftw(output_buffer.data(), count);
}

}  // namespace

Fourier::Fourier(int retained_k, int nphi) : K(retained_k), N(nphi) {
    if (K < 0 || N <= 0 || N % 2 == 0 || N < 4 * K + 1) {
        throw NumericalError(
            "Fourier requires K>=0 and odd N>=4*K+1");
    }
}

Vec Fourier::inverse(const Vec& hat, std::size_t entities,
                     bool physical_real, double tol) const {
    validate_entities(N, entities);
    const std::size_t modes = static_cast<std::size_t>(2 * K + 1);
    if (hat.size() != checked_product(modes, entities, "retained Fourier")) {
        throw NumericalError("retained Fourier inverse size mismatch");
    }
    if (!is_finite(hat) || !(tol >= 0.0)) {
        throw NumericalError("retained Fourier inverse input is invalid");
    }
    Vec bins(static_cast<std::size_t>(N) * entities);
    for (int stored = 0; stored <= 2 * K; ++stored) {
        const int mode = stored - K;
        for (std::size_t entity = 0; entity < entities; ++entity) {
            bins[bin_index(mode, N) * entities + entity] =
                hat[static_cast<std::size_t>(stored) * entities + entity];
        }
    }
    Vec samples =
        execute_transform(bins, N, static_cast<int>(entities), FFTW_BACKWARD);
    if (!physical_real) {
        return samples;
    }
    if (hermitian_defect(hat, entities) > tol) {
        throw NumericalError("retained spectrum is not Hermitian");
    }
    Vec real_samples(samples.size());
    double max_imaginary = 0.0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        max_imaginary = std::max(max_imaginary, std::abs(samples[i].imag()));
        real_samples[i] = samples[i].real();
    }
    if (max_imaginary > tol) {
        throw NumericalError("inverse physical samples have imaginary residue");
    }
    return real_samples;
}

Vec Fourier::forward(const Vec& samples, std::size_t entities) const {
    validate_entities(N, entities);
    if (samples.size() !=
            checked_product(static_cast<std::size_t>(N), entities,
                            "sample Fourier") ||
        !is_finite(samples)) {
        throw NumericalError("sample Fourier input is invalid");
    }
    Vec full =
        execute_transform(samples, N, static_cast<int>(entities), FFTW_FORWARD);
    const double inverse_scale = 1.0 / static_cast<double>(N);
    Vec hat(static_cast<std::size_t>(2 * K + 1) * entities);
    for (int stored = 0; stored <= 2 * K; ++stored) {
        const int mode = stored - K;
        const std::size_t source_bin = bin_index(mode, N) * entities;
        for (std::size_t entity = 0; entity < entities; ++entity) {
            hat[static_cast<std::size_t>(stored) * entities + entity] =
                full[source_bin + entity] * inverse_scale;
        }
    }
    return hat;
}

Vec Fourier::derivative(const Vec& samples, std::size_t entities) const {
    validate_entities(N, entities);
    if (samples.size() !=
            checked_product(static_cast<std::size_t>(N), entities,
                            "derivative Fourier") ||
        !is_finite(samples)) {
        throw NumericalError("derivative Fourier input is invalid");
    }
    Vec full =
        execute_transform(samples, N, static_cast<int>(entities), FFTW_FORWARD);
    const double inverse_scale = 1.0 / static_cast<double>(N);
    for (int bin = 0; bin < N; ++bin) {
        const int signed_mode = bin <= (N - 1) / 2 ? bin : bin - N;
        const Complex factor{0.0, static_cast<double>(signed_mode)};
        for (std::size_t entity = 0; entity < entities; ++entity) {
            const std::size_t index =
                static_cast<std::size_t>(bin) * entities + entity;
            full[index] = full[index] * factor * inverse_scale;
        }
    }
    return execute_transform(full, N, static_cast<int>(entities), FFTW_BACKWARD);
}

double Fourier::hermitian_defect(const Vec& hat, std::size_t entities) const {
    validate_entities(N, entities);
    if (hat.size() !=
        checked_product(static_cast<std::size_t>(2 * K + 1), entities,
                        "Hermitian Fourier")) {
        throw NumericalError("Hermitian defect input size mismatch");
    }
    double defect = 0.0;
    for (std::size_t entity = 0; entity < entities; ++entity) {
        defect = std::max(
            defect,
            std::abs(hat[static_cast<std::size_t>(K) * entities + entity].imag()));
    }
    for (int mode = 1; mode <= K; ++mode) {
        for (std::size_t entity = 0; entity < entities; ++entity) {
            const Complex negative =
                hat[static_cast<std::size_t>(K - mode) * entities + entity];
            const Complex positive =
                hat[static_cast<std::size_t>(K + mode) * entities + entity];
            defect = std::max(defect, std::abs(negative - std::conj(positive)));
        }
    }
    return defect;
}

}  // namespace fomhd
