#include "fomhd/backend.hpp"
#include "fomhd/physics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numbers>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using fomhd::Complex;
using fomhd::Compute;
using fomhd::Context;
using fomhd::FluidContext;
using fomhd::Geometry;
using fomhd::Operators;
using fomhd::Parameters;
using fomhd::StageResult;
using fomhd::State;
using fomhd::Vec;

constexpr Complex kI{0.0, 1.0};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void near_complex(Complex actual, Complex expected, double tolerance,
                  const std::string& message) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        throw std::runtime_error(message + ": actual=(" +
                                 std::to_string(actual.real()) + "," +
                                 std::to_string(actual.imag()) + ") expected=(" +
                                 std::to_string(expected.real()) + "," +
                                 std::to_string(expected.imag()) + ")");
    }
}

Parameters parameters() {
    Parameters p;
    p.species = {
        fomhd::Species{"ion", 1.67262192595e-27, 1.602176634e-19,
                       5.0 / 3.0, 0.0, 0.0, 0.0, 0.0},
        fomhd::Species{"electron", 9.1093837139e-31, -1.602176634e-19,
                       0.0, 0.0, 0.0, 0.0, 0.0},
    };
    p.epsilon0 = 8.8541878188e-12;
    p.mu0 = 1.25663706127e-6;
    p.B_min = 1.0e-8;
    p.dt = 1.0e-9;
    p.start_time = 0.0;
    p.rank_rtol = 1.0e-12;
    p.nsteps = 1;
    p.K = 1;
    p.Nphi = 9;
    p.memory_limit = 512U * 1024U * 1024U;
    p.device_memory_limit = 512U * 1024U * 1024U;
    p.inner_distance = 0.03;
    p.ref = fomhd::References{0.15, 1.0e-6, 1.0e15, 1.0e15, 1.0e5,
                              1.0e4, 1.0e-2, 1.0, 1.0e-2, 1.0e-1};
    // Match the KKT reference current rate to the physical electric response.
    const double response = std::pow(p.species[0].charge,2)*p.ref.ni/p.species[0].mass
        + std::pow(p.species[1].charge,2)*p.ref.ne/p.species[1].mass;
    p.ref.current=p.ref.time*p.ref.electric*(response+1/(p.mu0*p.ref.length*p.ref.length));
    auto vector_budget=[&](const std::string& name,const std::string& unit,double reference) {
        p.budgets[name+".linf"]={0,1e-10,reference,unit,fomhd::Norm::linf,true};
        p.budgets[name+".weighted_L2"]={0,1e-10,reference,unit,fomhd::Norm::weighted_l2,true};
    };
    vector_budget("gauss","V/m^2",1e13);
    vector_budget("longitudinal_projection","A/m^3",1e4);
    vector_budget("darwin_dynamic","A/(m^2*s)",p.ref.current/p.ref.time);
    vector_budget("transverse_divergence","V/m^2",p.ref.electric/p.ref.length);
    vector_budget("CT_increment","T/(m*s)",p.ref.electric/(p.ref.length*p.ref.length));
    return p;
}

class Fixture {
public:
    Fixture()
        : p(parameters()),
          g(fomhd::build_rectangle(7, 7, 0.5, -0.5, 0.15, 0.15)),
          op(fomhd::build_operators(g, p.K)),
          compute(),
          c{p, g, op, compute, p.Nphi, 0, 1, 1, 1, "", "", ""},
          tag{0, 1, 1, 0, 0.0, "test"} {}

    std::size_t mode_offset(int mode, std::size_t entities) const {
        return static_cast<std::size_t>(mode + p.K) * entities;
    }

    Parameters p;
    Geometry g;
    Operators op;
    Compute compute;
    Context c;
    fomhd::StageTag tag;
};

Vec uniform_bhat(const Fixture& f) {
    Vec result(static_cast<std::size_t>(f.p.Nphi) * 3 * f.op.Nc);
    for (int sample = 0; sample < f.p.Nphi; ++sample) {
        for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
            const std::size_t base =
                static_cast<std::size_t>(sample) * 3 * f.op.Nc;
            result[base + cell] = Complex{0.6};
            result[base + f.op.Nc + cell] = Complex{0.8};
            result[base + 2 * f.op.Nc + cell] = Complex{0.0};
        }
    }
    return result;
}

Vec uniform_BN(const Fixture& f) {
    return fomhd::scaled(uniform_bhat(f), 2.0);
}

template <typename DensityLaw, typename VelocityLaw>
State sampled_state(Fixture& f, DensityLaw density_law,
                    VelocityLaw velocity_law, double ion_pressure,
                    double electron_perp, double electron_parallel) {
    State state = fomhd::allocate_state(f.c);
    const std::size_t scalar_samples =
        static_cast<std::size_t>(f.p.Nphi) * f.op.Nc;
    const std::size_t vector_samples =
        static_cast<std::size_t>(f.p.Nphi) * 3 * f.op.Nc;
    Vec density(scalar_samples);
    Vec momentum(vector_samples);
    Vec perp(scalar_samples);
    Vec parallel(scalar_samples);
    for (int sample = 0; sample < f.p.Nphi; ++sample) {
        const double phi =
            2.0 * std::numbers::pi * static_cast<double>(sample) /
            static_cast<double>(f.p.Nphi);
        for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
            const auto velocity = velocity_law(f.g.cells.R[cell],
                                               f.g.cells.Z[cell], phi);
            const double n = density_law(f.g.cells.R[cell],
                                         f.g.cells.Z[cell], phi);
            const std::size_t s = static_cast<std::size_t>(sample);
            density[s * f.op.Nc + cell] = n;
            perp[s * f.op.Nc + cell] = electron_perp;
            parallel[s * f.op.Nc + cell] = electron_parallel;
            for (std::size_t component = 0; component < 3; ++component) {
                momentum[(s * 3 + component) * f.op.Nc + cell] =
                    f.p.species[0].mass * n * velocity[component];
            }
        }
    }
    for (int species = 0; species < 2; ++species) {
        const std::size_t s = static_cast<std::size_t>(species);
        state.fluid[s].n =
            f.compute.forward_fft(density, f.op.Nc, f.p.K, f.p.Nphi);
        state.fluid[s].mom =
            fomhd::scaled(f.compute.forward_fft(momentum, 3 * f.op.Nc, f.p.K, f.p.Nphi),
                          f.p.species[s].mass / f.p.species[0].mass);
        std::fill(state.fluid[s].perp.begin(),
                  state.fluid[s].perp.end(), Complex{});
        if (species == 0) {
            for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
                state.fluid[s].perp[
                    f.mode_offset(0, f.op.Nc) + cell] =
                    Complex{ion_pressure};
            }
        } else {
            state.fluid[s].perp =
                f.compute.forward_fft(perp, f.op.Nc, f.p.K, f.p.Nphi);
            state.fluid[s].parallel =
                f.compute.forward_fft(parallel, f.op.Nc, f.p.K, f.p.Nphi);
        }
    }
    return state;
}

void make_wall_compatible(Fixture& f, State& state) {
    const Vec BN = uniform_BN(f);
    const Vec bhat = uniform_bhat(f);
    for (int species = 0; species < 2; ++species) {
        const std::size_t s = static_cast<std::size_t>(species);
        const FluidContext fluid =
            fomhd::build_fluid(state.fluid[s], species, BN, bhat,
                               f.tag, f.c);
        const fomhd::Projection projection =
            fomhd::project_wall(fluid.wall, fluid.mom, f.c);
        state.fluid[s].mom =
            fomhd::forward(projection.value, 3 * f.op.Nc, f.c);
    }
}

State constant_gradient_state(Fixture& f) {
    const double density = f.p.ref.ni;
    return sampled_state(
        f,
        [density](double, double, double) { return density; },
        [](double R, double Z, double) {
            return std::array<double, 3>{2.0 * R, 3.0 * R, 5.0 * Z};
        },
        7.0, 2.0, 5.0);
}

State varying_gradient_state(Fixture& f) {
    const double density = f.p.ref.ni;
    return sampled_state(
        f,
        [density](double, double, double phi) {
            return density * (1.0 + 0.03 * std::cos(phi));
        },
        [](double R, double Z, double phi) {
            return std::array<double, 3>{
                2.0e4 * (R - 0.8),
                3.0e4 * (Z + 0.2) + 1.0e3 * std::sin(phi),
                1.5e4 * (R + Z),
            };
        },
        7.0, 2.0, 5.0);
}

void prepare_fluid_stage(Fixture& f, State& state, StageResult& stage) {
    stage.tag = f.tag;
    stage.rhs = fomhd::allocate_state(f.c);
    stage.BN = uniform_BN(f);
    stage.bhat = uniform_bhat(f);
    for (int species = 0; species < 2; ++species) {
        stage.fluid[static_cast<std::size_t>(species)] =
            fomhd::build_fluid(state.fluid[static_cast<std::size_t>(species)],
                               species, stage.BN, stage.bhat, f.tag, f.c);
        fomhd::build_non_electric(
            stage.fluid[static_cast<std::size_t>(species)], species,
            stage.BN, stage.bhat, f.c);
    }
}

Vec no_electric_current(Fixture& f, const StageResult& stage) {
    Vec result(
        static_cast<std::size_t>(2 * f.p.K + 1) * f.op.Ne);
    for (int species = 0; species < 2; ++species) {
        const FluidContext& fluid = stage.fluid[static_cast<std::size_t>(species)];
        const auto projection = fomhd::project_wall(
            fluid.wall, fluid.non_electric, f.c);
        Vec response = fomhd::scaled(
            projection.value,
            f.p.species[static_cast<std::size_t>(species)].charge /
                f.p.species[static_cast<std::size_t>(species)].mass);
        const Vec edge_samples =
            fomhd::sample_apply(f.op.I, response, f.p.Nphi, f.compute);
        result = fomhd::add(
            result, fomhd::forward(edge_samples, f.op.Ne, f.c));
    }
    return result;
}

Vec charge_density(Fixture& f, const StageResult& stage) {
    Vec samples(
        static_cast<std::size_t>(f.p.Nphi) * f.op.Nc);
    for (int species = 0; species < 2; ++species) {
        const FluidContext& fluid = stage.fluid[static_cast<std::size_t>(species)];
        const double charge =
            f.p.species[static_cast<std::size_t>(species)].charge;
        for (std::size_t index = 0; index < samples.size(); ++index) {
            samples[index] += charge * fluid.n[index];
        }
    }
    return fomhd::forward(samples, f.op.Nc, f.c);
}

Complex sample_integral(const Fixture& f, const Vec& values,
                        std::size_t components = 1) {
    require(values.size() ==
                static_cast<std::size_t>(f.p.Nphi) * components * f.op.Nc,
            "sample integral shape");
    Complex result{};
    for (int sample = 0; sample < f.p.Nphi; ++sample) {
        for (std::size_t component = 0; component < components; ++component) {
            for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
                result += f.op.M0[cell] *
                          values[(static_cast<std::size_t>(sample) *
                                      components +
                                  component) *
                                     f.op.Nc +
                                 cell];
            }
        }
    }
    return result * (2.0 * std::numbers::pi /
                     static_cast<double>(f.p.Nphi));
}

Complex edge_integral(const Fixture& f, const Vec& values) {
    require(values.size() ==
                static_cast<std::size_t>(f.p.Nphi) * f.op.Ne,
            "edge integral shape");
    Complex result{};
    for (int sample = 0; sample < f.p.Nphi; ++sample) {
        for (std::size_t edge = 0; edge < f.op.Ne; ++edge) {
            result += f.op.M1[edge] *
                      values[static_cast<std::size_t>(sample) * f.op.Ne +
                             edge];
        }
    }
    return result * (2.0 * std::numbers::pi /
                     static_cast<double>(f.p.Nphi));
}

Vec expected_pressure_tensor(const Fixture& f, const FluidContext& fluid,
                             int species, const Vec& bhat) {
    Vec result(static_cast<std::size_t>(f.p.Nphi) * 9 * f.op.Nc);
    for (int sample = 0; sample < f.p.Nphi; ++sample) {
        for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
            const std::size_t scalar =
                static_cast<std::size_t>(sample) * f.op.Nc + cell;
            const std::size_t vector =
                static_cast<std::size_t>(sample) * 3 * f.op.Nc;
            const std::size_t tensor =
                static_cast<std::size_t>(sample) * 9 * f.op.Nc;
            for (std::size_t a = 0; a < 3; ++a) {
                for (std::size_t b = 0; b < 3; ++b) {
                    Complex pressure{};
                    if (species == 0) {
                        pressure = a == b ? fluid.perp[scalar] : Complex{};
                    } else {
                        pressure =
                            (a == b ? fluid.perp[scalar] : Complex{}) +
                            (fluid.parallel[scalar] - fluid.perp[scalar]) *
                                bhat[vector + a * f.op.Nc + cell] *
                                bhat[vector + b * f.op.Nc + cell];
                    }
                    result[tensor + (a * 3 + b) * f.op.Nc + cell] = pressure;
                }
            }
        }
    }
    return result;
}

Vec deterministic_vector(std::size_t size, std::size_t seed,
                         bool imaginary = false) {
    Vec result(size);
    for (std::size_t i = 0; i < size; ++i) {
        const double real =
            (static_cast<double>((i * 13 + seed * 17) % 29) - 14.0) / 7.0;
        const double imag =
            imaginary
                ? (static_cast<double>((i * 7 + seed * 11) % 23) - 11.0) / 9.0
                : 0.0;
        result[i] = Complex{real, imag};
    }
    return result;
}

Complex weighted_dot(const Vec& left, const Vec& right,
                     const fomhd::RealVec& weights) {
    require(left.size() == right.size() && left.size() == weights.size(),
            "weighted dot shape");
    Complex result{};
    for (std::size_t i = 0; i < left.size(); ++i) {
        result += weights[i] * std::conj(left[i]) * right[i];
    }
    return result;
}

fomhd::RealVec repeat_real_weights(const fomhd::RealVec& weights,
                                   std::size_t count) {
    fomhd::RealVec result;
    result.reserve(weights.size() * count);
    for (std::size_t repeat = 0; repeat < count; ++repeat) {
        result.insert(result.end(), weights.begin(), weights.end());
    }
    return result;
}

void fluid_01_pressure_tensor_and_domain() {
    Fixture f;
    State state = sampled_state(
        f,
        [&f](double, double, double) { return f.p.ref.ni; },
        [](double, double, double) {
            return std::array<double, 3>{0.0, 0.0, 0.0};
        },
        7.0, 2.0, 5.0);
    const Vec BN = uniform_BN(f);
    const Vec bhat = uniform_bhat(f);
    const FluidContext electron =
        fomhd::build_fluid(state.fluid[1], 1, BN, bhat, f.tag, f.c);
    const std::size_t sample = 4;
    const std::size_t cell = f.g.cell_id(3, 3);
    const std::size_t tensor =
        sample * 9 * f.op.Nc + cell;
    near_complex(electron.raw_moment[tensor + (0 * 3 + 0) * f.op.Nc],
                 Complex{3.08}, 2.0e-13, "electron P_RR");
    near_complex(electron.raw_moment[tensor + (0 * 3 + 1) * f.op.Nc],
                 Complex{1.44}, 2.0e-13, "electron P_Rphi");
    near_complex(electron.raw_moment[tensor + (1 * 3 + 1) * f.op.Nc],
                 Complex{3.92}, 2.0e-13, "electron P_phiphi");
    near_complex(electron.raw_moment[tensor + (2 * 3 + 2) * f.op.Nc],
                 Complex{2.0}, 2.0e-13, "electron P_ZZ");

    const FluidContext ion =
        fomhd::build_fluid(state.fluid[0], 0, BN, bhat, f.tag, f.c);
    near_complex(ion.raw_moment[tensor + (0 * 3 + 0) * f.op.Nc],
                 Complex{7.0}, 2.0e-13, "ion P_RR");
    near_complex(ion.raw_moment[tensor + (0 * 3 + 1) * f.op.Nc],
                 Complex{}, 2.0e-13, "ion P_Rphi");
    near_complex(ion.raw_moment[tensor + (2 * 3 + 2) * f.op.Nc],
                 Complex{7.0}, 2.0e-13, "ion P_ZZ");

    state.fluid[1].n[f.mode_offset(0, f.op.Nc) + cell] = Complex{-1.0e15};
    bool rejected = false;
    try {
        static_cast<void>(
            fomhd::build_fluid(state.fluid[1], 1, BN, bhat, f.tag, f.c));
    } catch (const fomhd::Failure&) {
        rejected = true;
    }
    require(rejected, "build_fluid accepted a non-positive sample density");
}

void fluid_02_cylindrical_gradient() {
    Fixture f;
    State state = constant_gradient_state(f);
    const FluidContext ion = fomhd::build_fluid(
        state.fluid[0], 0, uniform_BN(f), uniform_bhat(f), f.tag, f.c);
    const Vec gradient = fomhd::velocity_gradient(ion, ion.u, f.c);
    const std::size_t cell = f.g.cell_id(3, 3);
    const std::size_t base = cell;
    near_complex(gradient[base + 0 * f.op.Nc], Complex{2.0}, 2.0e-12,
                 "Q_RR");
    near_complex(gradient[base + 1 * f.op.Nc], Complex{-3.0}, 2.0e-12,
                 "Q_Rphi");
    near_complex(gradient[base + 2 * f.op.Nc], Complex{}, 2.0e-12,
                 "Q_RZ");
    near_complex(gradient[base + 3 * f.op.Nc], Complex{3.0}, 2.0e-12,
                 "Q_phiR");
    near_complex(gradient[base + 4 * f.op.Nc], Complex{2.0}, 2.0e-12,
                 "Q_phiphi");
    near_complex(gradient[base + 5 * f.op.Nc], Complex{}, 2.0e-12,
                 "Q_phiZ");
    near_complex(gradient[base + 6 * f.op.Nc], Complex{}, 2.0e-12,
                 "Q_ZR");
    near_complex(gradient[base + 7 * f.op.Nc], Complex{}, 2.0e-12,
                 "Q_Zphi");
    near_complex(gradient[base + 8 * f.op.Nc], Complex{5.0}, 2.0e-12,
                 "Q_ZZ");
}

void fluid_03_weighted_adjoint() {
    Fixture f;
    State state = varying_gradient_state(f);
    const FluidContext ion = fomhd::build_fluid(
        state.fluid[0], 0, uniform_BN(f), uniform_bhat(f), f.tag, f.c);
    const std::size_t vector_size =
        static_cast<std::size_t>(f.p.Nphi) * 3 * f.op.Nc;
    const std::size_t tensor_size =
        static_cast<std::size_t>(f.p.Nphi) * 9 * f.op.Nc;
    const Vec x = deterministic_vector(vector_size, 3, true);
    const Vec y = deterministic_vector(tensor_size, 8, true);
    const Vec gradient = fomhd::velocity_gradient(ion, x, f.c);
    const Vec adjoint = fomhd::gradient_adjoint(ion, y, f.c);
    const fomhd::RealVec tensor_weights =
        repeat_real_weights(
            f.op.M0,
            static_cast<std::size_t>(9 * f.p.Nphi));
    const fomhd::RealVec node_weights =
        repeat_real_weights(
            f.op.M0,
            static_cast<std::size_t>(3 * f.p.Nphi));
    const Complex left = weighted_dot(y, gradient, tensor_weights);
    const Complex right = weighted_dot(adjoint, x, node_weights);
    const double scale =
        std::max({1.0, fomhd::norm(y), fomhd::norm(x)});
    near_complex(left, right, 2.0e-11 * scale,
                 "weighted cylindrical gradient adjoint");
}

void fluid_04_pressure_convection_pairing() {
    Fixture f;
    State state = varying_gradient_state(f);
    make_wall_compatible(f, state);
    StageResult stage;
    prepare_fluid_stage(f, state, stage);

    const Vec electron_pressure =
        expected_pressure_tensor(f, stage.fluid[1], 1, stage.bhat);
    const fomhd::RealVec tensor_weights =
        repeat_real_weights(
            f.op.M0,
            static_cast<std::size_t>(9 * f.p.Nphi));
    const Complex pressure_contraction =
        (2.0 * std::numbers::pi / f.p.Nphi) * weighted_dot(electron_pressure, stage.fluid[1].gradient,
                     tensor_weights);
    const Vec recomputed_pressure_force =
        fomhd::gradient_adjoint(stage.fluid[1], electron_pressure, f.c);
    const double force_fixture_defect = fomhd::max_abs(fomhd::subtract(
        recomputed_pressure_force, stage.fluid[1].pressure_force));
    const Complex pressure_work = sample_integral(
        f, Vec([&] {
            const auto& fluid = stage.fluid[1];
            Vec values(static_cast<std::size_t>(f.p.Nphi) * 3 * f.op.Nc);
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = std::conj(fluid.u[i]) * fluid.pressure_force[i];
            }
            return values;
        }()), 3);
    near_complex(pressure_work, pressure_contraction, 2.0e-9 *
                 std::max(1.0, std::abs(pressure_contraction)),
                 "pressure force and tensor contraction; recomputed force defect=" +
                     std::to_string(force_fixture_defect));

    const Complex ion_internal =
        sample_integral(f, stage.fluid[0].pdot_perp);
    const Complex ion_theta = sample_integral(
        f, Vec([&] {
            Vec values(static_cast<std::size_t>(f.p.Nphi) * f.op.Nc);
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = std::conj(stage.fluid[0].perp[i]) *
                            stage.fluid[0].theta[i];
            }
            return values;
        }()));
    near_complex(ion_internal / (f.p.species[0].gamma - 1.0),
                 -ion_theta, 2.0e-9,
                 "ion internal-energy pressure sign");

    Vec electron_rate(
        static_cast<std::size_t>(f.p.Nphi) * f.op.Nc);
    for (std::size_t i = 0; i < electron_rate.size(); ++i) {
        electron_rate[i] = stage.fluid[1].pdot_perp[i] +
                           0.5 * stage.fluid[1].pdot_parallel[i];
    }
    near_complex(sample_integral(f, electron_rate),
                 -pressure_contraction, 2.0e-9 *
                     std::max(1.0, std::abs(pressure_contraction)),
                 "CGL internal-energy pressure sign");

    Vec convection_pairing(
        static_cast<std::size_t>(f.p.Nphi) * 3 * f.op.Nc);
    for (int sample = 0; sample < f.p.Nphi; ++sample) {
        for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
            Complex speed_squared{};
            for (std::size_t component = 0; component < 3; ++component) {
                const std::size_t index =
                    (static_cast<std::size_t>(sample) * 3 + component) *
                        f.op.Nc +
                    cell;
                const Complex value = std::conj(stage.fluid[0].u[index]);
                convection_pairing[index] =
                    value * stage.fluid[0].convection[index];
                speed_squared += std::norm(stage.fluid[0].u[index]);
            }
            const std::size_t cell_index =
                static_cast<std::size_t>(sample) * f.op.Nc + cell;
            convection_pairing[static_cast<std::size_t>(sample) * 3 *
                                   f.op.Nc +
                               cell] -=
                0.5 * f.p.species[0].mass * speed_squared *
                stage.fluid[0].divergence[cell_index];
        }
    }
    const double convection_scale =
        std::max(1.0, fomhd::norm(convection_pairing));
    require(std::abs(sample_integral(f, convection_pairing, 3)) <
                2.0e-10 * convection_scale,
            "convection does not pair with density kinetic change");
}

void fluid_05_electric_magnetic_work() {
    Fixture f;
    State state = varying_gradient_state(f);
    make_wall_compatible(f, state);
    StageResult stage;
    prepare_fluid_stage(f, state, stage);

    const std::size_t radial_faces = f.op.DR.cols;
    const std::size_t axial_faces = f.op.DZ.cols;
    Vec electric_samples(
        static_cast<std::size_t>(f.p.Nphi) * f.op.Ne);
    for (int sample = 0; sample < f.p.Nphi; ++sample) {
        const double phi =
            2.0 * std::numbers::pi * static_cast<double>(sample) /
            static_cast<double>(f.p.Nphi);
        const std::size_t base =
            static_cast<std::size_t>(sample) * f.op.Ne;
        for (std::size_t face = 0; face < radial_faces; ++face) {
            electric_samples[base + face] =
                Complex{1.0 + 0.2 * std::cos(phi)};
        }
        for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
            electric_samples[base + radial_faces + cell] =
                Complex{0.7 * std::sin(phi)};
        }
        for (std::size_t face = 0; face < axial_faces; ++face) {
            electric_samples[base + radial_faces + f.op.Nc + face] =
                Complex{-0.3 + 0.1 * std::cos(phi)};
        }
    }
    const Vec electric_hat =
        fomhd::forward(electric_samples, f.op.Ne, f.c);
    const Vec force =
        fomhd::electric_response(stage.fluid[0], 0, electric_hat, f.c);
    const Vec wall_rows =
        fomhd::sample_apply(f.op.T_full, force, f.p.Nphi, f.compute);
    require(fomhd::max_abs(wall_rows) < 2.0e-11,
            "electric response retains a full wall-normal trace");

    Vec node_work(
        static_cast<std::size_t>(f.p.Nphi) * 3 * f.op.Nc);
    for (std::size_t i = 0; i < node_work.size(); ++i) {
        node_work[i] = std::conj(stage.fluid[0].u[i]) * force[i];
    }
    Vec current_work(
        static_cast<std::size_t>(f.p.Nphi) * f.op.Ne);
    for (int sample = 0; sample < f.p.Nphi; ++sample) {
        for (std::size_t edge = 0; edge < f.op.Ne; ++edge) {
            current_work[static_cast<std::size_t>(sample) * f.op.Ne + edge] =
                f.p.species[0].charge *
                std::conj(stage.fluid[0].flux[
                    static_cast<std::size_t>(sample) * f.op.Ne + edge]) *
                electric_samples[
                    static_cast<std::size_t>(sample) * f.op.Ne + edge];
        }
    }
    const Complex node_power = sample_integral(f, node_work, 3);
    const Complex edge_power = edge_integral(f, current_work);
    near_complex(node_power, edge_power,
                 2.0e-9 * std::max(1.0, std::abs(edge_power)),
                 "electric node and edge work");

    Vec magnetic_work(
        static_cast<std::size_t>(f.p.Nphi) * 3 * f.op.Nc);
    for (std::size_t i = 0; i < magnetic_work.size(); ++i) {
        magnetic_work[i] =
            std::conj(stage.fluid[0].u[i]) *
            stage.fluid[0].magnetic_force[i];
    }
    require(std::abs(sample_integral(f, magnetic_work, 3)) <
                2.0e-8,
            "magnetic force performs work");
}

void fluid_06_sample_layout_and_magnetic_strength() {
    Fixture f;
    State state = sampled_state(f,
        [&f](double, double, double phi) { return f.p.ref.ni * (1 + 0.1 * std::cos(phi)); },
        [](double, double, double) { return std::array<double,3>{2,3,5}; },
        1, 2, 3);
    auto fluid = fomhd::build_fluid(state.fluid[0],0,uniform_BN(f),uniform_bhat(f),f.tag,f.c);
    for (int l=0;l<f.p.Nphi;++l) {
        double n=f.p.ref.ni*(1+0.1*std::cos(2*std::numbers::pi*l/f.p.Nphi));
        for(std::size_t e=0;e<f.op.Ne;++e)
            near_complex(fluid.nf[l*f.op.Ne+e],n,2e-14*n,"density interpolation sample layout");
    }
    fomhd::build_non_electric(fluid,0,uniform_BN(f),uniform_bhat(f),f.c);
    const auto weak=fluid.magnetic_force;
    fomhd::build_non_electric(fluid,0,fomhd::scaled(uniform_BN(f),3),uniform_bhat(f),f.c);
    require(fomhd::max_abs(fomhd::subtract(fluid.magnetic_force,fomhd::scaled(weak,3))) < 1e-14,
            "Lorentz force must scale with magnetic magnitude");
    near_complex(weak[0], f.p.species[0].charge*fluid.n[0]*(-8.0),1e-16,"known Lorentz R force");
    near_complex(fluid.tensor[0],1.0,1e-14,"stored tensor must be thermal pressure");
    state.fluid[0].perp[f.mode_offset(0,f.op.Nc)]=-1;
    bool rejected=false;
    try { (void)fomhd::build_fluid(state.fluid[0],0,uniform_BN(f),uniform_bhat(f),f.tag,f.c); }
    catch(const fomhd::Failure&) {rejected=true;}
    require(rejected,"negative pressure must fail before arithmetic");
}

void field_01_poisson_all_rows() {
    Fixture f;
    Vec rho(
        static_cast<std::size_t>(2 * f.p.K + 1) * f.op.Nc);
    const Vec base = deterministic_vector(f.op.Nc, 5);
    for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
        rho[f.mode_offset(0, f.op.Nc) + cell] = base[cell];
        rho[f.mode_offset(1, f.op.Nc) + cell] =
            Complex{0.2 + 0.1 * static_cast<double>(cell),
                    -0.1 + 0.03 * static_cast<double>(cell)};
        rho[f.mode_offset(-1, f.op.Nc) + cell] =
            std::conj(rho[f.mode_offset(1, f.op.Nc) + cell]);
    }
    Complex net_charge{};
    for (std::size_t cell = 0; cell < f.op.Nc; ++cell) {
        net_charge += f.op.M0[cell] *
                      rho[f.mode_offset(0, f.op.Nc) + cell];
    }
    require(std::abs(net_charge) > 1.0e-3,
            "Poisson fixture must have nonzero net charge");

    const fomhd::FieldCache cache =
        fomhd::electrostatic(rho, f.tag, f.c);
    for (int mode = -f.p.K; mode <= f.p.K; ++mode) {
        const std::size_t k =
            static_cast<std::size_t>(mode + f.p.K);
        const Vec source = fomhd::slice(rho, k * f.op.Nc, f.op.Nc);
        const Vec potential =
            fomhd::slice(cache.phi, k * f.op.Nc, f.op.Nc);
        const Vec field =
            fomhd::slice(cache.EL, k * f.op.Ne, f.op.Ne);
        const Vec poisson_residual = fomhd::subtract(
            f.compute.mv(f.op.Lphi[k], potential),
            fomhd::scaled(source, 1.0 / f.p.epsilon0));
        const Vec gauss_residual = fomhd::subtract(
            f.compute.mv(f.op.DJ[k], field),
            fomhd::scaled(source, 1.0 / f.p.epsilon0));
        const double rhs_scale = std::max(
            1.0, fomhd::max_abs(fomhd::scaled(
                     source, 1.0 / f.p.epsilon0)));
        if (fomhd::max_abs(poisson_residual) >=
            f.p.rank_rtol * rhs_scale) {
            throw std::runtime_error(
                "Poisson residual mode " + std::to_string(mode) +
                " max=" + std::to_string(fomhd::max_abs(poisson_residual)) +
                " rhs=" + std::to_string(rhs_scale));
        }
        if (fomhd::max_abs(gauss_residual) >=
            f.p.rank_rtol * rhs_scale) {
            throw std::runtime_error(
                "Gauss residual mode " + std::to_string(mode) +
                " max=" + std::to_string(fomhd::max_abs(gauss_residual)) +
                " rhs=" + std::to_string(rhs_scale));
        }
    }
}

void field_02_longitudinal_projection() {
    Fixture f;
    const std::size_t modes = static_cast<std::size_t>(2 * f.p.K + 1);
    const Vec x = deterministic_vector(modes * f.op.Ne, 17, true);
    const Vec longitudinal = fomhd::longitudinal(x, f.c);
    const Vec transverse = fomhd::subtract(x, longitudinal);
    for (int mode = -f.p.K; mode <= f.p.K; ++mode) {
        const std::size_t k =
            static_cast<std::size_t>(mode + f.p.K);
        const Vec residual = f.compute.mv(
            f.op.DJ[k], fomhd::slice(transverse, k * f.op.Ne, f.op.Ne));
        require(fomhd::max_abs(residual) < 2.0e-9,
                "longitudinal projection left divergence");
    }
    require(fomhd::norm(longitudinal) > 1.0,
            "longitudinal projection fixture is trivial");
}

void field_03_complex_AE_properties() {
    Fixture f;
    const double density = f.p.ref.ni;
    State state = sampled_state(
        f,
        [density](double, double, double phi) {
            return density * (1.0 + 0.04 * std::cos(phi));
        },
        [](double, double, double) {
            return std::array<double, 3>{0.0, 0.0, 0.0};
        },
        1.0, 1.0, 1.0);
    StageResult stage;
    prepare_fluid_stage(f, state, stage);
    stage.fJ0.assign(
        static_cast<std::size_t>(2 * f.p.K + 1) * f.op.Ne, Complex{});
    const Vec rho = charge_density(f, stage);
    stage.cache = fomhd::electrostatic(rho, f.tag, f.c);
    fomhd::transverse(stage, f.c);

    const std::size_t size =
        static_cast<std::size_t>(2 * f.p.K + 1) * f.op.Ne;
    require(stage.AE.rows == size && stage.AE.cols == size,
            "AE was not materialized as full M*Ne square");
    const Vec x = deterministic_vector(size, 21, true);
    const Vec y = deterministic_vector(size, 29, true);
    const Complex alpha{0.4, -0.7};
    const Complex beta{-0.2, 0.3};
    const Vec linearity_residual = fomhd::subtract(
        f.compute.mv(stage.AE, fomhd::add(fomhd::scaled(x, alpha),
                                          fomhd::scaled(y, beta))),
        fomhd::add(fomhd::scaled(f.compute.mv(stage.AE, x), alpha),
                   fomhd::scaled(f.compute.mv(stage.AE, y), beta)));
    require(fomhd::norm(linearity_residual) <
                2.0e-11 * (std::abs(alpha)*fomhd::norm(f.compute.mv(stage.AE,x))
                          + std::abs(beta)*fomhd::norm(f.compute.mv(stage.AE,y))),
            "AE is not complex-linear");

    Vec response_action(size);
    for(int a=0;a<2;++a) {
        auto force=fomhd::electric_response(stage.fluid[a],a,x,f.c);
        auto edge=fomhd::sample_apply(f.op.I,force,f.p.Nphi,f.compute);
        response_action=fomhd::add(response_action,fomhd::scaled(fomhd::forward(edge,f.op.Ne,f.c),
                                  f.p.species[a].charge/f.p.species[a].mass));
    }
    require(fomhd::norm(fomhd::subtract(f.compute.mv(stage.AE,x),response_action))
            < 2e-12*fomhd::norm(response_action),"materialization must match complex physical action");
    const fomhd::RealVec weights =
        repeat_real_weights(
            f.op.M1,
            static_cast<std::size_t>(2 * f.p.K + 1));
    const Complex left =
        weighted_dot(x, f.compute.mv(stage.AE, y), weights);
    const Complex right =
        weighted_dot(f.compute.mv(stage.AE, x), y, weights);
    near_complex(left, right, 2.0e-10 *
                     std::max(1.0, std::abs(left)),
                 "AE is not weighted self-adjoint");
}

void field_04_darwin_fluid_consistency() {
    Fixture f;
    State state = varying_gradient_state(f);
    make_wall_compatible(f, state);
    StageResult stage;
    prepare_fluid_stage(f, state, stage);
    stage.fJ0 = no_electric_current(f, stage);
    const Vec rho = charge_density(f, stage);
    stage.cache = fomhd::electrostatic(rho, f.tag, f.c);
    fomhd::transverse(stage, f.c);
    for (int species = 0; species < 2; ++species) {
        fomhd::finish_fluid(stage, species, f.c);
    }

    const Vec response = fomhd::add(
        stage.fJ0, f.compute.mv(stage.AE, stage.cache.E));
    Vec actual(
        static_cast<std::size_t>(2 * f.p.K + 1) * f.op.Ne);
    for (int species = 0; species < 2; ++species) {
        actual = fomhd::add(
            actual,
            fomhd::scaled(
                fomhd::sample_apply(f.op.I,stage.rhs.fluid[static_cast<std::size_t>(species)].mom,
                                   2*f.p.K+1,f.compute),
                f.p.species[static_cast<std::size_t>(species)].charge /
                    f.p.species[static_cast<std::size_t>(species)].mass));
    }
    const Vec current_residual = fomhd::subtract(actual, response);
    require(fomhd::norm(current_residual) <
                2.0e-10 * std::max(1.0, fomhd::norm(response)),
            "Jdot(actual) differs from fJ0 + AE(E)");

    for (int species = 0; species < 2; ++species) {
        const Vec wall_rows = fomhd::sample_apply(
            f.op.T_full, stage.terms[static_cast<std::size_t>(species)].force,
            f.p.Nphi, f.compute);
        require(fomhd::max_abs(wall_rows) < 2.0e-10,
                "finish_fluid left a wall-normal force");
        require(fomhd::max_abs(
                    stage.terms[static_cast<std::size_t>(species)]
                        .multiplier_residual) < 2.0e-9,
                "wall multiplier equation residual");
    }

    const Vec full_divergence = f.compute.mv(
        f.op.Bc_full, stage.cache.ET);
    require(fomhd::max_abs(full_divergence) < 2.0e-9,
            "ET violates a dropped transverse constraint row");
    for (int mode = -f.p.K; mode <= f.p.K; ++mode) {
        const std::size_t k =
            static_cast<std::size_t>(mode + f.p.K);
        const Vec bdot =
            fomhd::slice(stage.rhs.B, k * f.op.MB.size(), f.op.MB.size());
        const Vec divergence =
            f.compute.mv(f.op.DB[k], bdot);
        const double ct_roundoff=64*std::numeric_limits<double>::epsilon()*f.op.DB[k].cols
            *fomhd::max_abs(f.op.DB[k])*fomhd::max_abs(bdot);
        require(fomhd::max_abs(divergence) <= ct_roundoff,
                "Bdot from ET violates the matrix-product rounding bound");
    }
    require(stage.residuals.contains("transverse_kkt") &&
                stage.residuals.contains("transverse_dynamic") &&
                stage.residuals.contains("transverse_divergence_full"),
            "Darwin raw residual maps are missing");
}

using Test = std::pair<std::string, std::function<void()>>;

const std::vector<Test>& registry() {
    static const std::vector<Test> tests{
        {"FLUID_01_pressure_tensor_and_domain",
         fluid_01_pressure_tensor_and_domain},
        {"FLUID_02_cylindrical_gradient", fluid_02_cylindrical_gradient},
        {"FLUID_03_weighted_adjoint", fluid_03_weighted_adjoint},
        {"FLUID_04_pressure_convection_pairing",
         fluid_04_pressure_convection_pairing},
        {"FLUID_05_electric_magnetic_work", fluid_05_electric_magnetic_work},
        {"FLUID_06_sample_layout_and_magnetic_strength", fluid_06_sample_layout_and_magnetic_strength},
        {"FIELD_01_poisson_all_rows", field_01_poisson_all_rows},
        {"FIELD_02_longitudinal_projection", field_02_longitudinal_projection},
        {"FIELD_03_complex_AE_properties", field_03_complex_AE_properties},
        {"FIELD_04_darwin_fluid_consistency",
         field_04_darwin_fluid_consistency},
    };
    return tests;
}

}  // namespace

int main(int argc, char** argv) {
    const auto& tests = registry();
    if (argc == 1) {
        std::size_t failures = 0;
        for (const auto& [name, test] : tests) {
            try {
                test();
                std::cout << "PASS " << name << '\n';
            } catch (const std::exception& error) {
                ++failures;
                std::cerr << "FAIL " << name << ": " << error.what() << '\n';
            }
        }
        std::cout << "SUMMARY " << tests.size() - failures << "/"
                  << tests.size() << " passed\n";
        return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    for (const auto& [name, test] : tests) {
        if (name == argv[1]) {
            try {
                test();
                std::cout << "PASS " << name << '\n';
                return EXIT_SUCCESS;
            } catch (const std::exception& error) {
                std::cerr << "FAIL " << name << ": " << error.what() << '\n';
                return EXIT_FAILURE;
            }
        }
    }
    std::cerr << "unknown test: " << argv[1] << '\n';
    return EXIT_FAILURE;
}
