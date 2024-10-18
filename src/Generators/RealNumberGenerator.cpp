#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>

#if __cplusplus >= 202002L
#include <ranges>
#endif

#include "Generators/RealNumberGenerator.hpp"

// `.entropy()` returns 0.0 if random device using a software-based pseudorandom generator
RealNumberGenerator::RealNumberGenerator() : m_engine(m_rdm_dev.entropy() ? m_rdm_dev() : time(nullptr)) {}

RealNumberGenerator::RealNumberGenerator(double from, double to) : m_from(from), m_to(to) {}

double RealNumberGenerator::operator()() { return get_double(m_from, m_to); }

double RealNumberGenerator::operator()(double from, double to) { return get_double(from, to); }

double RealNumberGenerator::get_double(double from, double to) { return std::uniform_real_distribution(from, to)(m_engine); }

void RealNumberGenerator::set_lower_bound(double val) { m_from = val; }

void RealNumberGenerator::set_upper_bound(double val) { m_to = val; }

void RealNumberGenerator::set(double from, double to)
{
    m_from = from;
    m_to = to;
}

std::vector<double> RealNumberGenerator::get_sequence(size_t count, double from, double to)
{
    if (count == 0ul)
        return {};

    auto dist{std::uniform_real_distribution<double>(from, to)};
    auto gen{std::bind(dist, m_engine)};

    std::vector<double> sequence(count);

#if __cplusplus >= 202002L
    std::ranges::generate(sequence, gen);
#else
    std::generate(sequence.begin(), sequence.end(), gen);
#endif

    return sequence;
}
