#ifndef REALNUMBERGENERATOR_HPP
#define REALNUMBERGENERATOR_HPP

#include <random>
#include <vector>

/**
 * @brief Generator for real numbers and their sequences.
 *
 * This class generates random real numbers and sequences of real numbers
 * using the Mersenne Twister engine. It supports generating random numbers
 * within a specified interval and allows the configuration of upper and lower
 * bounds for the generated numbers.
 */
class RealNumberGenerator final
{
private:
    double m_from{kdefault_min_value}, // Lower bound.
        m_to{kdefault_max_value};      // Upper bound.
    std::random_device m_rdm_dev;      // Special engine that requires a piece of attached hardware to PC
                                       // that generates truly non-deterministic random numbers.
    std::mt19937 m_engine;             // Mersenne Twister engine:
                                       // Fastest.
                                       // T = 2^19937 - 1 (The period of the pseudorandom sequence).
                                       // Memory usage ~2.5kB.

    static constexpr double kdefault_min_value{0.0}; // Default value of the lower bound.
    static constexpr double kdefault_max_value{1.0}; // Default value of the upper bound.

public:
    /**
     * @brief Constructs a RealNumberGenerator with default bounds (0.0, 1.0).
     *
     * Initializes the random number generator with a non-deterministic seed.
     */
    RealNumberGenerator();

    /**
     * @brief Constructs a RealNumberGenerator with specified bounds.
     *
     * Initializes the random number generator with the provided bounds [from, to].
     *
     * @param from Lower bound for generated numbers.
     * @param to Upper bound for generated numbers.
     */
    RealNumberGenerator(double from, double to);

    /**
     * @brief Destructor for RealNumberGenerator.
     *
     * Default destructor for RealNumberGenerator. It performs no special cleanup as the class
     * does not manage any resources that require manual handling.
     */
    ~RealNumberGenerator() = default;

    /**
     * @brief Generates a random real number in the configured interval [m_from, m_to].
     *
     * @return A random real number within the bounds [m_from, m_to].
     *
     * @note Ignoring this function's return value means the random number generated is discarded.
     */
    [[nodiscard("Random number should not be discarded")]] double operator()();

    /**
     * @brief Generates a random real number in the specified interval [from, to].
     *
     * @param from Lower bound for the generated number.
     * @param to Upper bound for the generated number.
     * @return A random real number within the bounds [from, to].
     *
     * @note Ignoring this function's return value means the random number generated is discarded.
     */
    [[nodiscard("Random number should not be discarded")]] double operator()(double from, double to);

    /**
     * @brief Generates a random real number in the specified interval [from, to].
     *
     * @param from Lower bound for the generated number.
     * @param to Upper bound for the generated number.
     * @return A random real number within the bounds [from, to].
     *
     * @note Ignoring this function's return value means the random number generated is discarded.
     */
    [[nodiscard("Random number should not be discarded")]] double get_double(double from, double to);

    /* === Setter methods for data members === */
    /**
     * @brief Sets the lower bound for random number generation.
     *
     * @param val The lower bound value.
     */
    void set_lower_bound(double);

    /**
     * @brief Sets the upper bound for random number generation.
     *
     * @param val The upper bound value.
     */
    void set_upper_bound(double);

    /**
     * @brief Sets the lower and upper bounds for random number generation.
     *
     * @param from Lower bound for the generated number.
     * @param to Upper bound for the generated number.
     */
    void set(double from, double to);

    /**
     * @brief Generates sequence of real numbers in specified interval.
     * @param count Count of numbers to generate.
     */
    std::vector<double> get_sequence(size_t count, double from = 0.0, double to = 1.0);
};

#endif // !REALNUMBERGENERATOR_HPP
