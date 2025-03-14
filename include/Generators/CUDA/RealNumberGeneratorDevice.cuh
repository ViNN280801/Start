#ifndef REALNUMBERGENERATORDEVICE_CUH
#define REALNUMBERGENERATORDEVICE_CUH

#ifdef USE_CUDA

#include <curand_kernel.h>
#include <random>

#include "Utilities/PreprocessorUtils.hpp"

/**
 * @class RealNumberGeneratorDevice
 * @brief A CUDA-compatible random number generator for generating real numbers in a specified range.
 *
 * This class abstracts the management of `curandState_t` and provides a simple interface for generating random numbers
 * on the device. The random state is automatically initialized based on the thread ID and a provided seed.
 */
class RealNumberGeneratorDevice
{
private:
    curandState_t *m_state;

public:
    /**
     * @brief Constructs the random number generator for the device.
     *
     * @param state Reference to an initialized curand state.
     */
    START_CUDA_DEVICE explicit RealNumberGeneratorDevice(curandState_t &state) : m_state(&state) {}

    /**
     * @brief Generates a random real number in the specified range [from, to].
     *
     * @param from Lower bound of the range.
     * @param to Upper bound of the range.
     * @return A random double in the range [from, to].
     */
    START_CUDA_HOST_DEVICE double generate(double from, double to)
    {
        if (from == to)
            return from;
        if (from > to)
        {
            double temp = from;
            from = to;
            to = temp;
        }

#ifdef __CUDA_ARCH__
        // Generate a random number in [0, 1) and scale to [from, to].
        double randomValue = curand_uniform_double(m_state);
        return from + randomValue * (to - from);
#else
        // Host implementation - use standard library random
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> distribution(from, to);
        return distribution(gen);
#endif
    }
};

#endif // !USE_CUDA
#endif // !REALNUMBERGENERATORDEVICE_CUH
