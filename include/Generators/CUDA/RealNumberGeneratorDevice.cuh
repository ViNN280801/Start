#ifndef REALNUMBERGENERATORDEVICE_CUH
#define REALNUMBERGENERATORDEVICE_CUH

#ifdef USE_CUDA

#include <curand_kernel.h>

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
    __device__ explicit RealNumberGeneratorDevice(curandState_t &state) : m_state(&state) {}

    /**
     * @brief Generates a random real number in the specified range [from, to].
     *
     * @param from Lower bound of the range.
     * @param to Upper bound of the range.
     * @return A random double in the range [from, to].
     */
    __device__ double generate(double from, double to)
    {
        if (from == to)
            return from;
        if (from > to)
        {
            double temp = from;
            from = to;
            to = temp;
        }

        // Generate a random number in [0, 1) and scale to [from, to].
        double randomValue = curand_uniform_double(m_state);
        return from + randomValue * (to - from);
    }
};

#endif // !USE_CUDA

#endif // !REALNUMBERGENERATORDEVICE_CUH
