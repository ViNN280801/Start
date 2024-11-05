#ifndef PARTICLEDEVICE_HPP
#define PARTICLEDEVICE_HPP

#ifdef USE_CUDA
#include <stddef.h>

struct ParticleDevice
{
    size_t id;
    int type; // Using `int` for `ParticleType`
    double x, y, z;
    double vx, vy, vz;
    double energy;
};
#endif // !USE_CUDA

#endif // !PARTICLEDEVICE_HPP
