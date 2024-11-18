#ifndef PARTICEMEMORYCONVERTER_CUH
#define PARTICEMEMORYCONVERTER_CUH

#ifdef USE_CUDA
#include "Particle/Particle.hpp"
#include "Particle/ParticleDevice.cuh"

inline ParticleDevice_t ParticleToDevice(Particle &p)
{
    ParticleDevice_t pd;
    pd.id = p.getId();
    pd.type = static_cast<int>(p.getType());
    pd.x = p.getX();
    pd.y = p.getY();
    pd.z = p.getZ();
    pd.vx = p.getVx();
    pd.vy = p.getVy();
    pd.vz = p.getVz();
    pd.energy = p.getEnergy_eV();
    return pd;
}

inline Particle DeviceToParticle(ParticleDevice_t const &pd) { return Particle(pd); }
#endif // !USE_CUDA

#endif // !PARTICEMEMORYCONVERTER_HPP
