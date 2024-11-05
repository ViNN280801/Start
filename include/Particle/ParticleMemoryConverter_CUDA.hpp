#ifndef PARTICEMEMORYCONVERTER_HPP
#define PARTICEMEMORYCONVERTER_HPP

#ifdef USE_CUDA
#include "Particle/Particle.hpp"
#include "Particle/ParticleDevice_CUDA.hpp"

inline ParticleDevice ParticleToDevice(Particle &p)
{
    ParticleDevice pd;
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

inline Particle DeviceToParticle(ParticleDevice const &pd)
{
    Particle p(static_cast<ParticleType>(pd.type),
               pd.x, pd.y, pd.z,
               pd.vx, pd.vy, pd.vz);
    return p;
}
#endif // !USE_CUDA

#endif // !PARTICEMEMORYCONVERTER_HPP
