#ifndef PARTICLEKERNELHELPERS_CUH
#define PARTICLEKERNELHELPERS_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>

#include "Geometry/CUDA/GeometryDeviceTypes.cuh"
#include "Particle/CUDA/ParticleDevice.cuh"
#include "Utilities/Constants.hpp"
#include "Utilities/PreprocessorUtils.hpp"

/**
 * @namespace cuda_kernels
 * @brief Contains CUDA kernel functions for particle dynamics processing
 */
namespace cuda_kernels
{
    /**
     * @brief Updates particle position based on its velocity and time step
     * 
     * @param particle Particle to update
     * @param dt Time step
     */
    START_CUDA_DEVICE void updateParticlePosition(ParticleDevice_t &particle, double dt) noexcept
    {
        // Update position components based on velocity
        particle.x += particle.vx * dt;
        particle.y += particle.vy * dt;
        particle.z += particle.vz * dt;
    }

    /**
     * @brief Calculates the square of the distance between two 3D points
     */
    START_CUDA_DEVICE double distanceSquared(double x1, double y1, double z1, double x2, double y2, double z2) noexcept
    {
        double dx = x2 - x1;
        double dy = y2 - y1;
        double dz = z2 - z1;
        return dx * dx + dy * dy + dz * dz;
    }

    /**
     * @brief Checks if a value is between two bounds (inclusive)
     */
    START_CUDA_DEVICE STARTCONSTEXPR bool isBetween(double value, double lower, double upper) noexcept
    {
        return value >= lower && value <= upper;
    }

    /**
     * @brief Performs a simple collision with gas molecules
     * 
     * This is a simplified implementation to be used when full collision models
     * are not available on the GPU. For more accurate results, implement the 
     * HS, VHS, and VSS models as device functions.
     * 
     * @param particle Particle to update
     * @param gasType Type of gas
     * @param gasConcentration Concentration of gas
     * @param dt Time step
     * @param scatteringModelType Collision model type (0=HS, 1=VHS, 2=VSS)
     */
    START_CUDA_DEVICE void collideWithGas(ParticleDevice_t &particle, 
                                       int gasType,
                                       double gasConcentration,
                                       double dt,
                                       int scatteringModelType) noexcept
    {
        // We will use a simple collision model for now
        // This should be replaced with proper implementations of HS, VHS, and VSS models
        
        // Calculate velocity magnitude
        double v_mag = sqrt(particle.vx * particle.vx + particle.vy * particle.vy + particle.vz * particle.vz);
        
        if (v_mag < 1e-12) {
            return; // Skip if velocity is very small
        }
        
        // Damping factor based on gas concentration and time step
        double damping = 0.0;
        
        // Very basic scaling of the damping based on the gas concentration
        // This is a placeholder and should be replaced with proper physics
        if (gasConcentration > 0.0) {
            // The higher the concentration, the stronger the damping
            damping = 0.01 * gasConcentration * dt;
            
            // Limit maximum damping
            if (damping > 0.1) damping = 0.1;
            
            // Apply damping to velocity components
            particle.vx *= (1.0 - damping);
            particle.vy *= (1.0 - damping);
            particle.vz *= (1.0 - damping);
            
            // Update energy based on new velocity
            double new_v_mag = sqrt(particle.vx * particle.vx + particle.vy * particle.vy + particle.vz * particle.vz);
            double energy_ratio = (new_v_mag * new_v_mag) / (v_mag * v_mag);
            particle.energy *= energy_ratio;
        }
    }
    
    /**
     * @brief Checks if a particle is inside a tetrahedron
     * 
     * @param particle The particle to check
     * @param tetra The tetrahedron
     * @return True if the particle is inside the tetrahedron
     */
    START_CUDA_DEVICE bool isParticleInTetrahedron(ParticleDevice_t const& particle, DeviceTetrahedron const& tetra) noexcept
    {
        DevicePoint point(particle.x, particle.y, particle.z);
        return tetra.contains_point(point);
    }
}

#endif // USE_CUDA

#endif // PARTICLEKERNELHELPERS_CUH 