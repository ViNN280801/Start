#ifndef PARTICLEDYNAMICSPROCESSORCUDA_CUH
#define PARTICLEDYNAMICSPROCESSORCUDA_CUH

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <mutex>
#include <shared_mutex>
#include <string_view>

#include "FiniteElementMethod/GSMAssembler.hpp"
#include "Geometry/CubicGrid.hpp"
#include "Geometry/GeometryTypes.hpp"
#include "Particle/CUDA/ParticleDevice.cuh"
#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"

/**
 * @brief Structure for holding CUDA kernel parameters
 */
struct ParticleDynamicsKernelParams
{
    double timeMoment;
    double timeStep;
    double gasConcentration;
    int scatteringModelType;
    int gasType;
};

/**
 * @class ParticleDynamicsProcessorCUDA
 * @brief CUDA implementation for parallel processing of particle dynamics
 * 
 * This class provides CUDA-based implementations of particle dynamics calculations 
 * including electromagnetic forces, particle movement tracking, gas collisions,
 * and surface collisions.
 */
class ParticleDynamicsProcessorCUDA
{
private:
    /**
     * @brief Prepares particles for CUDA processing by copying to device memory.
     * 
     * @param particles Vector of particles to process
     * @return ParticleDeviceArray containing device-side particles
     */
    static ParticleDeviceArray prepareParticlesForGPU(ParticleVector &particles);

    /**
     * @brief Copies processed particles back from device to host memory.
     * 
     * @param deviceParticles The processed particles on the device
     * @param hostParticles The original particles vector to update
     */
    static void updateHostParticles(ParticleDeviceArray &deviceParticles, ParticleVector &hostParticles);

    /**
     * @brief Converts scattering model string to integer representation for CUDA kernel
     * 
     * @param scatteringModel String view of the scattering model name
     * @return Integer representation of the model (0=HS, 1=VHS, 2=VSS)
     */
    static constexpr int convertScatteringModelToInt(std::string_view scatteringModel) noexcept;

    /**
     * @brief Converts gas name to particle type integer for CUDA kernel
     * 
     * @param gasName String view of the gas name
     * @return Integer value of the ParticleType enum
     */
    static constexpr int convertGasNameToParticleType(std::string_view gasName) noexcept;

public:
    /**
     * @brief Processes particles in parallel using CUDA
     * 
     * @param timeMoment Current simulation time
     * @param timeStep Time step for integration
     * @param particles Vector of particles to process
     * @param cubicGrid Pointer to the simulation grid
     * @param gsmAssembler Pointer to the finite element assembler
     * @param surfaceMesh Reference to the surface mesh for collision detection
     * @param particleTracker Map for tracking particle movements
     * @param settledParticlesIds Set of settled particle IDs
     * @param sh_mutex_settledParticlesCounterMap Shared mutex for particle counter
     * @param mutex_particlesMovementMap Mutex for movement tracking
     * @param particleMovementMap Map storing particle movement history
     * @param stopSubject Reference to the stop condition handler
     * @param scatteringModel The scattering model used for gas collisions
     * @param gasName The type of gas used for collisions
     * @param gasConcentration The concentration of gas molecules
     */
    static void process(double timeMoment,
                        double timeStep,
                        ParticleVector &particles,
                        std::shared_ptr<CubicGrid> cubicGrid,
                        std::shared_ptr<GSMAssembler> gsmAssembler,
                        SurfaceMesh &surfaceMesh,
                        ParticleTrackerMap &particleTracker,
                        ParticlesIDSet &settledParticlesIds,
                        std::shared_mutex &sh_mutex_settledParticlesCounterMap,
                        std::mutex &mutex_particlesMovementMap,
                        ParticleMovementMap &particleMovementMap,
                        StopSubject &stopSubject,
                        std::string_view scatteringModel,
                        std::string_view gasName,
                        double gasConcentration);
};

#endif // USE_CUDA

#endif // PARTICLEDYNAMICSPROCESSORCUDA_CUH 