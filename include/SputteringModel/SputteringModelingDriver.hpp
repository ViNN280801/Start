#ifndef SPUTTERINGMODELINGDRIVER_HPP
#define SPUTTERINGMODELINGDRIVER_HPP

#include <barrier>
#include <future>
#include <mutex>
#include <shared_mutex>

#include "Geometry/Mesh.hpp"
#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"

class SputteringModelingDriver : public StopSubject
{
private:
    std::string m_mesh_filename;                      ///< Filename of the mesh.
    static std::mutex m_particlesMovementMutex;       ///< Mutex for synchronizing access to the trajectories of particles.
    static std::shared_mutex m_settledParticlesMutex; ///< Mutex for synchronizing access to settled particle IDs.
    static std::atomic_flag m_stop_processing;        ///< Flag-checker for condition (counter >= size of particles).
    std::shared_ptr<StopFlagObserver> m_stopObserver; ///< Observer for managing stop requests.
    SurfaceMesh m_surfaceMesh;                        ///< Surface mesh that contains cell data for all the cells and AABB tree for the surface mesh.
    ParticleMovementMap m_particlesMovement;          ///< Map to store all the particle movements: (Particle ID | All positions).
    ParticlesIDSet m_settledParticlesIds;             ///< Set of the particle IDs that are been settled (need to avoid checking already settled particles).

    void _initializeObservers();
    void _ginitialize();
    void _updateSurfaceMesh();
    void _gfinalize();

public:
    SputteringModelingDriver(std::string_view mesh_filename, std::string_view physicalGroupName);
    ~SputteringModelingDriver();

    void startModeling(double simtime, double timeStep, unsigned int numThreads,
                       ParticleVector &particles, std::string_view scatteringModel,
                       std::string_view gasName, double pressure, double temperature);

    TriangleCellMap const &getCellsWithSettledParticles() const { return m_surfaceMesh.getTriangleCellMap(); }
};

#endif // !SPUTTERINGMODELINGDRIVER_HPP
