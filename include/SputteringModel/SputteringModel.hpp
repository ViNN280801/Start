#ifndef SPUTTERINGMODEL_HPP
#define SPUTTERINGMODEL_HPP

#include <barrier>
#include <future>
#include <mutex>
#include <shared_mutex>

#include "Geometry/Mesh/Surface/SurfaceMesh.hpp"
#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/PICTypes.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"
#include "SessionManagement/GmshSessionManager.hpp"

class SputteringModel : public StopSubject
{
private:
    GmshSessionManager gsm;                           ///< RAII object to manage gmsh session state.
    std::string m_mesh_filename;                      ///< Filename of the mesh.
    static std::mutex m_particlesMovementMutex;       ///< Mutex for synchronizing access to the trajectories of particles.
    static std::shared_mutex m_settledParticlesMutex; ///< Mutex for synchronizing access to settled particle IDs.
    static std::atomic_flag m_stop_processing;        ///< Flag-checker for condition (counter >= size of particles).
    std::shared_ptr<StopFlagObserver> m_stopObserver; ///< Observer for managing stop requests.
    SurfaceMesh m_surfaceMesh;                        ///< Surface mesh that contains cell data for all the cells and AABB tree for the surface mesh.
    ParticleMovementMap m_particlesMovement;          ///< Map to store all the particle movements: (Particle ID | All positions).
    ParticlesIDSet m_settledParticlesIds;             ///< Set of the particle IDs that are been settled (need to avoid checking already settled particles).
    int m_particleWeight;                             ///< Weight of the modeling particle.

    void _initializeObservers();
    void _ginitialize();
    void _updateSurfaceMesh();

    void _distributeSettledParticles();
    void _writeHistogramToFile();

    void _gfinalize();

public:
    SputteringModel(std::string_view mesh_filename, std::string_view physicalGroupName);
    ~SputteringModel();

    void startModeling(unsigned int numThreads);

    TriangleCellMap const &getCellsWithSettledParticles() const { return m_surfaceMesh.getTriangleCellMap(); }
};

#endif // !SPUTTERINGMODEL_HPP
