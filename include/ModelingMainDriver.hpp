#ifndef MODELINGMAINDRIVER_HPP
#define MODELINGMAINDRIVER_HPP

#include <barrier>
#include <future>
#include <mutex>
#include <shared_mutex>

#include "FiniteElementMethod/MatrixEquationSolver.hpp"
#include "Geometry/CubicGrid.hpp"
#include "Geometry/Mesh.hpp"
#include "Particle/Particle.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/StopModelingObserver.hpp"
#include "SessionManagement/GmshSessionManager.hpp"
#include "Utilities/ConfigParser.hpp"

/**
 * @brief This class represents the main driver that manages the PIC part and surface collision tracker.
 *
 * Main algorithm:
 * 1. Get the mesh from the mesh file. Fill basis gradient functions \( \nabla \varphi_i \) for each node.
 * 2. Construct GSM (Global Stiffness Matrix). Set Boundary Conditions (BC) to the GSM and vector \(\mathbf{b}\).
 *    Solve \( A \mathbf{x} = \mathbf{b} \), where \( A = \text{GSM} \).
 * 3. Spawn particles from the sources specified in the configuration file.
 * 4. Track particles in the mesh.
 * 5. Collect node charge densities (\( \rho_i \)) for each node at every time step.
 * 6. Pass \( \rho_i \) to the vector \(\mathbf{b}\) to solve \( A \mathbf{x} = \mathbf{b} \) (Dirichlet Boundary Conditions).
 * 7. Solve the equation \( A \mathbf{x} = \mathbf{b} \) at every time step.
 * 8. Calculate electric potential (\( \varphi_i \)) of each node from the resulting \(\mathbf{x}\) vector in \( A \mathbf{x} = \mathbf{b} \).
 * 9. Calculate the electric field \( \mathbf{E} = (E_x, E_y, E_z) \) of each mesh cell based on \( \varphi_i \) and \( \nabla \varphi_i \):
 *    \f[
 *    \mathbf{E}_{\text{cell}} = -\frac{1}{6V} \sum_i (\varphi_i \cdot \nabla \varphi_i).
 *    \f]
 * 10. Electro-magnetic pushing of the particles (update velocity):
 *     10.1. Update position.
 *     10.2. Check if the particle is inside the mesh. If not, classify the particle as a settled particle and stop tracking it.
 * 11. Perform gas collision statistical checking (HS/VHS/VSS). If a collision occurs, update velocity:
 *     11.1. Update position.
 *     11.2. Check if the particle is inside the mesh.
 * 12. Check if the particle is settled. If so, increment the count of settled particles on the relevant triangle in the mesh.
 * 13. Save all collected particles in triangles (surface mesh) to an \texttt{.hdf5} file.
 * 14. Optionally, save all trajectories of the particles to create an animation.
 */
class ModelingMainDriver : public StopSubject
{
private:
    std::string m_config_filename;                    ///< Filename of the configuration to parse it.
    static std::mutex m_PICTrackerMutex;              ///< Mutex for synchronizing access to the particles in tetrahedrons.
    static std::mutex m_nodeChargeDensityMapMutex;    ///< Mutex for synchronizing access to the charge densities in nodes.
    static std::mutex m_particlesMovementMutex;       ///< Mutex for synchronizing access to the trajectories of particles.
    static std::shared_mutex m_settledParticlesMutex; ///< Mutex for synchronizing access to settled particle IDs.
    static std::atomic_flag m_stop_processing;        ///< Flag-checker for condition (counter >= size of particles).
    std::shared_ptr<StopFlagObserver> m_stopObserver; ///< Observer for managing stop requests.

    /* All the neccessary data members from the mesh. */
    GmshSessionManager m_gmshSessionManager; ///< Object of the volume creator that is RAII object that initializes and finalizes GMSH. Needed to initialize all necessary objects from the mesh.
    ConfigParser m_config;                   ///< `ConfigParser` object to get all the simulation physical parameters.
    SurfaceMesh m_surfaceMesh;               ///< Surface mesh containing triangle cell map, triangles vector and AABB tree for the collision finding optimizations.

    /* All the neccessary data members for the simulation. */
    ParticleVector m_particles;           ///< Projective particles.
    double m_gasConcentration;            ///< Gas concentration. Needed to use colide projectives with gas mechanism.
    ParticlesIDSet m_settledParticlesIds; ///< Set of the particle IDs that are been settled (need to avoid checking already settled particles).

    ParticleMovementMap m_particlesMovement; ///< Map to store all the particle movements: (Particle ID | All positions).
    ParticleTrackerMap m_particleTracker;    ///< Global particle in cell tracker (Time moment: (Tetrahedron ID | Particles inside)).

    /// @brief Initializes observer for the stopping modeling process when all particles are settled.
    void _initializeObservers();

    /* Initializers for all the necessary objects. */
    /**
     * @brief Initializes particles based on the configuration settings.
     * @details This method generates particles either from point sources or surface sources,
     *          depending on the configuration file. The generated particles are appended
     *          to the existing particle list. If no particles are generated, an exception is thrown.
     * @throws std::runtime_error if no particles are generated, which may indicate a misconfiguration.
     * @note This method requires a valid configuration file specifying particle sources.
     */
    void _initializeParticles();
    /* =========================================== */

    /// @brief Global initializator. Uses all the initializers above.
    void _ginitialize();

    /* Finalizers for all the necessary objects. */
    /**
     * @brief Saves the particle movements to a JSON file.
     *
     * This function saves the contents of m_particlesMovement to a JSON file named "particles_movements.json".
     * It handles exceptions and provides a warning message if the map is empty.
     */
    void _saveParticleMovements() const;

    /// @brief Using HDF5Handler to update the mesh according to the settled particles.
    void _updateSurfaceMesh();
    /* =========================================== */

    /// @brief Global finalizator. Updates
    void _gfinalize();

public:
    /**
     * @brief Constructs the ModelingMainDriver object and initializes it using a configuration file.
     *
     * This constructor initializes the driver for the modeling process using the provided configuration
     * filename. It sets up necessary components based on the configuration.
     *
     * @param config_filename A string view of the path to the configuration file.
     */
    ModelingMainDriver(std::string_view config_filename);

    /**
     * @brief Dtor. Finalizes all the processes.
     *
     * @details Updates surface mesh (updates counter of the settled particles in .hdf5 file)
     *          and saves trajectories of the particles movements to the json.
     *          Has checkings after writing a .json file with all the particle movements.
     */
    ~ModelingMainDriver();

    /**
     * @brief Starts the modeling process.
     *
     * This function initiates the entire modeling process, which typically involves setting up
     * the simulation, processing particles, solving equations, and updating the system at each time step.
     */
    void startModeling();
};

#endif // !MODELINGMAINDRIVER_HPP
