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
class ModelingMainDriver final
{
private:
    std::string m_config_filename; ///< Filename of the configuration to parse it.

    static constexpr short const kdefault_max_numparticles_to_anim{5'000}; ///< Maximal count of particles to do animation.

    static std::mutex m_PICTrackerMutex;              ///< Mutex for synchronizing access to the particles in tetrahedrons.
    static std::mutex m_nodeChargeDensityMapMutex;    ///< Mutex for synchronizing access to the charge densities in nodes.
    static std::mutex m_particlesMovementMutex;       ///< Mutex for synchronizing access to the trajectories of particles.
    static std::shared_mutex m_settledParticlesMutex; ///< Mutex for synchronizing access to settled particle IDs.

    /* All the neccessary data members from the mesh. */
    MeshTriangleParamVector _triangleMesh;   ///< Triangle mesh params acquired from the mesh file. Surface mesh.
    TriangleVector _triangles;               ///< Triangles extracted from the triangle mesh params `_triangleMesh` (surface mesh). Need to initialize AABB tree.
    AABB_Tree_Triangle _surfaceMeshAABBtree; ///< AABB tree for the surface mesh to effectively detect collisions with surface.
    GmshSessionManager _gmshSessionManager;  ///< Object of the volume creator that is RAII object that initializes and finalizes GMSH. Needed to initialize all necessary objects from the mesh.

    /* All the neccessary data members for the simulation. */
    ParticleVector m_particles;                           ///< Projective particles.
    double _gasConcentration;                             ///< Gas concentration. Needed to use colide projectives with gas mechanism.
    std::set<size_t> _settledParticlesIds;                ///< Set of the particle IDs that are been settled (need to avoid checking already settled particles).
    std::map<size_t, size_t> _settledParticlesCounterMap; ///< Map to handle settled particles: (Triangle ID | Counter of settled particle in this triangle).

    ConfigParser m_config;                                                ///< `ConfigParser` object to get all the simulation physical paramters.
    std::map<size_t, std::vector<Point>> m_particlesMovement;             ///< Map to store all the particle movements: (Particle ID | All positions).
    std::map<double, std::map<size_t, ParticleVector>> m_particleTracker; ///< Global particle in cell tracker (Time moment: (Tetrahedron ID | Particles inside)).

    /**
     * @brief Broadcasts the triangle mesh data from the root rank (rank 0) to all other ranks.
     * @details This method ensures that all processes in the MPI communicator receive the
     *          triangle mesh data from rank 0. The data includes triangle IDs, vertex coordinates,
     *          surface areas, and counters. On rank 0, the data is prepared and broadcast to
     *          all other ranks, which then reconstruct the mesh locally.
     *
     * @note This function uses MPI to perform the broadcasting. The triangle mesh on non-root
     *       ranks is cleared and reconstructed based on the broadcasted data.
     */
    void _broadcastTriangleMesh();

    /* Initializers for all the necessary objects. */
    /**
     * @brief Initializes the surface mesh by loading it from a file on the root rank and broadcasting it.
     * @details This method loads the surface mesh data on rank 0 using a mesh filename
     *          retrieved from the configuration. It then calls `_broadcastTriangleMesh()` to
     *          distribute the data across all ranks, ensuring all processes have a copy of the mesh.
     * @throws std::runtime_error if the mesh file is missing or cannot be loaded.
     */
    void _initializeSurfaceMesh();

    /**
     * @brief Initializes the Axis-Aligned Bounding Box (AABB) tree for the surface mesh.
     * @details This method creates an AABB tree using the valid (non-degenerate) triangles
     *          from the surface mesh. If the mesh is empty or contains only degenerate triangles,
     *          an exception is thrown.
     * @throws std::runtime_error if the surface mesh is empty or contains only degenerate triangles,
     *         preventing AABB construction.
     * @note This method should be called after `initializeSurfaceMesh()`.
     */
    void _initializeSurfaceMeshAABB();

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
