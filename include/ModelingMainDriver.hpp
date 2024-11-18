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

#ifdef USE_CUDA
#include "Geometry/AABBTreeDevice.cuh"
#include "Particle/ParticleDevice.cuh"
#endif // !USE_CUDA

/**
 * @brief This class represents main driver that manages the PIC part and surface collision tracker.
 * Main algo:
 *      1. Get the mesh from the mesh file. Filling basis gradient functions ∇φi for each node.
 *      2. Construct GSM (Global Stiffness Matrix) at once. Setting BC (Boundary Conditions) to the GSM and vector `b`. Ax=b, where A = GSM.
 *      3. Spawn particles from the specified in configuration file sources.
 *      4. Tracking particles in the mesh.
 *      5. Collecting node charge densities (ρi) on each time step for each node.
 *      6. Pass ρi to the vector `b` to solve equation Ax=b. (Dirichlet Boundary conditions).
 *      7. Solve the equation Ax=b on each time step.
 *      8. Calculate electric potential (φi) of the each node from the `x` vector in result from Ax=b.
 *      9. Calculate electic field E=(Ex,Ey,Ez) of the each mesh cell based on the φi and ∇φi. E=-∇φ -> E_cell=-1/(6V)*Σ(φi⋅∇φi).
 *      10. Electro-magnetic pushing of the particles (update velocity).
 *          10.1. Update position.
 *          10.2. Check is particle inside mesh. If no - particle = settled particle - stop tracking it (incrementing count of the settled particles on certain triangle on the mesh).
 *      11. Gas collision statistical checking (HS/VHS/VSS). If colided - update velocity.
 *          11.1-11.2. Repeat 10.1. and 10.2.
 *      13. Check if particle settled. If so - incrementing count of the settled particles on certain triangle on the mesh.
 *      14. Saving all the collected particles in triangles (surface mesh) to .hdf5 file.
 *      15*. Saving all trajectories of the particles to make animation.
 */
class ModelingMainDriver final
{
private:
    static constexpr short const kdefault_max_numparticles_to_anim{5'000}; ///< Maximal count of particles to do animation.

    static std::mutex m_PICTracker_mutex;              ///< Mutex for synchronizing access to the particles in tetrahedrons.
    static std::mutex m_nodeChargeDensityMap_mutex;    ///< Mutex for synchronizing access to the charge densities in nodes.
    static std::mutex m_particlesMovement_mutex;       ///< Mutex for synchronizing access to the trajectories of particles.
    static std::shared_mutex m_settledParticles_mutex; ///< Mutex for synchronizing access to settled particle IDs.
    static std::atomic_flag m_stop_processing;         ///< Flag-checker for condition (counter >= size of particles).

    /* All the neccessary data members from the mesh. */
    MeshTriangleParamVector _triangleMesh;   ///< Triangle mesh params acquired from the mesh file. Surface mesh.
    TriangleVector _triangles;               ///< Triangles extracted from the triangle mesh params `_triangleMesh` (surface mesh). Need to initialize AABB tree.
    AABB_Tree_Triangle _surfaceMeshAABBtree; ///< AABB tree for the surface mesh to effectively detect collisions with surface.
    GmshSessionManager _gmshSessionManager;  ///< Object of the volume creator that is RAII object that initializes and finalizes GMSH. Needed to initialize all necessary objects from the mesh.

    /* All the neccessary data members for the simulation. */
    ParticleVector m_particles;                        ///< Projective particles.
    double _gasConcentration;                          ///< Gas concentration. Needed to use colide projectives with gas mechanism.
    std::set<int> _settledParticlesIds;                ///< Set of the particle IDs that are been settled (need to avoid checking already settled particles).
    std::map<size_t, int> _settledParticlesCounterMap; ///< Map to handle settled particles: (Triangle ID | Counter of settled particle in this triangle).

    ConfigParser m_config;                                                ///< `ConfigParser` object to get all the simulation physical paramters.
    std::map<size_t, std::vector<Point>> m_particlesMovement;             ///< Map to store all the particle movements: (Particle ID | All positions).
    std::map<double, std::map<size_t, ParticleVector>> m_particleTracker; ///< Global particle in cell tracker (Time moment: (Tetrahedron ID | Particles inside)).

#ifdef USE_CUDA
    ParticleDevice_t *md_particles = nullptr;
    AABBNodeDevice_t *md_nodes = nullptr;
    TriangleDevice_t *md_triangles = nullptr;
    size_t md_particleCount = 0ul;
    size_t md_nodeCount = 0ul;
    size_t md_triangleCount = 0ul;

    AABBTreeDevice md_aabbTreeDevice;

    void _initializeDeviceMemory();
    void _freeDeviceMemory();
#endif

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

    /**
     * @brief 1st step of the PIC (Particle-In-Cell) modeling.
     *        Initializes the Finite Element Method (FEM) components.
     *
     * @details This function initializes the global stiffness matrix assembler,
     *          creates a cubic grid for the tetrahedron mesh, sets the boundary conditions,
     *          and initializes the solution vector.
     *
     * @param config The configuration object containing the necessary parameters.
     * @param assemblier The global stiffness matrix assembler to be initialized.
     * @param cubicGrid The cubic grid structure for the tetrahedron mesh to be created.
     * @param boundaryConditions A map of boundary conditions to be set.
     * @param solutionVector The solution vector to be initialized.
     */
    void _initializeFEM(std::shared_ptr<GSMAssemblier> &assemblier,
                        std::shared_ptr<CubicGrid> &cubicGrid,
                        std::map<GlobalOrdinal, double> &boundaryConditions,
                        std::shared_ptr<VectorManager> &solutionVector);

    /**
     * @brief Processes particles in parallel using multiple threads.
     *
     * This function splits the particles among the given number of threads and asynchronously or
     * deferred processes each segment of particles using the provided function. The launch policy
     * can be either immediate execution (async) or deferred execution.
     *
     * @tparam Function A callable type that represents the function to be executed in each thread.
     * @tparam Args Variadic template for additional arguments to be passed to the function.
     * @param num_threads The number of threads to use for processing.
     * @param function The member function to be executed in each thread.
     * @param launch_policy The launch policy, which can be std::launch::async or std::launch::deferred.
     * @param args Additional arguments to be forwarded to the function.
     *
     * @throws std::invalid_argument if num_threads exceeds the number of available hardware threads.
     */
    template <typename Function, typename... Args>
    void _processWithThreads(unsigned int num_threads, Function &&function, std::launch launch_plicy, Args &&...args);

    /**
     * @brief Processes particle tracking within a specified range of particles.
     *
     * This function processes particle tracking for particles within the specified range
     * [start_index, end_index). It performs operations based on the time `t` and interacts
     * with the provided cubic grid and GSM assembler. The node charge density map is updated
     * accordingly during the process.
     *
     * @param start_index The starting index of the particles to process.
     * @param end_index The ending index of the particles to process.
     * @param t The current time of the simulation.
     * @param cubicGrid A shared pointer to the 3D cubic grid object used for particle tracking.
     * @param assemblier A shared pointer to the GSM assembler that handles particle interactions.
     * @param nodeChargeDensityMap A reference to a map that tracks the charge density at each node.
     */
    void _processParticleTracker(size_t start_index, size_t end_index, double t,
                                 std::shared_ptr<CubicGrid> cubicGrid, std::shared_ptr<GSMAssemblier> assemblier,
                                 std::map<GlobalOrdinal, double> &nodeChargeDensityMap);

    /**
     * @brief Solves the equation for the system using node charge density and boundary conditions.
     *
     * This function solves the system of equations based on the provided node charge density map
     * and boundary conditions. It updates the solution vector accordingly.
     *
     * @param nodeChargeDensityMap A reference to a map containing the charge density at each node.
     * @param assemblier A shared pointer to the GSM assembler that constructs the system of equations.
     * @param solutionVector A shared pointer to the vector manager that holds the solution.
     * @param boundaryConditions A reference to a map containing boundary conditions for the system.
     * @param time The current simulation time.
     */
    void _solveEquation(std::map<GlobalOrdinal, double> &nodeChargeDensityMap,
                        std::shared_ptr<GSMAssemblier> &assemblier,
                        std::shared_ptr<VectorManager> &solutionVector,
                        std::map<GlobalOrdinal, double> &boundaryConditions, double time);

    /**
     * @brief Processes particle-in-cell (PIC) and surface collision tracking within a particle range.
     *
     * This function tracks particle-in-cell interactions and surface collisions for particles
     * in the specified range [start_index, end_index). It updates the state of the particles based on
     * the time `t` and interactions with the provided cubic grid and GSM assembler.
     *
     * @param start_index The starting index of the particles to process.
     * @param end_index The ending index of the particles to process.
     * @param t The current time of the simulation.
     * @param cubicGrid A shared pointer to the 3D cubic grid object used for particle tracking.
     * @param assemblier A shared pointer to the GSM assembler that handles particle and surface interactions.
     */
    void _processPIC_and_SurfaceCollisionTracker(size_t start_index, size_t end_index, double t,
                                                 std::shared_ptr<CubicGrid> cubicGrid, std::shared_ptr<GSMAssemblier> assemblier);

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
