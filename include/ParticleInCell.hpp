#ifndef PARTICLEINCELL_HPP
#define PARTICLEINCELL_HPP

#include <barrier>
#include <mutex>
#include <shared_mutex>

#include "FiniteElementMethod/MatrixEquationSolver.hpp"
#include "Generators/VolumeCreator.hpp"
#include "Geometry/Mesh.hpp"
#include "ParticleTracker/Grid3D.hpp"
#include "Particles/Particle.hpp"
#include "Utilities/ConfigParser.hpp"

class ParticleInCell final
{
private:
    static constexpr short const kdefault_polynomOrder{1};                 ///< Polynom order. Responds for count of the basis functions.
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
    GMSHVolumeCreator vc;                    ///< Object of the volume creator that is RAII object that initializes and finalizes GMSH. Needed to initialize all necessary objects from the mesh.

    /* All the neccessary data members for the simulation. */
    ParticleVector m_particles;                        ///< Projective particles.
    double _gasConcentration;                          ///< Gas concentration. Needed to use colide projectives with gas mechanism.
    std::set<int> _settledParticlesIds;                ///< Set of the particle IDs that are been settled (need to avoid checking already settled particles).
    std::map<size_t, int> _settledParticlesCounterMap; ///< Map to handle settled particles: (Triangle ID | Counter of settled particle in this triangle).

    ConfigParser m_config;                                                ///< `ConfigParser` object to get all the simulation physical paramters.
    std::map<size_t, std::vector<Point>> m_particlesMovement;             ///< Map to store all the particle movements: (Particle ID | All positions).
    std::map<double, std::map<size_t, ParticleVector>> m_particleTracker; ///< Global particle in cell tracker (Time moment: (Tetrahedron ID | Particles inside)).

    /**
     * @brief Checks the validity of the provided mesh filename.
     *
     * This function performs several checks to ensure that the provided mesh filename
     * is valid and can be opened. It checks if the filename is empty, if the file exists,
     * and if the file has the correct `.msh` extension. If any of these conditions are not met,
     * an error message is logged and a `std::runtime_error` is thrown.
     *
     * @throws std::runtime_error if the filename is empty, if the file does not exist, or if the file does not have a `.msh` extension.
     */
    void checkMeshfilename() const;

    /* Initializers for all the necessary objects. */
    void initializeSurfaceMesh();
    void initializeSurfaceMeshAABB();
    void initializeParticles();

    /// @brief Global initializator. Uses all the initializers above.
    void initialize();

    /**
     * @brief Initializes the Finite Element Method (FEM) components.
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
    void initializeFEM(std::shared_ptr<GSMatrixAssemblier> &assemblier,
                       std::shared_ptr<Grid3D> &cubicGrid,
                       std::map<GlobalOrdinal, double> &boundaryConditions,
                       std::shared_ptr<SolutionVector> &solutionVector);

    /**
     * @brief Saves the particle movements to a JSON file.
     *
     * This function saves the contents of m_particlesMovement to a JSON file named "particles_movements.json".
     * It handles exceptions and provides a warning message if the map is empty.
     */
    void saveParticleMovements() const;

    /**
     * @brief Checker for ray-triangle intersection.
     * @param ray Finite ray object - line segment.
     * @param triangle Triangle object.
     * @return ID with what triangle ray intersects, otherwise max `size_t` value (-1ul).
     */
    size_t isRayIntersectTriangle(Ray const &ray, MeshTriangleParam const &triangle);

    /// @brief Returns count of threads from config. Has initial checkings and warning msg when num threads occupies 80% of all the threads.
    unsigned int getNumThreads() const;

    /// @brief Using HDF5Handler to update the mesh according to the settled particles.
    void updateSurfaceMesh();

    /**
     * @brief Runs the given function in a multithreaded mode, splitting the work into segments.
     *
     * @param num_threads Count of the using threads.
     * @tparam Function Type of the function.
     * @tparam Args Types of the function arguments.
     * @param function The function to execute.
     * @param args Arguments for the function.
     */
    template <typename Function, typename... Args>
    void processWithThreads(unsigned int num_threads, Function &&function, Args &&...args);

    void processParticleTracker(size_t start_index, size_t end_index, double t,
                                std::shared_ptr<Grid3D> cubicGrid, std::shared_ptr<GSMatrixAssemblier> assemblier,
                                std::map<GlobalOrdinal, double> &nodeChargeDensityMap);
    void solveEquation(std::map<GlobalOrdinal, double> &nodeChargeDensityMap,
                       std::shared_ptr<GSMatrixAssemblier> &assemblier,
                       std::shared_ptr<SolutionVector> &solutionVector,
                       std::map<GlobalOrdinal, double> &boundaryConditions, double time);
    void processSurfaceCollisionTracker(size_t start_index, size_t end_index, double t,
                                        std::shared_ptr<Grid3D> cubicGrid, std::shared_ptr<GSMatrixAssemblier> assemblier);

public:
    ParticleInCell(std::string_view config_filename);
    void startSimulation();
};

#endif // !PARTICLEINCELL_HPP
