#include <concepts>
#include <gmsh.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <vector>

/// @brief Types of mesh.
enum class MeshType
{
    Uniform, ///< Uniform mesh.
    Adaptive ///< Adaptive mesh.
};

/**
 * @brief A class to create two plates and apply either uniform or adaptive meshing.
 */
class TwoPlatesCreator
{
private:
    MeshType m_mesh_type; ///< The type of mesh to apply.
    double m_cell_size;   ///< The size of the mesh cells (for uniform mesh).
    double m_unit;        ///< Scaling coefficient (e.g., for millimeters).

    /**
     * @brief Initializes Gmsh and prepares the model with two plates.
     *
     * Algorithm:
     * 1. Initialize Gmsh.
     * 2. Add a new model named "TwoPlates".
     * 3. Create two plates using `gmsh::model::occ::addBox`.
     * 4. Synchronize the OpenCASCADE model with Gmsh.
     *
     * Example usage:
     * @code
     * TwoPlatesCreator creator(MeshType::Uniform, 1.0, 1.0);
     * creator.initializeGmsh();
     * @endcode
     */
    void _initializeGmsh();

    /**
     * @brief Adds two plates to the model.
     *
     * Algorithm:
     * 1. Use `gmsh::model::occ::addBox` to create the first plate.
     * 2. Use `gmsh::model::occ::addBox` to create the second plate.
     * 3. Synchronize the OpenCASCADE model with Gmsh.
     */
    void _addPlates();

    /**
     * @brief Retrieves all boundary tags for all volumes in the model.
     * @return A vector of boundary tags.
     *
     * Algorithm:
     * 1. Get all volume entities using `gmsh::model::getEntities`.
     * 2. For each volume:
     *    - Retrieve boundary entities using `gmsh::model::getBoundary`.
     *    - Filter boundary entities to include only surfaces.
     * 3. Return a vector of all boundary surface tags.
     */
    std::vector<int> _getAllBoundaryTags();

    /**
     * @brief Applies the specified type of mesh to the model.
     *
     * Algorithm:
     * 1. If the mesh type is uniform:
     *    - Set `Mesh.MeshSizeMin` and `Mesh.MeshSizeMax` to the cell size.
     * 2. If the mesh type is adaptive:
     *    - Retrieve boundary tags using `getAllBoundaryTags`.
     *    - Convert boundary tags to `std::vector<double>`.
     *    - Create a `Distance` field for boundary refinement.
     *    - Create a `Threshold` field for adaptive mesh sizing.
     *    - Set the `Threshold` field as the background mesh.
     */
    void _applyMesh();

    /**
     * @brief Applies adaptive mesh refinement to the model.
     * @param boundary_tags A vector of boundary surface tags.
     *
     * Algorithm:
     * 1. Convert `boundary_tags` to `std::vector<double>`.
     * 2. Create a `Distance` field for the specified boundary surfaces.
     * 3. Create a `Threshold` field to define size variation.
     * 4. Set the `Threshold` field as the background mesh.
     */
    void _applyAdaptiveMesh(const std::vector<int> &boundary_tags);

    /**
     * @brief Generates the mesh based on the specified mesh type.
     *
     * Algorithm:
     * 1. Depending on the mesh type:
     *    - For uniform mesh:
     *      - Set `Mesh.MeshSizeMin` and `Mesh.MeshSizeMax` to the cell size.
     *    - For adaptive mesh:
     *      - Get boundary tags using `getAllBoundaryTags`.
     *      - Create a `Distance` field for boundary refinement.
     *      - Define a `Threshold` field with minimum and maximum sizes.
     *      - Set the `Threshold` field as the background mesh.
     * 2. Generate the 2D mesh using `gmsh::model::mesh::generate`.
     * 3. Write the mesh to a file named "TwoPlates.msh".
     * 4. Launch the Gmsh GUI to display the generated mesh.
     *
     * Example usage:
     * @code
     * TwoPlatesCreator creator(MeshType::Adaptive, 0.0, 1.0);
     * creator.initializeGmsh();
     * creator.generateMesh();
     * @endcode
     */
    void _generateMesh();

public:
    /**
     * @brief Constructor to initialize the TwoPlatesCreator and create two plates by default.
     * @param mesh_type The type of mesh to apply.
     * @param cell_size The size of the mesh cells (used only for uniform mesh).
     * @param unit Scaling coefficient (e.g., 1.0 for millimeters).
     * @throws std::invalid_argument If cell_size is non-positive for uniform mesh.
     *
     * Example usage:
     * @code
     * TwoPlatesCreator creator(MeshType::Uniform, 1.0, 1.0); // For uniform mesh
     * @endcode
     */
    TwoPlatesCreator(MeshType mesh_type, double cell_size = 1.0, double unit = 1.0);

    /// @brief Destructor to finalize the Gmsh session.
    ~TwoPlatesCreator();

    /// @brief Runs the Gmsh application to show the model.
    void show();
};
