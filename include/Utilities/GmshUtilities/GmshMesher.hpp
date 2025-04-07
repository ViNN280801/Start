#ifndef GMSH_MESHER_HPP
#define GMSH_MESHER_HPP

#include <string>
#include <vector>

#include "Utilities/GmshUtilities/GmshUtils.hpp"

/// @brief Types of mesh.
enum class MeshType
{
    Uniform, ///< Uniform mesh.
    Adaptive ///< Adaptive mesh (non-uniform).
};

/**
 * @brief Utility class for generating and applying meshes in Gmsh.
 *
 * This class encapsulates operations related to mesh generation and allows
 * applying uniform and adaptive meshes to a Gmsh model. It provides a clean
 * interface for interacting with Gmsh mesh settings and generating mesh files.
 */
class GmshMesher
{
private:
    std::string m_mesh_filename;      ///< Filename of the mesh file.
    double m_unit;                    ///< Unit scaling factor (e.g. [mm], [cm], [m]).
    double m_cell_size;               ///< Default cell size for uniform meshes.
    MeshType m_mesh_type;             ///< Type of the mesh (Uniform, Adaptive).

public:
    /**
     * @brief Constructor for GmshMesher.
     *
     * Initializes the GmshMesher with specified unit scaling and default cell size.
     * 
     * @param mesh_filename Filename of the name with .msh extension. 
     *                      If the filename does not end with `.msh`, the extension is appended automatically.
     * @param mesh_type Type of the mesh to apply (need for checking).
     * @param cell_size Default cell size for uniform meshes.
     * @param unit Unit scaling factor (e.g., 1.0 for millimeters). All mesh parameters
     *             such as sizes and distances are scaled using this factor.
     *
     * @note The `unit` parameter is crucial for ensuring consistency in model dimensions.
     *
     * **Example Usage:**
     * @code
     * GmshMesher mesher(1.0, 5.0);
     * @endcode
     */
    GmshMesher(std::string_view mesh_filename, MeshType mesh_type, double cellSize, double unit);

    /**
     * @brief Applies a mesh to the Gmsh model, either uniform or adaptive.
     *
     * This method determines the type of mesh to apply based on the `meshType` parameter.
     * If the mesh type is `Uniform`, it applies a uniform mesh across the model.
     * If the mesh type is `Adaptive`, it applies an adaptive mesh using the specified boundary tags
     * and mesh field parameters.
     *
     * @param meshType The type of mesh to apply:
     *                 - `MeshType::Uniform` for a uniform mesh.
     *                 - `MeshType::Adaptive` for an adaptive mesh.
     * @param boundaryTags A vector of boundary surface tags for the adaptive mesh.
     *                     If empty or unspecified, all boundary tags will be fetched using `GmshUtils::getAllBoundaryTags()`.
     * @param inFieldTag The field ID used for the distance field in the adaptive mesh. Default is `1`.
     * @param sizeMin The minimum size of mesh elements in the adaptive mesh. Default is `1.0`.
     * @param sizeMax The maximum size of mesh elements in the adaptive mesh. Default is `100.0`.
     * @param distMin The minimum distance for applying `sizeMin` in the adaptive mesh. Default is `5.0`.
     * @param distMax The maximum distance for applying `sizeMax` in the adaptive mesh. Default is `50.0`.
     *
     * @throws std::invalid_argument If:
     * - `boundaryTags` is empty for `MeshType::Adaptive`.
     * - Any of `sizeMin`, `sizeMax`, `distMin`, or `distMax` are non-positive.
     * - `sizeMin` > `sizeMax` or `distMin` > `distMax`.
     * @throws std::runtime_error If Gmsh is not initialized before applying the mesh.
     * @throws std::invalid_argument If an invalid `meshType` is passed.
     *
     * @note This method assumes that Gmsh has been initialized and a valid model exists.
     *
     * **Algorithm:**
     * 1. Check if Gmsh is initialized using `GmshUtils::gmshInitializeCheck()`. Throw an error if not.
     * 2. Based on the `meshType`:
     *    - If `MeshType::Uniform`, call `applyUniformMesh()` to set a uniform mesh.
     *    - If `MeshType::Adaptive`, call `applyAdaptiveMesh()` with the provided boundary tags and parameters.
     *    - If the `meshType` is invalid, throw an exception.
     *
     * **Example Usage:**
     * @code
     * GmshMesher mesher(1.0, 5.0);
     *
     * // Apply a uniform mesh
     * mesher.applyMesh(MeshType::Uniform);
     *
     * // Apply an adaptive mesh with custom parameters
     * std::vector<int> boundaryTags = {1, 2, 3};
     * mesher.applyMesh(MeshType::Adaptive, boundaryTags, 1, 2.0, 50.0, 10.0, 100.0);
     * @endcode
     *
     * **Output:**
     * - For `MeshType::Uniform`, the mesh will have uniform cell sizes.
     * - For `MeshType::Adaptive`, the mesh will vary based on distance from the specified boundary surfaces.
     *
     * **Performance Considerations:**
     * - Adaptive meshing involves additional computational overhead for distance and threshold field calculations.
     * - Ensure that the boundary tags provided are accurate to avoid unnecessary computations.
     *
     * **Limitations:**
     * - Requires Gmsh to be initialized before calling this method.
     * - The adaptive mesh parameters must be carefully chosen to avoid overlapping or conflicting fields.
     *
     * @see applyUniformMesh()
     * @see applyAdaptiveMesh()
     */
    void applyMesh(MeshType meshType,
                   std::vector<int> const &boundaryTags,
                   int inFieldTag = 1,
                   double sizeMin = 1.0,
                   double sizeMax = 100.0,
                   double distMin = 5.0,
                   double distMax = 50.0) const;

    /**
     * @brief Applies a uniform mesh to the Gmsh model.
     *
     * This method sets a uniform mesh size throughout the model by setting
     * `Mesh.MeshSizeMin` and `Mesh.MeshSizeMax` to the specified cell size.
     *
     * **Example Usage:**
     * @code
     * mesher.applyUniformMesh();
     * @endcode
     */
    void applyUniformMesh() const;

    /**
     * @brief Applies an adaptive mesh to the Gmsh model.
     *
     * This method uses Gmsh mesh fields to create an adaptive mesh. It defines a
     * distance field and a threshold field to vary mesh sizes based on distances
     * from specified boundary surfaces.
     *
     * @param boundaryTags A vector of surface tags to which the adaptive mesh is applied.
     * @param inFieldTag The field number to use as input for the threshold field.
     * @param sizeMin Minimum size of mesh elements.
     * @param sizeMax Maximum size of mesh elements.
     * @param distMin Minimum distance at which `sizeMin` is enforced.
     * @param distMax Maximum distance at which `sizeMax` is enforced.
     *
     * @note The field parameters allow fine-grained control over mesh adaptation.
     *
     * **Example Usage:**
     * @code
     * std::vector<int> boundaryTags = {1, 2, 3};
     * mesher.applyAdaptiveMesh(boundaryTags, 1, 1.0, 100.0, 5.0, 50.0);
     * @endcode
     */
    void applyAdaptiveMesh(std::vector<int> const &boundaryTags,
                           int inFieldTag = 1,
                           double sizeMin = 1.0,
                           double sizeMax = 100.0,
                           double distMin = 5.0,
                           double distMax = 50.0) const;

    /**
     * @brief Generates a mesh file for the Gmsh model with the specified dimension.
     *
     * This method generates a mesh of the given dimension (2D or 3D) for the current Gmsh model
     * and writes it to the specified file. If the provided filename does not have the `.msh` extension,
     * it will be appended automatically. The method validates input parameters and provides error handling
     * for potential failures during mesh generation.
     *
     * @param dimension The dimension of the mesh to generate (2D or 3D).
     *                  Valid values are:
     *                  - `2`: Generate a 2D mesh.
     *                  - `3`: Generate a 3D mesh.
     *
     * @throws std::invalid_argument If:
     * - `filename` is empty.
     * - `dimension` is not 2 or 3.
     *
     * @throws std::exception If an error occurs during mesh generation.
     *
     * @note This method assumes that the Gmsh model has been initialized and populated with geometry and mesh settings.
     *
     * @details
     * **Algorithm:**
     * 1. Validate the `filename`:
     *    - If the filename is empty, throw an exception.
     *    - If the filename does not end with `.msh`, append the `.msh` extension.
     * 2. Validate the `dimension`:
     *    - Ensure the dimension is either `2` (2D) or `3` (3D). Otherwise, throw an exception.
     * 3. Attempt to generate the mesh and save it to the file:
     *    - Call `gmsh::model::mesh::generate(dimension)` to generate the mesh.
     *    - Call `gmsh::write(mesh_filename)` to write the mesh to the specified file.
     * 4. Handle exceptions:
     *    - If a standard exception is thrown, log the error and rethrow it.
     *    - If an unknown exception is thrown, log a generic error and rethrow it.
     *
     * **Example Usage:**
     * @code
     * GmshMesher mesher(1.0, 5.0); // Initialize with unit scaling and cell size.
     * mesher.generateMeshfile("example_mesh", 2); // Generate a 2D mesh and save it to "example_mesh.msh".
     * @endcode
     *
     * **Output:**
     * - On success, the mesh file is saved to the specified location with the provided filename or filename + `.msh`.
     * - If an error occurs, it is logged, and the exception is propagated to the caller.
     *
     * **Performance Considerations:**
     * - For complex models or high-resolution meshes, the mesh generation process might take significant time.
     * - Ensure sufficient memory is available for 3D meshes in large models.
     *
     * **Limitations:**
     * - Only supports dimensions 2D and 3D. Other values for `dimension` will result in an exception.
     *
     * @see gmsh::model::mesh::generate()
     * @see gmsh::write()
     */
    void generateMeshfile(int dimension) const;
};

#endif // !GMSH_MESHER_HPP
