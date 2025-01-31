#ifndef GMSHUTILS_HPP
#define GMSHUTILS_HPP

#include <gmsh.h>

#include "Geometry/GeometryTypes.hpp"
#include "Utilities/ConfigParser.hpp"

class GmshUtils
{
private:
    static std::string const kdefault_code_work_in_one_session; ///< Default code that points Gmsh is working in one session without openning any file.

public:
    /**
     * @brief Ensures that Gmsh is initialized before performing any operations.
     *
     * This function checks whether the Gmsh environment is initialized. If it is not,
     * an error message is logged, and an exception is thrown.
     *
     * @throws std::runtime_error if Gmsh is not initialized.
     *
     * @note This function must be called before using any Gmsh API functions.
     */
    static void gmshInitializeCheck();

    /**
     * @brief Validates and initializes the Gmsh model, ensuring the mesh file is correctly opened.
     *
     * This function ensures that the Gmsh environment is properly initialized and the specified mesh file
     * is opened if required. It performs the following steps:
     * 1. Calls `GmshUtils::gmshInitializeCheck()` to verify that Gmsh is initialized.
     * 2. If `meshFilename` is not equal to `special_code`, it:
     *    - Uses `util::check_gmsh_mesh_file()` to validate the existence and integrity of the file.
     *    - Opens the file using `gmsh::open()`, allowing subsequent operations on the mesh.
     *
     * @param meshFilename The filename of the Gmsh mesh to be validated and opened.
     * @param special_code A special identifier indicating that the function is operating within an active Gmsh session
     *                     (without explicitly opening a file). If `meshFilename` matches `special_code`, no file is opened.
     *
     * @throws std::runtime_error If Gmsh is not initialized or if the file validation fails (e.g., file does not exist,
     *                            incorrect extension, empty file, or inaccessible).
     *
     * @note This function is primarily used internally to ensure mesh files are handled correctly before querying
     *       or modifying the Gmsh model.
     *
     * @see GmshUtils::gmshInitializeCheck()
     * @see util::check_gmsh_mesh_file()
     */
    static void checkAndOpenMesh(std::string_view meshFilename);

    /**
     * @brief Retrieves all boundary surface tags from the Gmsh model.
     * @param meshFilename The filename of the mesh (`.msh` format). Default is `kdefault_code_work_in_one_session`,
     *        which means the function assumes the mesh is already loaded into Gmsh.
     *
     * This function extracts all surface tags that form the boundaries of 3D volume elements
     * present in the model.
     *
     * @return std::vector<int> A vector containing the tags of boundary surfaces.
     *
     * @details
     * **Algorithm:**
     * 1. Retrieve all 3D volume entities using `gmsh::model::getEntities(3)`.
     * 2. Extract boundary surfaces for each volume using `gmsh::model::getBoundary`.
     * 3. Store unique surface tags in a vector.
     *
     * @throws std::runtime_error if there are no volume entities or no boundary surfaces found.
     *
     * @warning This function only works with 3D models. It does not retrieve boundaries for 2D elements.
     *
     * @note The returned tags correspond to surfaces that enclose the volume elements.
     */
    static std::vector<int> getAllBoundaryTags(std::string_view meshFilename = kdefault_code_work_in_one_session);

    /**
     * @brief Retrieves the integer tag of a physical group by its name.
     *
     * This function searches for a physical group with the specified name in the model
     * and returns its tag.
     *
     * @param physicalGroupName The name of the physical group.
     * @param meshFilename The filename of the mesh (`.msh` format). Default is `kdefault_code_work_in_one_session`,
     *        which means the function assumes the mesh is already loaded into Gmsh.
     *
     * @return int The tag corresponding to the physical group.
     *
     * @throws std::runtime_error If the physical group is not found.
     *
     * @details
     * **Algorithm:**
     * 1. Ensure that Gmsh is initialized.
     * 2. If a mesh file is specified, open it.
     * 3. Retrieve all physical groups using `gmsh::model::getPhysicalGroups(2)`.
     * 4. Search for the group matching `physicalGroupName` and return its tag.
     *
     * @note This function only searches within **surface** physical groups (dimension = 2).
     */
    static int getPhysicalGroupTagByName(std::string_view physicalGroupName, std::string_view meshFilename = kdefault_code_work_in_one_session);

    /**
     * @brief Computes centroids of triangular cells belonging to a given physical group.
     *
     * This function calculates the centroid of each triangular cell in a given physical group.
     * The centroid is computed as the average of the three vertex coordinates.
     *
     * @param physicalGroupName The name of the physical group to process.
     * @param meshFilename The filename of the `.msh` file. Default is `kdefault_code_work_in_one_session`.
     *
     * @return TriangleCellCentersMap A map containing:
     *  - **Key:** Triangle cell ID.
     *  - **Value:** A 3D coordinate array representing the centroid.
     *
     * @throws std::runtime_error If the physical group does not contain any triangles.
     *
     * @details
     * **Algorithm:**
     * 1. Find the tag of the physical group.
     * 2. Retrieve the node coordinates for the group.
     * 3. Identify triangular cells using `gmsh::model::mesh::getElementsByType(2)`.
     * 4. Compute the centroid for each triangle.
     * 5. Store the centroids in a map and return them.
     *
     * @note The function assumes that all elements in the physical group are triangular.
     */
    static TriangleCellCentersMap getCellCentersByPhysicalGroupName(std::string_view physicalGroupName, std::string_view meshFilename = kdefault_code_work_in_one_session);

    /**
     * @brief Extracts all triangle cells associated with a physical group.
     *
     * This function retrieves triangular elements from the specified physical group and
     * constructs a `TriangleCellMap`, which contains each triangle's geometry, area, and
     * an initial count of deposited particles.
     *
     * @param physicalGroupName The name of the physical group.
     * @param meshFilename The filename of the `.msh` mesh file. Default is `kdefault_code_work_in_one_session`.
     *
     * @return TriangleCellMap A map of triangle cells.
     *
     * @throws std::runtime_error If the physical group does not contain any triangles.
     *
     * @details
     * **Algorithm:**
     * 1. Find the physical group tag.
     * 2. Retrieve node coordinates.
     * 3. Identify triangular elements.
     * 4. Construct `TriangleCell` objects with their area and initial particle count.
     *
     * @note Only triangular cells are extracted. Degenerate triangles (zero area) are skipped.
     */
    static TriangleCellMap getCellsByPhysicalGroupName(std::string_view physicalGroupName, std::string_view meshFilename = kdefault_code_work_in_one_session);

    /**
     * @brief Retrieves all physical groups in the Gmsh model.
     *
     * This function returns a list of all physical groups defined in the current Gmsh model, including their:
     * - **Dimension** (`dim`): The geometric dimension of the group (e.g., 0 for points, 1 for lines, 2 for surfaces, 3 for volumes).
     * - **Tag** (`tag`): A unique integer identifier assigned to the physical group by Gmsh.
     * - **Name** (`name`): The user-defined name of the physical group.
     *
     * **Algorithm:**
     * 1. Calls `checkAndOpenMesh()` to ensure Gmsh is initialized and the correct mesh file is loaded.
     * 2. Retrieves all physical groups using `gmsh::model::getPhysicalGroups()`, returning a list of `(dim, tag)` pairs.
     * 3. For each retrieved physical group:
     *    - Extracts the associated name using `gmsh::model::getPhysicalName()`.
     *    - Constructs a tuple `(dim, tag, name)` and adds it to the output vector.
     * 4. If no physical groups are found, throws an exception.
     *
     * @param meshFilename The filename of the Gmsh mesh file. If left as the default `kdefault_code_work_in_one_session`,
     *                     the function assumes the Gmsh session is already active and does not explicitly open a file.
     *
     * @return A vector of tuples where:
     *         - The first element (`int`) is the geometric dimension of the physical group.
     *         - The second element (`int`) is the unique integer tag assigned to the group.
     *         - The third element (`std::string`) is the name of the physical group.
     *
     * @throws std::runtime_error If no physical groups are found or if an error occurs during retrieval.
     *
     * @see GmshUtils::hasPhysicalGroup()
     * @see checkAndOpenMesh()
     *
     * **Example Usage:**
     * @code
     * auto physicalGroups = GmshUtils::getAllPhysicalGroups();
     * for (const auto& [dim, tag, name] : physicalGroups)
     * {
     *     std::cout << "Physical Group - Dim: " << dim << ", Tag: " << tag << ", Name: " << name << std::endl;
     * }
     * @endcode
     */
    static std::vector<std::tuple<int, int, std::string>> getAllPhysicalGroups(std::string_view meshFilename = kdefault_code_work_in_one_session);

    /**
     * @brief Checks if a physical group with the specified name exists in the Gmsh model.
     *
     * This function determines whether a physical group (a named entity in Gmsh) exists within the model.
     * It executes the following steps:
     * 1. Calls `checkAndOpenMesh()` to validate the mesh file or confirm an active session.
     * 2. Retrieves all physical groups in the model using `GmshUtils::getAllPhysicalGroups()`.
     * 3. Iterates over the retrieved groups and checks if any match the specified `physicalGroupName`.
     *
     * @param physicalGroupName The name of the physical group to check for existence.
     * @param meshFilename The filename of the Gmsh mesh file. If left as the default `kdefault_code_work_in_one_session`,
     *                     the function assumes the Gmsh session is already active and does not explicitly open a file.
     *
     * @return `true` if a physical group with the given name exists; otherwise, `false`.
     *
     * @throws std::runtime_error If Gmsh is not initialized or the physical groups cannot be retrieved.
     *
     * @note This function is compatible with both **C++17** and **C++20**:
     *       - In **C++20**, it utilizes `std::ranges::find_if` for better performance and readability.
     *       - In **C++17**, it falls back to `std::find_if`.
     *
     * @see GmshUtils::getAllPhysicalGroups()
     * @see checkAndOpenMesh()
     *
     * **Example Usage:**
     * @code
     * if (GmshUtils::hasPhysicalGroup("BoundarySurface"))
     * {
     *     std::cout << "Physical group 'BoundarySurface' exists in the model.\n";
     * }
     * else
     * {
     *     std::cerr << "Physical group not found.\n";
     * }
     * @endcode
     */
    static bool hasPhysicalGroup(std::string_view physicalGroupName, std::string_view meshFilename = kdefault_code_work_in_one_session);

/**
 * @brief Finds the tag of a surface by matching its 3D coordinates.
 *
 * This function searches for a surface that contains all the given 3D points.
 *
 * @tparam MatrixType A matrix-like structure containing 3D points.
 * @tparam ValueType The numerical type of the coordinates.
 *
 * @param surfaceCoords A list of 3D points defining the target surface.
 * @param meshFilename The filename of the `.msh` file.
 *
 * @return int The tag of the matching surface, or `-1` if not found.
 *
 * @throws std::runtime_error If no matching surface is found.
 *
 * @details
 * **Algorithm:**
 * 1. Validate the input matrix.
 * 2. Retrieve all surfaces using `gmsh::model::getEntities(2)`.
 * 3. Extract node coordinates for each surface.
 * 4. Compare with the provided coordinates.
 * 5. Return the matching surface tag.
 *
 * @note This function requires the model to be loaded in Gmsh before execution.
 */
#if __cplusplus >= 202002L
    template <typename ValueType>
    static int findSurfaceTagByCoords(Matrix<ValueType> auto const &surfaceCoords, std::string_view meshFilename = kdefault_code_work_in_one_session)
#else
    template <typename MatrixType, typename ValueType>
    std::enable_if<is_matrix_v<MatrixType, ValueType>, int> static int findSurfaceTagByCoords(MatrixType const &surfaceCoords)
#endif
    {
        checkAndOpenMesh(meshFilename);

        if (surfaceCoords.empty())
            throw std::runtime_error("Surface coords are empty.");
        for (auto const &point : surfaceCoords)
            if (point.size() != 3ul)
                throw std::runtime_error(util::stringify("Assuming 3D space, but have point with ", point.size(), " coords"));

        std::vector<std::pair<int, int>> surfaces;
        gmsh::model::getEntities(surfaces, 2);

        int targetSurfaceTag{-1};
        for (auto const &surface : surfaces)
        {
            int currentSurfaceTag{surface.second};
            std::vector<size_t> nodeTags;
            std::vector<double> coords, parametricCoords;
            gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords, 2, currentSurfaceTag, true, true);

            std::set<std::array<double, 3ul>> surfaceNodes;
            for (size_t i{}; i < coords.size(); i += 3)
                surfaceNodes.insert({coords.at(i), coords.at(i + 1), coords.at(i + 2)});

            bool isTargetSurface{true};
            for (auto const &targetCoord : surfaceCoords)
            {
                std::array<double, 3ul> targetArray = {targetCoord[0], targetCoord[1], targetCoord[2]};
                if (surfaceNodes.find(targetArray) == surfaceNodes.cend())
                {
                    isTargetSurface = false;
                    break;
                }
            }

            if (isTargetSurface)
            {
                targetSurfaceTag = currentSurfaceTag;
                break;
            }
        }

        // There is no need to check 'targetSufaceTag' on equality with '-1' because Gmsh automatically
        // used ref on this variable and assign physical group name to this tag when mesh will build.
        // So, just return, even if now this tag is equal to '-1'.
        return targetSurfaceTag;
    }
};

#endif // !GMSHUTILS_HPP
