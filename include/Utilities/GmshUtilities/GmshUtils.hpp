#ifndef GMSHUTILS_HPP
#define GMSHUTILS_HPP

#include <gmsh.h>

#include "Geometry/GeometryTypes.hpp"
#include "Utilities/ConfigParser.hpp"

class GmshUtils
{
public:
    /**
     * @brief Checker whether Gmsh initialized or not. If not initialized - print error message.
     *
     * @return int '-1' if not initialized, '0' otherwise.
     */
    static int gmshInitializeCheck();

    /**
     * @brief Retrieves all boundary tags for all the volumes in the model.
     * @return A vector of boundary tags.
     *
     * Algorithm:
     * 1. Get all volume entities using `gmsh::model::getEntities`.
     * 2. For each volume:
     *    - Retrieve boundary entities using `gmsh::model::getBoundary`.
     *    - Filter boundary entities to include only surfaces.
     * 3. Return a vector of all boundary surface tags.
     */
    static std::vector<int> getAllBoundaryTags();

    /**
     * @brief Finds integer tag of the physical group by specified name.
     * @param physicalGroupName Physical group name to find.
     * @return
     */
    static int getPhysicalGroupTagByName(std::string_view physicalGroupName);

    /**
     * @brief Get the cell (triangle) centers from the specified by name physical group.
     * @details std::unordered_map<size_t, std::array<double, 3ul>> where: (key - cell ID (tag)|value - center of this triangle).
     *
     * @param physicalGroupName Physical group name to find.
     * @return std::unordered_map<size_t, std::array<double, 3ul>>
     */
    static std::unordered_map<size_t, std::array<double, 3ul>> getCellCentersByPhysicalGroupName(std::string_view physicalGroupName);

    
    static std::unordered_map<size_t, Triangle> getCellsByPhysicalGroupName(std::string_view physicalGroupName);

/**
 * @brief Finds the tag of a surface in the Gmsh model that matches a set of 3D coordinates.
 *
 * This function searches through all surfaces in the Gmsh model to find one whose nodes match the
 * given target coordinates. The coordinates are provided as a matrix-like structure (a range of ranges)
 * where the innermost elements represent 3D points.
 *
 * @tparam MatrixType The type of the container representing the matrix of 3D points. Must satisfy the `is_matrix_v` trait.
 * @tparam ValueType The type of the individual 3D coordinate values (e.g., `double`, `float`, or `int`).
 *
 * @param surfaceCoords A matrix-like structure containing the target 3D coordinates. Each row represents a 3D point,
 *                      and the innermost elements represent the x, y, and z coordinates of that point.
 *
 * @return The tag of the surface in the Gmsh model that matches the target coordinates.
 *         If no matching surface is found, returns `-1` and logs an error.
 *
 * @throws std::runtime_error If Gmsh API calls fail unexpectedly.
 *
 * @details
 * **Algorithm:**
 * 1. Validate the input `surfaceCoords`:
 *    - Ensure it is not empty.
 *    - Ensure each row contains exactly three elements (representing 3D space).
 * 2. Retrieve all surfaces from the Gmsh model using `gmsh::model::getEntities`.
 * 3. For each surface:
 *    - Retrieve its nodes using `gmsh::model::mesh::getNodes`.
 *    - Store the nodes in a set of 3D points for efficient searching.
 *    - Check if all the target coordinates exist within the surface's nodes.
 * 4. If a surface matches all the target coordinates, return its tag.
 * 5. If no surface matches, return `-1` and log an error message.
 *
 * @note This function is compatible with both C++17 and C++20.
 *       - In C++20, the `Matrix` concept is used for type validation.
 *       - In C++17, `std::enable_if` with the `is_matrix_v` trait is used.
 *
 * @note The Gmsh model must be initialized and populated before calling this function.
 *
 * **Example Usage:**
 * @code
 * // Example coordinates
 * std::vector<std::vector<double>> coords = {
 *     {20.0, 5.0, 100.0},
 *     {20.0, 15.0, 100.0},
 *     {80.0, 5.0, 100.0},
 *     {80.0, 15.0, 100.0}
 * };
 *
 * // Find the target surface tag
 * int tag = _findSurfaceTagByCoords(coords);
 * if (tag != -1) {
 *     std::cout << "Found surface tag: " << tag << std::endl;
 * } else {
 *     std::cerr << "No matching surface found!" << std::endl;
 * }
 * @endcode
 *
 * **Performance Considerations:**
 * - The function uses a `std::set` to store and compare surface nodes, which ensures efficient lookups
 *   but may incur additional overhead for large surfaces.
 * - The algorithm performs a linear search over all surfaces in the Gmsh model. For models with many surfaces,
 *   this may impact performance.
 *
 * **Compatibility:**
 * - **C++20:** Uses the `Matrix` concept for compile-time type enforcement.
 * - **C++17:** Uses `std::enable_if` with the `is_matrix_v` trait for SFINAE-based type enforcement.
 */
#if __cplusplus >= 202002L
    template <typename ValueType>
    static int findSurfaceTagByCoords(Matrix<ValueType> auto const &surfaceCoords)
#else
    template <typename MatrixType, typename ValueType>
    std::enable_if<is_matrix_v<MatrixType, ValueType>, int> static int findSurfaceTagByCoords(MatrixType const &surfaceCoords)
#endif
    {
        if (GmshUtils::gmshInitializeCheck() == -1)
            return -1;

        if (surfaceCoords.empty())
        {
            ERRMSG("Surface coords are empty.");
            return -1;
        }
        for (auto const &point : surfaceCoords)
        {
            if (point.size() != 3ul)
            {
                ERRMSG(util::stringify("Assuming 3D space, but have point with ", point.size(), " coords"));
                return -1;
            }
        }

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

        if (targetSurfaceTag == -1)
        {
            ERRMSG("Error finding taget surface.");
        }

        return targetSurfaceTag;
    }
};

#endif // !GMSHUTILS_HPP
