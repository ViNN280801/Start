#ifndef MESH_HPP
#define MESH_HPP

#include <iostream>
#include <map>
#include <optional>
#include <string_view>
#include <tuple>
#include <vector>

#include "MathVector.hpp"
#include "RayTriangleIntersection.hpp"

// Overloaded output streams for mesh params. //
std::ostream &operator<<(std::ostream &os, MeshTriangleParam const &meshParam);
std::ostream &operator<<(std::ostream &os, MeshTetrahedronParam const &meshParam);

/**
 * @brief Constructs an AABB tree from a given mesh parameter vector.
 *
 * This function builds an Axis-Aligned Bounding Box (AABB) tree from a collection
 * of triangles, which are extracted from the provided mesh parameters. The AABB
 * tree can be used for efficient geometric queries such as collision detection,
 * ray intersection, and nearest point computation.
 *
 * @param meshParams A vector containing mesh parameters, where each entry
 * represents a triangle in the mesh. Each MeshParam is expected to contain a
 * Triangle object along with its associated properties.
 *
 * @return An optional AABB tree constructed from the provided mesh parameters.
 * If the input meshParams is empty or only contains degenerate triangles,
 * the function returns std::nullopt.
 */
std::optional<AABB_Tree_Triangle> constructAABBTreeFromMeshParams(MeshTriangleParamVector const &meshParams);

/**
 * @brief Calculates the volume of a tetrahedron.
 * @details This function computes the volume of a tetrahedron by utilizing the CGAL library. The volume is calculated
 *          based on the determinant of a matrix constructed from the coordinates of the tetrahedron's vertices. The formula
 *          for the volume of a tetrahedron given its vertices A, B, C, and D is |dot(AB, cross(AC, AD))| / 6.
 * @param tetrahedron The tetrahedron whose volume is to be calculated.
 * @return The volume of the tetrahedron.
 */
double calculateVolumeOfTetrahedron(Tetrahedron const &tetrahedron);

/// @brief Represents GMSH mesh.
class Mesh
{
private:
    static std::optional<size_t> isRayIntersectTriangleImpl(Ray const &ray, MeshTriangleParam const &triangle);
    static std::optional<std::tuple<size_t, Point>>
    getIntersectionPointImpl(Ray const &ray, MeshTriangleParam const &triangle);
    static double calcTetrahedronVolume(MathVector<double> const &a, MathVector<double> const &b, MathVector<double> const &c, MathVector<double> const &d);

public:
    /**
     * @brief Sets the mesh size factor (globally -> for all objects).
     * This method sets the mesh size factor for generating the mesh.
     * The mesh size factor controls the size of mesh elements in the mesh.
     * @param meshSizeFactor The factor to set for mesh size.
     */
    static void setMeshSize(double meshSizeFactor);

    /**
     * @brief Retrieves parameters from a Gmsh mesh file.
     * This function reads information about triangles from a Gmsh .msh file.
     * It calculates triangle properties such as coordinates, side lengths, area, etc.
     * @param msh_filename The filename of the Gmsh .msh file to parse.
     * @return A vector containing information about each triangle in the mesh.
     */
    static MeshTriangleParamVector getMeshParams(std::string_view msh_filename);

    /**
     * @brief Gets mesh parameters for tetrahedron elements from a Gmsh .msh file.
     * @details This method opens a specified .msh file, reads the mesh nodes and tetrahedron elements,
     *          then calculates the parameters for each tetrahedron (centroid and volume).
     * @param msh_filename The name of the .msh file to be read.
     * @return A vector of tuples, each containing the tetrahedron ID, its vertices, and volume.
     */
    static MeshTetrahedronParamVector getTetrahedronMeshParams(std::string_view msh_filename);

    /**
     * @brief Checks if a given ray intersects with a triangle.
     *
     * This function determines whether a finite ray intersects with a given triangle.
     * If an intersection occurs, the ID of the intersected triangle is returned.
     * Otherwise, an empty `std::optional<size_t>` is returned, indicating no intersection.
     *
     * @param ray A constant reference to the ray (line segment) to be checked.
     * @param triangle A constant reference to the triangle to check for intersection.
     * @return An `std::optional<size_t>` containing the ID of the intersected triangle
     *         if the ray intersects, or `std::nullopt` if there is no intersection.
     */
    static std::optional<size_t> isRayIntersectTriangle(Ray const &ray, MeshTriangleParam const &triangle) { return isRayIntersectTriangleImpl(ray, triangle); }
    static std::optional<size_t> isRayIntersectTriangle(Ray &&ray, MeshTriangleParam const &triangle) { return isRayIntersectTriangleImpl(std::move(ray), triangle); }
    static std::optional<size_t> isRayIntersectTriangle(Ray const &ray, MeshTriangleParam &&triangle) { return isRayIntersectTriangleImpl(ray, std::move(triangle)); }
    static std::optional<size_t> isRayIntersectTriangle(Ray &&ray, MeshTriangleParam &&triangle) { return isRayIntersectTriangleImpl(std::move(ray), std::move(triangle)); }

    /**
     * @brief Gets intersection point of ray and triangle if ray intersects the triangle.
     * @param ray The ray to check for intersection with the triangle.
     * @param triangle MeshParam representing the triangle with which the line segment
     *                 is tested for intersection. It includes the necessary parameters to define
     *                 a triangle in 3D space.
     *
     * @return Returns the ID of the triangle where the particle has settled if an intersection occurs
     *         and intersection point.
     *         If the particle doesn't intersect with the specified triangle, it returns `std::nullopt`
     */
    static std::optional<std::tuple<size_t, Point>>
    getIntersectionPoint(Ray const &ray, MeshTriangleParam const &triangle) { return getIntersectionPointImpl(ray, triangle); }
    static std::optional<std::tuple<size_t, Point>>
    getIntersectionPoint(Ray &&ray, MeshTriangleParam const &triangle) { return getIntersectionPointImpl(std::move(ray), triangle); }
    static std::optional<std::tuple<size_t, Point>>
    getIntersectionPoint(Ray const &ray, MeshTriangleParam &&triangle) { return getIntersectionPointImpl(ray, std::move(triangle)); }
    static std::optional<std::tuple<size_t, Point>>
    getIntersectionPoint(Ray &&ray, MeshTriangleParam &&triangle) { return getIntersectionPointImpl(std::move(ray), std::move(triangle)); }

    /**
     * @brief Checker for point inside the tetrahedron.
     * @param point `Point_3` from CGAL.
     * @param tetrahedron `Tetrahedron_3` from CGAL.
     * @return `true` if point within the tetrahedron, otherwise `false`.
     */
    static bool isPointInsideTetrahedron(Point const &point, Tetrahedron const &tetrahedron);

    /**
     * @brief Calculates volume value from the specified mesh file.
     * @param msh_filename The filename of the Gmsh .msh file to parse.
     * @return Volume value.
     */
    static double getVolumeFromTetrahedronMesh(std::string_view msh_filename);

    /**
     * @brief Gets ID of tetrahedrons and corresponding IDs of elements within. Useful for FEM.
     * @param msh_filename Mesh file.
     * @return Map with key = tetrahedron's ID, value = list of nodes inside.
     */
    static std::map<size_t, std::vector<size_t>> getTetrahedronNodesMap(std::string_view msh_filename);

    /**
     * @brief Map for global mesh nodes with all neighbour tetrahedrons.
     * @param msh_filename Mesh file.
     * @return Map with key = node ID, value = list of neighbour tetrahedrons to this node.
     */
    static std::map<size_t, std::vector<size_t>> getNodeTetrahedronsMap(std::string_view msh_filename);

    /**
     * @brief Retrieves node coordinates from a mesh file. Useful for visualization and FEM calculations.
     * @param msh_filename The name of the mesh file (.msh) from which node coordinates are extracted.
     * @return A map where the key is the node ID and the value is an array of three elements (x, y, z) representing the coordinates of the node.
     */
    static std::map<size_t, std::array<double, 3>> getTetrahedronNodeCoordinates(std::string_view msh_filename);

    /**
     * @brief Retrieves the boundary nodes of a tetrahedron mesh.
     *
     * @details This function opens a Gmsh file specified by `msh_filename` and retrieves
     *          the nodes associated with the boundary elements (typically triangles in
     *          3D meshes). It filters out unique nodes to identify the actual boundary nodes,
     *          which are essential for various mesh-based calculations and visualizations.
     *
     * @param msh_filename The path to the mesh file in Gmsh format as a string view.
     * @return std::vector<size_t> A vector containing the unique tags of boundary nodes.
     *
     * @throws std::exception Propagates any exceptions thrown by the Gmsh API, which are
     *         caught and handled by printing the error message to the standard error output.
     */
    static std::vector<size_t> getTetrahedronMeshBoundaryNodes(std::string_view msh_filename);

    /**
     * @brief Calculates the geometric centers of all tetrahedrons in a given mesh.
     *
     * @details This function opens a mesh file in Gmsh format specified by `msh_filename` and computes
     *          the geometric centers of each tetrahedron. The center of a tetrahedron is calculated as
     *          the arithmetic mean of its vertices' coordinates. These centers are often used for
     *          various geometric and physical calculations, such as finding the centroid of mass in
     *          finite element analysis or for visualizing properties that vary across the mesh volume.
     *
     * @param msh_filename The path to the mesh file in Gmsh format, represented as a string view.
     *          This file should contain the tetrahedral mesh data from which the tetrahedron centers
     *          will be computed.
     * @return std::map<size_t, std::array<double, 3>> A map where each key is a tetrahedron ID and
     *         the value is an array representing the XYZ coordinates of its geometric center. This map
     *         provides a convenient way to access the center of each tetrahedron by its identifier.
     *
     * @throws std::exception Propagates any exceptions thrown by file handling or the Gmsh API, which
     *         might occur during file opening, reading, or processing. These exceptions are typically
     *         caught and should be handled to avoid crashes and ensure that the error is reported properly.
     */
    static std::map<size_t, std::array<double, 3>> getTetrahedronCenters(std::string_view msh_filename);
};

#endif // !MESH_HPP
