#ifndef GEOMETRY_TYPES_HPP
#define GEOMETRY_TYPES_HPP

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <CGAL/intersections.h>

#define CGAL_TO_DOUBLE(var) CGAL::to_double(var)

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using Ray = Kernel::Segment_3; // Finite ray - line segment.
using Triangle = Kernel::Triangle_3;
using Tetrahedron = Kernel::Tetrahedron_3;

/**
 * @brief Represents triangle mesh parameters (for surfaces):
 * size_t, double, double, double, double, double, double, double, double, double, double, int.
 * id,  x1,     y1,     z1,     x2,     y2,     z2,     x3,     y3,     z3,     dS,     counter.
 * `counter` is counter of settled objects on specific triangle (defines by its `id` field).
 */
using MeshTriangleParam = std::tuple<size_t, Triangle, double, int>;
using MeshTriangleParamVector = std::vector<MeshTriangleParam>;
using TriangleVector = std::vector<Triangle>;
using TriangleVectorConstIter = TriangleVector::const_iterator;

// Custom property map with CGAL::AABB_triangle_primitive.
using TrianglePrimitive = CGAL::AABB_triangle_primitive_3<Kernel, TriangleVectorConstIter>;
using TriangleTraits = CGAL::AABB_traits_3<Kernel, TrianglePrimitive>;
using AABB_Tree_Triangle = CGAL::AABB_tree<TriangleTraits>;

/**
 * @brief Represents tetrahedron mesh parameters (for volumes):
 * int,     Point,         Point,         Point,          Point,    double.
 * id,  vertex1(x,y,z), vertex2(x,y,z), vertex3(x,y,z),  vertex3(x,y,z),  dV.
 */
using MeshTetrahedronParam = std::tuple<size_t, Tetrahedron, double>;
using MeshTetrahedronParamVector = std::vector<MeshTetrahedronParam>;

#ifdef USE_CUDA
/// @brief Structure representing a 3D point or vector.
struct Vec3Device_t
{
    double x;
    double y;
    double z;

    START_CUDA_HOST_DEVICE Vec3Device_t() : x(0.0), y(0.0), z(0.0) {}
    START_CUDA_HOST_DEVICE Vec3Device_t(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    START_CUDA_HOST_DEVICE Vec3Device_t(const Vec3Device_t &other) = default;
    START_CUDA_HOST_DEVICE Vec3Device_t &operator=(const Vec3Device_t &other) = default;
    START_CUDA_HOST_DEVICE Vec3Device_t(Vec3Device_t &&other) = default;
    START_CUDA_HOST_DEVICE Vec3Device_t &operator=(Vec3Device_t &&other) = default;
};

/// @brief Structure representing a triangle in 3D space.
struct TriangleDevice_t
{
    Vec3Device_t v0; ///< First vertex of the triangle.
    Vec3Device_t v1; ///< Second vertex of the triangle.
    Vec3Device_t v2; ///< Third vertex of the triangle.

    START_CUDA_HOST_DEVICE TriangleDevice_t() {}
    START_CUDA_HOST_DEVICE TriangleDevice_t(const Vec3Device_t &v0_, const Vec3Device_t &v1_, const Vec3Device_t &v2_) : v0(v0_), v1(v1_), v2(v2_) {}
    START_CUDA_HOST_DEVICE TriangleDevice_t(const TriangleDevice_t &other) = default;
    START_CUDA_HOST_DEVICE TriangleDevice_t &operator=(const TriangleDevice_t &other) = default;
    START_CUDA_HOST_DEVICE TriangleDevice_t(TriangleDevice_t &&other) = default;
    START_CUDA_HOST_DEVICE TriangleDevice_t &operator=(TriangleDevice_t &&other) = default;
};
#endif // !USE_CUDA

#endif // !GEOMETRY_TYPES_HPP
