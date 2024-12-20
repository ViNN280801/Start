#ifndef GEOMETRY_TYPES_HPP
#define GEOMETRY_TYPES_HPP

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <CGAL/intersections.h>

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

#endif // !GEOMETRY_TYPES_HPP
