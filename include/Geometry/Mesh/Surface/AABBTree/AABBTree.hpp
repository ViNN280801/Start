#ifndef AABB_TREE_HPP
#define AABB_TREE_HPP

#include "Geometry/Basics/BaseTypes.hpp"

using TriangleVector = std::vector<Triangle>;                                               ///< Vector of triangles.
using TriangleVectorConstIter = TriangleVector::const_iterator;                             ///< Constant iterator for a vector of triangles.
using TrianglePrimitive = CGAL::AABB_triangle_primitive_3<Kernel, TriangleVectorConstIter>; ///< Primitive for representing triangles in an AABB tree for efficient spatial queries.
using TriangleTraits = CGAL::AABB_traits_3<Kernel, TrianglePrimitive>;                      ///< Traits class defining the properties and operations of triangle primitives for use in an AABB tree.
using AABBTree = CGAL::AABB_tree<TriangleTraits>;                                           ///< Axis-Aligned Bounding Box (AABB) tree for accelerating spatial queries (e.g., intersections, nearest neighbors) on triangles.

using TriangleVector_ref = TriangleVector &;
using TriangleVector_cref = TriangleVector const &;
using AABBTree_ref = AABBTree &;
using AABBTree_cref = AABBTree const &;

class AABBTreeConstructionError : public std::runtime_error
{
public:
    AABBTreeConstructionError(std::string_view message) : std::runtime_error(message.data()) {}
};

#endif // !AABB_TREE_HPP
