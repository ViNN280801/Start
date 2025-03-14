#ifndef BASE_TYPES_HPP
#define BASE_TYPES_HPP

#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Segment_3.h>
#include <CGAL/intersections.h>

#include "Utilities/Utilities.hpp"

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel; ///< Kernel for exact predicates and inexact constructions.
using Point = Kernel::Point_3;                                      ///< 3D point type.
using Segment = Kernel::Segment_3;                                  ///< 3D line segment type.
using Triangle = Kernel::Triangle_3;                                ///< 3D triangle type.
using Tetrahedron = Kernel::Tetrahedron_3;                          ///< 3D tetrahedron type.

using Point_rref = Point &&;
using Point_ref = Point &;
using Point_cref = Point const &;
using Segment_ref = Segment &;
using Segment_cref = Segment const &;
using Triangle_ref = Triangle &;
using Triangle_cref = Triangle const &;
using Tetrahedron_ref = Tetrahedron &;
using Tetrahedron_cref = Tetrahedron const &;

#endif // !BASE_TYPES_HPP
