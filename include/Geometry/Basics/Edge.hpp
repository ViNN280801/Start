#ifndef EDGE_HPP
#define EDGE_HPP

#include "Geometry/Basics/BaseTypes.hpp"

/**
 * @brief Edge struct.
 * @details This struct is used to store the edges of the triangles.
 */
struct edge_t
{
    Point p1, p2;

    /**
     * @brief Constructor for edge_t struct.
     * @details This constructor is used to create an edge from two points.
     *          How it works:
     *          - If a < b, then p1 = a and p2 = b.
     *          - If a > b, then p1 = b and p2 = a.
     * @param a The first point of the edge.
     * @param b The second point of the edge.
     */
    edge_t(Point_cref a, Point_cref b);
};

using edge_t_ref = edge_t &;
using edge_t_cref = edge_t const &;
using edge_t_vector = std::vector<edge_t>;

using Edge = edge_t;
using Edge_ref = edge_t_ref;
using Edge_cref = edge_t_cref;
using EdgeVector = edge_t_vector;

/**
 * @brief EdgeHash struct.
 * @details This struct is used to hash the edges of the triangles.
 */
struct edge_hash_t
{
    /**
     * @brief operator() for EdgeHash struct.
     * @details This operator is used to hash the edges of the triangles using the CGAL hash function.
     *          The hash function is a simple XOR of the hashes of the two points of the edge.
     *          May be used in `unordered_map` to store the edges of the triangles.
     * @param e The edge to hash.
     * @return size_t The hash of the edge.
     */
    size_t operator()(edge_t_cref e) const noexcept;
};

using edge_hash_t_ref = edge_hash_t &;
using edge_hash_t_cref = edge_hash_t const &;

using EdgeHash = edge_hash_t;
using EdgeHash_ref = edge_hash_t_ref;
using EdgeHash_cref = edge_hash_t_cref;

/**
 * @brief getTriangleEdges function.
 * @details This function is used to get the edges of the triangles.
 * @param triangle The triangle to get the edges of.
 * @return std::vector<edge_t> The edges of the triangle.
 */
edge_t_vector getTriangleEdges(Triangle_cref triangle) noexcept;

/**
 * @brief operator== for Edge struct.
 * @details This operator is used to compare the edges of the triangles.
 */
bool operator==(edge_t_cref a, edge_t_cref b) noexcept;

#endif // !EDGE_HPP
