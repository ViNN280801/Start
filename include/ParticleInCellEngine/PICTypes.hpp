#ifndef PICTYPES_HPP
#define PICTYPES_HPP

#include <map>
#include <set>

#include "FiniteElementMethod/FEMTypes.hpp"
#include "Geometry/GeometryTypes.hpp"
#include "Particle/Particle.hpp"

/**
 * @brief Map of charge densities for nodes in a mesh.
 * @details This type associates each node ID with its corresponding charge density.
 *          The structure is as follows:
 *          - Key: Node ID (GlobalOrdinal)
 *          - Value: Charge density at the node (double)
 */
using NodeChargeDensitiesMap = std::map<GlobalOrdinal, double>;

/**
 * @brief Tracker for particles inside tetrahedrons over time.
 * @details This type organizes particles by simulation time and tetrahedron ID.
 *          The structure is as follows:
 *          - Key: Simulation time (double)
 *          - Value: Map of tetrahedron ID to the particles inside it:
 *            - Key: Tetrahedron ID (size_t)
 *            - Value: Vector of particles inside the tetrahedron (ParticleVector)
 */
using ParticleTrackerMap = std::map<double, std::map<size_t, ParticleVector>>;

/**
 * @brief Set of particle IDs that have settled on a 2D mesh.
 * @details This type stores the IDs of particles that are considered settled
 *          after colliding or interacting with the 2D mesh.
 *          - Element: Particle ID (size_t)
 */
using ParticlesIDSet = std::set<size_t>;

/**
 * @brief Map of the boundary conditions.
 * @details This type associates each node ID with its corresponding boundary condition value.
 *          The structure is as follows:
 *          - Key: Node ID (GlobalOrdinal)
 *          - Value: Value of the boundary conditions (double)
 */
using BoundaryConditionsMap = std::map<GlobalOrdinal, double>;

/**
 * @typedef ParticleMovementMap
 * @brief Tracks the movement of particles across a simulation.
 *
 * This type maps particle IDs to a vector of points representing their movement over time.
 *
 * - Key: Particle ID (size_t)
 * - Value: Vector of points representing the particle's trajectory (std::vector<Point>)
 */
using ParticleMovementMap = std::map<size_t, std::vector<Point>>;

/**
 * @typedef SettledParticlesCounterMap
 * @brief Tracks the number of particles settled on each triangle.
 *
 * This type maps triangle IDs to a counter of the number of particles settled on the triangle.
 *
 * - Key: Triangle ID (size_t)
 * - Value: Count of settled particles (size_t)
 */
using SettledParticlesCounterMap = std::map<size_t, size_t>;

#endif // !PICTYPES_HPP