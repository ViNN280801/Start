#ifndef PARTICLE_DISTRIBUTOR_HPP
#define PARTICLE_DISTRIBUTOR_HPP

#include "Geometry/Mesh/Surface/SurfaceMesh.hpp"
#include "Utilities/Constants.hpp"
#include "Utilities/PreprocessorUtils.hpp"

/**
 * @brief Class responsible for distributing particles across a surface mesh
 *
 * This class handles the distribution of particles across a surface mesh based on
 * the count of settled particles. It supports both uniform and Gaussian distribution
 * methods and uses multi-level neighborhood analysis for realistic distribution.
 */
class ParticleDistributor
{
public:
    /**
     * @brief Distribution type for particle placement
     */
    enum class DistributionType
    {
        Uniform = 1, ///< Uniform distribution of particles
        Gaussian = 2 ///< Gaussian distribution with distance falloff
    };

    /**
     * @brief Result structure for particle distribution operation
     */
    struct DistributionResult_t
    {
        std::vector<std::array<double, 3>> positions;  ///< 3D positions of generated particles
        std::unordered_map<size_t, size_t> cellCounts; ///< Map of cell IDs to particle counts
        double normalizationFactor;                    ///< Factor used to normalize counts
    };

    /**
     * @brief Distributes particles across the mesh based on settled particle counts
     *
     * @param surfaceMesh The mesh containing settled particles
     * @param particleWeight Weight factor for each model particle
     * @param distributionType Type of distribution to use
     * @param neighborLevels Number of neighbor levels to consider for distribution
     * @return DistributionResult_t The generated particles and count information
     *
     * @details Algorithm:
     *  1. For each cell with settled particles:
     *     a. Find multi-level neighbors
     *     b. Calculate weight factors based on area and distance
     *     c. Distribute particles to each cell based on weights
     *     d. Generate positions using barycentric coordinates
     *  2. Apply normalization to maintain correct total particle count
     */
    static DistributionResult_t distributeParticles(
        SurfaceMesh &surfaceMesh,
        int particleWeight,
        DistributionType distributionType = DistributionType::Gaussian,
        int neighborLevels = 4);

private:
    /**
     * @brief Finds multi-level neighbors of a cell
     *
     * @param surfaceMesh The surface mesh
     * @param cellId The ID of the cell to find neighbors for
     * @param levels Number of levels to search
     * @return std::vector<size_t> Vector of neighbor cell IDs
     */
    static std::vector<size_t> _findMultiLevelNeighbors(
        SurfaceMesh const &surfaceMesh,
        size_t cellId,
        int levels);

    /**
     * @brief Calculates weights for neighbor cells based on distance and area
     *
     * @param triangleMap Map of triangle cells
     * @param cellId Source cell ID
     * @param neighborCells Vector of neighbor cell IDs
     * @param sigma Standard deviation for Gaussian falloff
     * @return std::unordered_map<size_t, double> Map of cell IDs to weight factors
     */
    static std::unordered_map<size_t, double> _calculateWeights(
        std::unordered_map<size_t, TriangleCell> const &triangleMap,
        size_t cellId,
        std::vector<size_t> const &neighborCells,
        double sigma);

    /**
     * @brief Generates a particle position within a triangle using barycentric coordinates
     *
     * @param cell The triangle cell
     * @param dist Distribution type to use
     * @param gen Random number generator
     * @param uniform_dist Uniform distribution
     * @param normal_dist Normal distribution
     * @return std::array<double, 3> The generated 3D position
     */
    static std::array<double, 3> _generateParticlePosition(
        TriangleCell const &cell,
        DistributionType dist,
        std::mt19937 &gen,
        std::uniform_real_distribution<> &uniform_dist,
        std::normal_distribution<> &normal_dist);
};

#endif // !PARTICLE_DISTRIBUTOR_HPP
