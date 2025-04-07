#include <CGAL/squared_distance_3.h>
#include <omp.h>

#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDistributor.hpp"

ParticleDistributor::DistributionResult_t ParticleDistributor::distributeParticles(
    SurfaceMesh &surfaceMesh,
    int particleWeight,
    DistributionType distributionType,
    int neighborLevels)
{
    DistributionResult_t result;
    result.normalizationFactor = 1.0;

    double const weight = std::pow(10, particleWeight);
    auto const &triangleMap = surfaceMesh.getTriangleCellMap();

    std::vector<std::array<double, 3>> realParticles;
    std::unordered_map<size_t, size_t> realCounts;

#pragma omp parallel
    {
        // Thread-safe random number generator
        std::random_device rd;
        std::mt19937 gen(rd() + omp_get_thread_num());

        std::uniform_real_distribution<> uniform_dist(0.0, 1.0);
        std::normal_distribution<> normal_dist(0.0, 0.05);

        std::vector<std::array<double, 3>> local_particles;
        std::unordered_map<size_t, size_t> localCounts;

#pragma omp for schedule(dynamic)
        for (size_t idx = 0; idx < triangleMap.size(); ++idx)
        {
            auto it = std::next(triangleMap.begin(), idx);
            auto const &[id, cell] = *it;

            if (cell.count == 0)
                continue;

            // Get multi-level neighbors
            std::vector<size_t> target_cells = _findMultiLevelNeighbors(surfaceMesh, id, neighborLevels);

            // Calculate average cell size for adaptive sigma
            double avg_cell_size = 0.0;
            double total_area = 0.0;
            for (auto const &cell_id : target_cells)
            {
                auto const &target_cell = triangleMap.at(cell_id);
                total_area += target_cell.area;
            }
            avg_cell_size = std::sqrt(total_area / target_cells.size());
            double sigma = avg_cell_size * 2.0;
            if (sigma < 0.5)
                sigma = 0.5;

            // Calculate weights for each neighboring cell
            auto cell_weights = _calculateWeights(triangleMap, id, target_cells, sigma);

            // Calculate total weight
            double total_weight = 0.0;
            for (auto const &[cell_id, weight_value] : cell_weights)
                total_weight += weight_value;

            // Calculate particles for each cell
            for (auto const &cell_id : target_cells)
            {
                auto const &target_cell = triangleMap.at(cell_id);
                double ratio = cell_weights[cell_id] / total_weight;

                // Calculate number of particles for this cell
                size_t num_particles = std::max(size_t(1),
                                                static_cast<size_t>(std::round(cell.count * weight * ratio)));

#pragma omp critical
                {
                    localCounts[cell_id] += num_particles;
                }

                // Generate particle position
                auto particle_pos = _generateParticlePosition(
                    target_cell,
                    distributionType,
                    gen,
                    uniform_dist,
                    normal_dist);

                local_particles.push_back(particle_pos);
            }
        }

#pragma omp critical
        {
            realParticles.insert(realParticles.end(), local_particles.begin(), local_particles.end());
            for (auto const &[cid, cnt] : localCounts)
                realCounts[cid] += cnt;
        }
    }

    // Calculate normalization factor
    size_t total_model_particles = 0;
    size_t total_real_particles = 0;

    for (auto const &[id, cell] : triangleMap)
        total_model_particles += cell.count;

    for (auto const &[id, count] : realCounts)
        total_real_particles += count;

    // Apply normalization if needed
    if (total_real_particles > 0)
    {
        double total_expected = total_model_particles * weight;
        result.normalizationFactor = total_expected / total_real_particles;

        LOGMSG(util::stringify("Normalization factor for particle distribution: ",
                               result.normalizationFactor, " (expected: ", total_expected,
                               ", generated: ", total_real_particles, ")"));
    }

    result.positions = std::move(realParticles);
    result.cellCounts = std::move(realCounts);

    // Apply normalization to triangle counts
#pragma omp parallel for
    for (size_t idx = 0; idx < triangleMap.size(); ++idx)
    {
        auto it = std::next(triangleMap.begin(), idx);
        auto const &[id, cell] = *it;

        if (result.cellCounts.count(id))
        {
            // Apply normalization to maintain correct total count
            const_cast<TriangleCell &>(cell).count =
                static_cast<size_t>(result.cellCounts[id] * result.normalizationFactor);
        }
    }

    return result;
}

std::vector<size_t> ParticleDistributor::_findMultiLevelNeighbors(
    SurfaceMesh const &surfaceMesh,
    size_t cellId,
    int levels)
{
    // Get multi-level neighbors (neighbors of neighbors)
    std::unordered_set<size_t> expanded_neighbors{cellId};
    std::vector<size_t> current_level = {cellId};

    // For each level, find neighbors of current cells
    for (int level = 0; level < levels; ++level)
    {
        std::vector<size_t> next_level;
        for (auto const &cell_id : current_level)
        {
            auto neighbors = surfaceMesh.getNeighborCells(cell_id);
            for (auto const &n_id : neighbors)
            {
                if (expanded_neighbors.insert(n_id).second)
                {
                    next_level.push_back(n_id);
                }
            }
        }
        current_level = next_level;
        if (current_level.empty())
            break;
    }

    // Convert to vector
    return std::vector<size_t>(expanded_neighbors.begin(), expanded_neighbors.end());
}

std::unordered_map<size_t, double> ParticleDistributor::_calculateWeights(
    std::unordered_map<size_t, TriangleCell> const &triangleMap,
    size_t cellId,
    std::vector<size_t> const &neighborCells,
    double sigma)
{
    // Calculate total area and distance-based weights
    std::unordered_map<size_t, double> cell_weights;

    auto const &sourceCell = triangleMap.at(cellId);
    Point source_centroid = TriangleCell::compute_centroid(sourceCell.triangle);

    for (auto const &cell_id : neighborCells)
    {
        auto const &target_cell = triangleMap.at(cell_id);
        Point target_centroid = TriangleCell::compute_centroid(target_cell.triangle);

        // Calculate distance between centroids
        double distance = std::sqrt(
            std::pow(source_centroid.x() - target_centroid.x(), 2) +
            std::pow(source_centroid.y() - target_centroid.y(), 2) +
            std::pow(source_centroid.z() - target_centroid.z(), 2));

        // Weight based on area and distance (gaussian-like falloff)
        double weight_factor = target_cell.area * std::exp(-(distance * distance) / (2 * sigma * sigma));
        

        cell_weights[cell_id] = weight_factor;
    }

    return cell_weights;
}

std::array<double, 3> ParticleDistributor::_generateParticlePosition(
    TriangleCell const &cell,
    DistributionType dist,
    std::mt19937 &gen,
    std::uniform_real_distribution<> &uniform_dist,
    std::normal_distribution<> &normal_dist)
{
    // Get triangle vertices for proper distribution
    Point v0 = cell.triangle.vertex(0);
    Point v1 = cell.triangle.vertex(1);
    Point v2 = cell.triangle.vertex(2);

    // Generate particles using correct method for distribution
    double r1, r2;
    if (dist == DistributionType::Uniform)
    {
        r1 = uniform_dist(gen);
        r2 = uniform_dist(gen);
    }
    else
    {
        r1 = std::abs(normal_dist(gen));
        r2 = std::abs(normal_dist(gen));
        r1 = std::min(r1, 1.0);
        r2 = std::min(r2, 1.0);
    }

    double u = 1.0 - std::sqrt(r1);
    double v = std::sqrt(r1) * (1.0 - r2);
    double w = 1.0 - u - v;

    // Compute point using barycentric coordinates
    double x = u * v0.x() + v * v1.x() + w * v2.x();
    double y = u * v0.y() + v * v1.y() + w * v2.y();
    double z = u * v0.z() + v * v1.z() + w * v2.z();

    // Add small noise to avoid grid-like patterns
    // Use edge length for better scale estimation
    double e1 = std::sqrt(CGAL::squared_distance(v0, v1));
    double e2 = std::sqrt(CGAL::squared_distance(v1, v2));
    double e3 = std::sqrt(CGAL::squared_distance(v2, v0));
    double avg_edge = (e1 + e2 + e3) / 3.0;
    double noise_scale = avg_edge * 0.05;

    if (dist == DistributionType::Uniform)
    {
        x += (uniform_dist(gen) - 0.5) * noise_scale;
        y += (uniform_dist(gen) - 0.5) * noise_scale;
        z += (uniform_dist(gen) - 0.5) * noise_scale;
    }
    else
    {
        x += normal_dist(gen) * noise_scale;
        y += normal_dist(gen) * noise_scale;
        z += normal_dist(gen) * noise_scale;
    }

    return {x, y, z};
}
