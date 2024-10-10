#include "Generators/ParticleGenerator.hpp"
#include "Generators/RealNumberGenerator.hpp"

ParticleVector ParticleGenerator::byVelocities(size_t count, ParticleType type,
                                               double minx, double miny, double minz,
                                               double maxx, double maxy, double maxz,
                                               double minvx, double minvy, double minvz,
                                               double maxvx, double maxvy, double maxvz)
{
    RealNumberGenerator rng;
    return _generate(count, [&]()
                     { return Particle(type,
                                       rng(minx, maxx), rng(miny, maxy), rng(minz, maxz),
                                       rng(minvx, maxvx), rng(minvy, maxvy), rng(minvz, maxvz)); });
}

ParticleVector ParticleGenerator::byVelocities(size_t count, ParticleType type,
                                               double x, double y, double z,
                                               double vx, double vy, double vz)
{
    ParticleVector particles;
    return _generate(count, [&]()
                     { return Particle(type, x, y, z, vx, vy, vz); });
}

ParticleVector ParticleGenerator::byVelocityModule(size_t count, ParticleType type,
                                                   double x, double y, double z,
                                                   double v, double theta, double phi)
{
    RealNumberGenerator rng;
    return _generate(count, [&]()
                     {
        theta = rng(0, theta);
		phi = rng(0, phi);

        double vx{v * sin(theta) * cos(phi)},
			vy{v * sin(theta) * sin(phi)},
			vz{v * cos(theta)};

        return Particle(type, x, y, z, vx, vy, vz); });
}

ParticleVector ParticleGenerator::fromPointSource(std::vector<point_source_t> const &source)
{
    ParticleVector particles;
    for (auto const &sourceData : source)
    {
        std::array<double, 3> thetaPhi = {sourceData.expansionAngle, sourceData.phi, sourceData.theta};
        ParticleType type{util::getParticleTypeFromStrRepresentation(sourceData.type)};

        for (size_t i{}; i < sourceData.count; ++i)
            particles.emplace_back(type,
                                   Point(sourceData.baseCoordinates.at(0), sourceData.baseCoordinates.at(1), sourceData.baseCoordinates.at(2)),
                                   sourceData.energy,
                                   thetaPhi);
    }
    return particles;
}

ParticleVector ParticleGenerator::fromSurfaceSource(std::vector<surface_source_t> const &source)
{
    ParticleVector particles;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (auto const &sourceData : source)
    {
        size_t num_cells{sourceData.baseCoordinates.size()},
            particles_per_cell{sourceData.count / num_cells},
            remainder_particles_count{sourceData.count % num_cells};

        std::vector<std::string> keys;
        for (auto const &item : sourceData.baseCoordinates)
            keys.emplace_back(item.first);

        // Randomly distribute the remainder particles.
        std::shuffle(keys.begin(), keys.end(), gen);
        std::vector<size_t> cell_particle_count(num_cells, particles_per_cell);
        for (size_t i{}; i < remainder_particles_count; ++i)
            cell_particle_count[i]++;

        size_t cell_index{};
        ParticleType type{util::getParticleTypeFromStrRepresentation(sourceData.type)};
        for (auto const &item : sourceData.baseCoordinates)
        {
            auto const &cell_centre_str{item.first};
            auto const &normal{item.second};

            // Parse the cell center coordinates from string to double.
            std::istringstream iss(cell_centre_str);
            std::vector<double> cell_centre;
            double coord;
            while (iss >> coord)
            {
                cell_centre.push_back(coord);
                if (iss.peek() == ',')
                    iss.ignore();
            }

            for (size_t i{}; i < cell_particle_count[cell_index]; ++i)
            {
                // Calculate theta and phi based on the normal.
                double theta{std::acos(normal.at(2) / std::sqrt(normal.at(0) * normal.at(0) + normal.at(1) * normal.at(1) + normal.at(2) * normal.at(2)))},
                    phi{std::atan2(normal.at(1), normal.at(0))};

                std::array<double, 3> thetaPhi = {0, phi, theta}; // Assume that there is no expansion with surface source.
                particles.emplace_back(type,
                                       Point(cell_centre.at(0), cell_centre.at(1), cell_centre.at(2)),
                                       sourceData.energy,
                                       thetaPhi);
            }
            ++cell_index;
        }
    }
    return particles;
}
