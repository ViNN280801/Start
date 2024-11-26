#include "Generators/Host/ParticleGeneratorHost.hpp"

std::vector<double> parseCoordinates(const std::string &cell_centre_str)
{
    std::vector<double> cell_centre;
    std::stringstream ss(cell_centre_str);
    std::string coord_str;

    while (std::getline(ss, coord_str, ','))
    {
        try
        {
            cell_centre.push_back(std::stod(coord_str));
        }
        catch (const std::invalid_argument &)
        {
            throw std::invalid_argument("Invalid coordinate value in cell center: " + cell_centre_str);
        }
    }

    if (cell_centre.size() != 3)
    {
        throw std::invalid_argument("Invalid number of coordinates in cell center: " + cell_centre_str);
    }

    return cell_centre;
}

ParticleVector ParticleGeneratorHost::fromPointSource(std::vector<point_source_t> const &source)
{
    ParticleVector particles;
    for (auto const &sourceData : source)
    {
        ParticleType type{util::getParticleTypeFromStrRepresentation(sourceData.type)};
        if (type == ParticleType::Unknown)
            throw std::invalid_argument("Unknown particle type received");

        std::array<double, 3> thetaPhi = {sourceData.expansionAngle, sourceData.phi, sourceData.theta};

        if (sourceData.count == 0)
            throw std::logic_error("There is no need to generate 0 objects");
        if (sourceData.energy == 0)
        {
            WARNINGMSG("Be careful! Point source with zero energy is used.");
        }

        for (size_t i{}; i < sourceData.count; ++i)
        {
            particles.emplace_back(type,
                                   Point(sourceData.baseCoordinates.at(0), sourceData.baseCoordinates.at(1), sourceData.baseCoordinates.at(2)),
                                   sourceData.energy,
                                   thetaPhi);
        }
    }
    return particles;
}

ParticleVector ParticleGeneratorHost::fromSurfaceSource(std::vector<surface_source_t> const &source)
{
    ParticleVector particles;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (auto const &sourceData : source)
    {
        ParticleType type{util::getParticleTypeFromStrRepresentation(sourceData.type)};
        if (type == ParticleType::Unknown)
            throw std::invalid_argument("Unknown particle type received");
        if (sourceData.count == 0)
            throw std::logic_error("There is no need to generate 0 objects");
        if (sourceData.baseCoordinates.empty())
            throw std::invalid_argument("Base coordinates for surface source are empty.");
        if (sourceData.energy == 0)
        {
            WARNINGMSG("Be careful! Surface source with zero energy is used.");
        }

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
        for (auto const &item : sourceData.baseCoordinates)
        {
            auto const &cell_centre_str{item.first};
            auto const &normal{item.second};

            // Validate and normalize the normal vector.
            if (normal.size() != 3)
            {
                throw std::invalid_argument("Invalid normal vector size. Expected 3 components.");
            }
            double magnitude = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            if (magnitude == 0.0)
            {
                throw std::invalid_argument("Normal vector magnitude is zero, cannot calculate angles.");
            }

            double nx = normal[0] / magnitude;
            double ny = normal[1] / magnitude;
            double nz = normal[2] / magnitude;

            // Parse the cell center coordinates.
            std::vector<double> cell_centre = parseCoordinates(cell_centre_str);

            for (size_t i{}; i < cell_particle_count[cell_index]; ++i)
            {
                // Calculate theta and phi based on the normalized normal vector.
                double theta{std::acos(nz)};
                double phi{std::atan2(ny, nx)};

                std::array<double, 3> thetaPhi = {0, phi, theta}; // Assume no expansion with surface source.
                particles.emplace_back(type,
                                       Point(cell_centre[0], cell_centre[1], cell_centre[2]),
                                       sourceData.energy,
                                       thetaPhi);
            }
            ++cell_index;
        }
    }
    return particles;
}
