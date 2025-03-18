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
            START_THROW_EXCEPTION(ParticleGeneratorsInvalidCoordinateException,
                                  util::stringify("Invalid coordinate value in cell center: ", 
                                  cell_centre_str));
        }
    }

    if (cell_centre.size() != 3)
    {
        START_THROW_EXCEPTION(ParticleGeneratorsInvalidNumberOfCoordinatesException, 
                              util::stringify("Invalid number of coordinates in cell center: ", 
                              cell_centre_str));
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
        {
            START_THROW_EXCEPTION(ParticleGeneratorsUnknownParticleTypeException,
                                  util::stringify("Unknown particle type received: ", 
                                  sourceData.type));
        }

        std::array<double, 3> thetaPhi = {sourceData.expansionAngle, sourceData.phi, sourceData.theta};

        if (sourceData.count == 0)
        {
            START_THROW_EXCEPTION(ParticleGeneratorsZeroCountException,
                                  "There is no need to generate 0 objects");
        }

        if (sourceData.energy == 0)
        {
            START_THROW_EXCEPTION(ParticleGeneratorsZeroEnergyException,
                                  util::stringify("Point source with zero energy is used: ", 
                                  sourceData.energy, " eV"));
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

ParticleVector ParticleGeneratorHost::fromSurfaceSource(std::vector<surface_source_t> const &source, double expansionAngle)
{
    ParticleVector particles;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (auto const &sourceData : source)
    {
        ParticleType type{util::getParticleTypeFromStrRepresentation(sourceData.type)};
        if (type == ParticleType::Unknown)
        {
            START_THROW_EXCEPTION(ParticleGeneratorsUnknownParticleTypeException,
                                  util::stringify("Unknown particle type received: ", 
                                  sourceData.type));
        }
        if (sourceData.count == 0)
        {
            START_THROW_EXCEPTION(ParticleGeneratorsZeroCountException,
                                  "There is no need to generate 0 objects");
        }
        if (sourceData.baseCoordinates.empty())
        {
            START_THROW_EXCEPTION(ParticleGeneratorsEmptyBaseCoordinatesException,
                                  "Base coordinates for surface source are empty.");
        }
        if (sourceData.energy == 0)
        {
            START_THROW_EXCEPTION(ParticleGeneratorsZeroEnergyException,
                                  util::stringify("Surface source with zero energy is used: ", 
                                  sourceData.energy, " eV"));
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
                START_THROW_EXCEPTION(ParticleGeneratorsInvalidNormalVectorSizeException,
                                      "Invalid normal vector size. Expected 3 components.");
            }
            double magnitude = std::sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
            if (magnitude == 0.0)
            {
                START_THROW_EXCEPTION(ParticleGeneratorsZeroNormalVectorMagnitudeException,
                                      "Normal vector magnitude is zero, cannot calculate angles.");
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

                std::array<double, 3> thetaPhi = {expansionAngle, phi, theta};
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
