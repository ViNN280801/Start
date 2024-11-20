#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Generators/ParticleGenerator.hpp"
#include "Particle/CUDA/ParticleDevice.cuh"
#include "Particle/CUDA/ParticleDeviceMemoryConverter.cuh"
#include "Particle/ParticleUtils.hpp"
#include "Utilities/CUDA/DeviceUtils.cuh"
#include "Utilities/Utilities.hpp"

__global__ void generateParticlesFromPointSourceKernel(ParticleDevice_t *particles, size_t count,
                                                       double3 position, double energy_eV, int type,
                                                       double expansionAngle, double phiCalculated, double thetaCalculated,
                                                       unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    curandState_t state;
    curand_init(seed, idx, 0, &state);

    // Initialize particle attributes
    particles[idx].id = idx;
    particles[idx].type = type;
    particles[idx].x = position.x;
    particles[idx].y = position.y;
    particles[idx].z = position.z;

    VelocityVector velocity = ParticleUtils::calculateVelocityFromEnergy_eV(energy_eV, // This method inside converts eV to J
                                                                            ParticleUtils::getMassFromType(static_cast<ParticleType>(type)),
                                                                            {expansionAngle, phiCalculated, thetaCalculated},
                                                                            &state);
    particles[idx].vx = velocity.getX();
    particles[idx].vy = velocity.getY();
    particles[idx].vz = velocity.getZ();
    particles[idx].energy = energy_eV; // Putting energy in J, because in prev step it had been
                                       // converted from eV to J. We need energy in J because Particle
                                       // class stores energy in J, to properly recalculate velocities
                                       // and energy.
}

START_PARTICLE_VECTOR ParticleGenerator::fromPointSource(const std::vector<point_source_t> &source)
{
    if (source.empty())
        throw std::logic_error("Point source list is empty");

    size_t totalParticles = 0;
    for (const auto &sourceData : source)
    {
        if (sourceData.count == 0)
            throw std::logic_error("Cannot generate 0 particles from a point source");
        totalParticles += sourceData.count;
    }

    ParticleDeviceArray deviceParticles;
    deviceParticles.resize(totalParticles);

    size_t offset = 0;
    for (const auto &sourceData : source)
    {
        double3 position = {sourceData.baseCoordinates[0], sourceData.baseCoordinates[1], sourceData.baseCoordinates[2]};
        int type = static_cast<int>(util::getParticleTypeFromStrRepresentation(sourceData.type));
        size_t count = sourceData.count;
        double energy_eV = sourceData.energy;
        unsigned long long seed = 123'456'789ull;

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        generateParticlesFromPointSourceKernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceParticles.begin() + offset, count, position, energy_eV, type,
            sourceData.expansionAngle, sourceData.phi, sourceData.theta, seed);

        START_CHECK_CUDA_ERROR(cudaGetLastError(), "Error during point source particle generation");
        offset += count;
    }

    START_CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "Failed to synchronize after point source generation");
    return deviceParticles;
}

__global__ void generateParticlesFromSurfaceSourceKernel(ParticleDevice_t *particles, size_t count,
                                                         double3 *cellCenters, double3 *normals,
                                                         double energy_eV, int type,
                                                         unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    // Initialize curand state for this thread
    curandState_t state;
    curand_init(seed, idx, 0, &state);

    // Set particle properties
    particles[idx].id = idx;
    particles[idx].type = type;
    particles[idx].x = cellCenters[idx].x;
    particles[idx].y = cellCenters[idx].y;
    particles[idx].z = cellCenters[idx].z;

    // Calculate theta and phi from normals
    double3 normal = normals[idx];
    double normal_length = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    double thetaCalculated = acos(normal.z / normal_length);
    double phiCalculated = atan2(normal.y, normal.x);

    // Assuming that there is no no expansion angle for surface source as per CPU code,
    // so, formula: theta = thetaCalculated + random_uniform * thetaUsers; will simplified:
    // theta = thetaCalculated
    VelocityVector velocity = ParticleUtils::calculateVelocityFromEnergy_eV(energy_eV, // This method inside converts eV to J
                                                                            ParticleUtils::getMassFromType(static_cast<ParticleType>(type)),
                                                                            {0, phiCalculated, thetaCalculated}, // Passing expansion angle = 0.
                                                                            &state);
    particles[idx].vx = velocity.getX();
    particles[idx].vy = velocity.getY();
    particles[idx].vz = velocity.getZ();
    particles[idx].energy = energy_eV; // Putting energy in J, because in prev step it had been
                                       // converted from eV to J. We need energy in J because Particle
                                       // class stores energy in J, to properly recalculate velocities
                                       // and energy.
}

START_PARTICLE_VECTOR ParticleGenerator::fromSurfaceSource(const std::vector<surface_source_t> &source)
{
    if (source.empty())
        throw std::logic_error("Surface source list is empty");

    ParticleDeviceArray deviceParticles;
    std::vector<double3> cellCentersExpanded, normalsExpanded;
    size_t totalParticles = 0;

    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto &sourceData : source)
    {
        if (sourceData.count == 0)
            throw std::logic_error("Cannot generate 0 particles from a surface source");

        ParticleType type = util::getParticleTypeFromStrRepresentation(sourceData.type);
        if (type == ParticleType::Unknown)
            throw std::invalid_argument("Unknown particle type received");

        size_t num_cells = sourceData.baseCoordinates.size();
        size_t particles_per_cell = sourceData.count / num_cells;
        size_t remainder_particles_count = sourceData.count % num_cells;

        // Collect keys (cell center strings)
        std::vector<std::string> keys;
        for (const auto &item : sourceData.baseCoordinates)
            keys.emplace_back(item.first);

        // Randomly distribute the remainder particles.
        std::shuffle(keys.begin(), keys.end(), gen);
        std::vector<size_t> cell_particle_count(num_cells, particles_per_cell);
        for (size_t i = 0; i < remainder_particles_count; ++i)
            cell_particle_count[i]++;

        size_t cell_index = 0;
        for (const auto &item : sourceData.baseCoordinates)
        {
            auto const &cell_centre_str = item.first;
            auto const &normal = item.second;

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

            size_t particles_in_cell = cell_particle_count[cell_index];
            for (size_t i = 0; i < particles_in_cell; ++i)
            {
                // Append the cell center and normal for each particle
                double3 center = make_double3(cell_centre[0], cell_centre[1], cell_centre[2]);
                cellCentersExpanded.push_back(center);
                normalsExpanded.push_back(make_double3(normal[0], normal[1], normal[2]));
            }

            cell_index++;
        }

        totalParticles += sourceData.count;
    }

    // Now, cellCentersExpanded and normalsExpanded have totalParticles elements
    deviceParticles.resize(totalParticles);

    double3 *d_cellCenters = nullptr;
    double3 *d_normals = nullptr;
    START_CHECK_CUDA_ERROR(cudaMalloc(&d_cellCenters, cellCentersExpanded.size() * sizeof(double3)), "Failed to allocate cell centers");
    START_CHECK_CUDA_ERROR(cudaMalloc(&d_normals, normalsExpanded.size() * sizeof(double3)), "Failed to allocate normals");

    cudaMemcpy(d_cellCenters, cellCentersExpanded.data(), cellCentersExpanded.size() * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, normalsExpanded.data(), normalsExpanded.size() * sizeof(double3), cudaMemcpyHostToDevice);

    // Since we've expanded the cell centers and normals, we can launch a single kernel
    double energy_eV = source.front().energy; // Assuming same energy for all sources; adjust as needed
    unsigned long long seed = 123'456'789ull;

    int threadsPerBlock = 256;
    int blocksPerGrid = (totalParticles + threadsPerBlock - 1) / threadsPerBlock;

    int type = static_cast<int>(util::getParticleTypeFromStrRepresentation(source.front().type)); // Adjust if types differ

    generateParticlesFromSurfaceSourceKernel<<<blocksPerGrid, threadsPerBlock>>>(
        deviceParticles.begin(), totalParticles, d_cellCenters, d_normals,
        energy_eV, type, seed);

    START_CHECK_CUDA_ERROR(cudaGetLastError(), "Error during surface source particle generation");
    START_CHECK_CUDA_ERROR(cudaDeviceSynchronize(), "Failed to synchronize after surface source generation");

    cudaFree(d_cellCenters);
    cudaFree(d_normals);

    return deviceParticles;
}

#endif // !USE_CUDA
