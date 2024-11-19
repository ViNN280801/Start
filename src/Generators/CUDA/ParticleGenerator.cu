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

/**
 * @brief Generates a random real number in the specified range [from, to] using curand.
 *
 * This function is compatible with CUDA and uses the curand library for random
 * number generation.
 *
 * @param from Lower bound of the range.
 * @param to Upper bound of the range.
 * @param state curand state initialized in the kernel.
 * @return A random double in the range [from, to].
 */
__device__ double generate_real_number(double from, double to, curandState_t &state)
{
    if (from == to)
        return from;
    if (from > to)
    {
        // Swap to avoid invalid range
        double temp = from;
        from = to;
        to = temp;
    }

    // Generate a uniform random number in [0, 1)
    double randomValue = curand_uniform_double(&state);

    // Scale to [from, to]
    return from + randomValue * (to - from);
}

__global__ void generateParticlesFromPointSourceKernel(ParticleDevice_t *particles, size_t count,
                                                       double3 position, double energy_eV, int type,
                                                       double expansionAngle, double phiCalculated, double thetaCalculated,
                                                       unsigned long long seed)
{
    // GUI sends energy in eV, so, we need to convert it from eV to J:
    double energy_J = util::convert_energy_eV_to_energy_J(energy_eV);

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

    double theta = {thetaCalculated + generate_real_number(-1, 1, state) * expansionAngle};
    double v = sqrt(2 * energy_J / ParticleUtils::getMassFromType(static_cast<ParticleType>(type)));
    double vx = v * sin(theta) * cos(phiCalculated);
    double vy = v * sin(theta) * sin(phiCalculated);
    double vz = v * cos(theta);

    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;
    particles[idx].energy = energy_J;
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

        cuda_utils::check_cuda_err(cudaGetLastError(), "Error during point source particle generation");
        offset += count;
    }

    cuda_utils::check_cuda_err(cudaDeviceSynchronize(), "Failed to synchronize after point source generation");
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

    // Convert energy from eV to J
    double energy_J = util::convert_energy_eV_to_energy_J(energy_eV);

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
    double thetaUsers = 0.0; // No expansion angle for surface source as per CPU code

    // Random number between -1 and 1 (if needed)
    double random_uniform = curand_uniform(&state) * 2.0 - 1.0;

    // Calculate theta (since thetaUsers is 0, theta remains thetaCalculated)
    double theta = thetaCalculated + random_uniform * thetaUsers;

    double v = sqrt(2 * energy_J / ParticleUtils::getMassFromType(static_cast<ParticleType>(type)));
    double vx = v * sin(theta) * cos(phiCalculated);
    double vy = v * sin(theta) * sin(phiCalculated);
    double vz = v * cos(theta);

    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;
    particles[idx].energy = energy_J;
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
    cuda_utils::check_cuda_err(cudaMalloc(&d_cellCenters, cellCentersExpanded.size() * sizeof(double3)), "Failed to allocate cell centers");
    cuda_utils::check_cuda_err(cudaMalloc(&d_normals, normalsExpanded.size() * sizeof(double3)), "Failed to allocate normals");

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

    cuda_utils::check_cuda_err(cudaGetLastError(), "Error during surface source particle generation");
    cuda_utils::check_cuda_err(cudaDeviceSynchronize(), "Failed to synchronize after surface source generation");

    cudaFree(d_cellCenters);
    cudaFree(d_normals);

    return deviceParticles;
}

#endif // !USE_CUDA
