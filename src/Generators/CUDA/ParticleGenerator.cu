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
                                                         double energy_eV, int type, size_t numCells,
                                                         unsigned long long seed)
{
    // GUI sends energy in eV, so, we need to convert it from eV to J:
    double energy_J = util::convert_energy_eV_to_energy_J(energy_eV);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    size_t cellIdx = idx % numCells;
    particles[idx].id = idx;
    particles[idx].type = type;
    particles[idx].x = cellCenters[cellIdx].x;
    particles[idx].y = cellCenters[cellIdx].y;
    particles[idx].z = cellCenters[cellIdx].z;

    // Calculate angles from normals for direction
    double theta = acos(normals[cellIdx].z / sqrt(normals[cellIdx].x * normals[cellIdx].x +
                                                  normals[cellIdx].y * normals[cellIdx].y +
                                                  normals[cellIdx].z * normals[cellIdx].z));
    double phi = atan2(normals[cellIdx].y, normals[cellIdx].x);

    double vx = energy_J * sin(theta) * cos(phi);
    double vy = energy_J * sin(theta) * sin(phi);
    double vz = energy_J * cos(theta);

    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;
    particles[idx].energy = energy_J;
}

START_PARTICLE_VECTOR ParticleGenerator::fromSurfaceSource(const std::vector<surface_source_t> &source)
{
    if (source.empty())
        throw std::logic_error("Surface source list is empty");

    size_t totalParticles = 0;
    size_t totalCells = 0;
    for (const auto &sourceData : source)
    {
        if (sourceData.count == 0)
            throw std::logic_error("Cannot generate 0 particles from a surface source");
        totalParticles += sourceData.count;
        totalCells += sourceData.baseCoordinates.size();
    }

    ParticleDeviceArray deviceParticles;
    deviceParticles.resize(totalParticles);

    std::vector<double3> cellCenters, normals;
    for (const auto &sourceData : source)
    {
        for (const auto &item : sourceData.baseCoordinates)
        {
            std::istringstream iss(item.first);
            double x, y, z;
            iss >> x >> y >> z;
            double3 center = make_double3(x, y, z);

            cellCenters.push_back(center);
            normals.push_back(make_double3(item.second[0], item.second[1], item.second[2]));
        }
    }

    double3 *d_cellCenters = nullptr;
    double3 *d_normals = nullptr;
    cuda_utils::check_cuda_err(cudaMalloc(&d_cellCenters, cellCenters.size() * sizeof(double3)), "Failed to allocate cell centers");
    cuda_utils::check_cuda_err(cudaMalloc(&d_normals, normals.size() * sizeof(double3)), "Failed to allocate normals");

    cudaMemcpy(d_cellCenters, cellCenters.data(), cellCenters.size() * sizeof(double3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_normals, normals.data(), normals.size() * sizeof(double3), cudaMemcpyHostToDevice);

    size_t particleOffset = 0;
    for (const auto &sourceData : source)
    {
        size_t count = sourceData.count;
        size_t numCells = sourceData.baseCoordinates.size();
        double energy_eV = sourceData.energy;
        unsigned long long seed = 123'456'789ull;

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        generateParticlesFromSurfaceSourceKernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceParticles.begin() + particleOffset, count, d_cellCenters, d_normals,
            energy_eV, static_cast<int>(util::getParticleTypeFromStrRepresentation(sourceData.type)), numCells, seed);

        cuda_utils::check_cuda_err(cudaGetLastError(), "Error during surface source particle generation");
        particleOffset += count;
    }

    cuda_utils::check_cuda_err(cudaDeviceSynchronize(), "Failed to synchronize after surface source generation");

    cudaFree(d_cellCenters);
    cudaFree(d_normals);

    return deviceParticles;
}

#endif // !USE_CUDA
