#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Generators/ParticleGenerator.hpp"
#include "Particle/ParticleDevice.cuh"
#include "Particle/ParticleMemoryConverter.cuh"
#include "Particle/ParticleUtils.hpp"
#include "Utilities/DeviceUtils.cuh"

__global__ void generateParticlesKernel(ParticleDevice_t *particles, size_t count, int type,
                                        double minx, double miny, double minz,
                                        double maxx, double maxy, double maxz,
                                        double minvx, double minvy, double minvz,
                                        double maxvx, double maxvy, double maxvz,
                                        unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    curandState_t state;
    curand_init(seed, idx, 0, &state);

    particles[idx].id = idx;
    particles[idx].type = type;

    // Randomized position
    particles[idx].x = minx + (maxx - minx) * curand_uniform_double(&state);
    particles[idx].y = miny + (maxy - miny) * curand_uniform_double(&state);
    particles[idx].z = minz + (maxz - minz) * curand_uniform_double(&state);

    // Randomized velocity
    particles[idx].vx = minvx + (maxvx - minvx) * curand_uniform_double(&state);
    particles[idx].vy = minvy + (maxvy - minvy) * curand_uniform_double(&state);
    particles[idx].vz = minvz + (maxvz - minvz) * curand_uniform_double(&state);

    double mass = ParticleUtils::getMassFromType(static_cast<ParticleType>(type));
    particles[idx].energy = 0.5 * mass * (particles[idx].vx * particles[idx].vx + particles[idx].vy * particles[idx].vy + particles[idx].vz * particles[idx].vz);
}

START_PARTICLE_VECTOR ParticleGenerator::byVelocities(size_t count, ParticleType type,
                                                      double minx, double miny, double minz,
                                                      double maxx, double maxy, double maxz,
                                                      double minvx, double minvy, double minvz,
                                                      double maxvx, double maxvy, double maxvz)
{
    if (count == 0)
        throw std::logic_error("Cannot generate 0 particles");

    ParticleDeviceArray_t deviceParticles;
    deviceParticles.resize(count);

    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    generateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        deviceParticles.begin(), count, static_cast<int>(type), minx, miny, minz,
        maxx, maxy, maxz, minvx, minvy, minvz, maxvx, maxvy, maxvz, 123'456'789ull);

    cuda_utils::check_cuda_err(cudaGetLastError(), "Error during kernel execution: generateParticlesKernel");
    cuda_utils::check_cuda_err(cudaDeviceSynchronize(), "Failed to synchronize device after particle generation");

    return deviceParticles;
}

__global__ void generateFixedParticlesKernel(ParticleDevice_t *particles, size_t count, int type,
                                             double x, double y, double z,
                                             double vx, double vy, double vz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    particles[idx].id = idx;
    particles[idx].type = type;
    particles[idx].x = x;
    particles[idx].y = y;
    particles[idx].z = z;
    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;

    double mass = ParticleUtils::getMassFromType(static_cast<ParticleType>(type));
    particles[idx].energy = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
}

START_PARTICLE_VECTOR ParticleGenerator::byVelocities(size_t count, ParticleType type,
                                                      double x, double y, double z,
                                                      double vx, double vy, double vz)
{
    if (count == 0)
        throw std::logic_error("Cannot generate 0 particles");

    ParticleDeviceArray_t deviceParticles;
    deviceParticles.resize(count);

    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    generateFixedParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        deviceParticles.begin(), count, static_cast<int>(type), x, y, z, vx, vy, vz);

    cuda_utils::check_cuda_err(cudaGetLastError(), "Error during kernel execution: generateFixedParticlesKernel");
    cuda_utils::check_cuda_err(cudaDeviceSynchronize(), "Failed to synchronize device after particle generation");

    return deviceParticles;
}

__global__ void generateParticlesWithVelocityModuleKernel(ParticleDevice_t *particles, size_t count, int type,
                                                          double x, double y, double z,
                                                          double v, double maxTheta, double maxPhi,
                                                          unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count)
        return;

    curandState_t state;
    curand_init(seed, idx, 0, &state);

    double theta = maxTheta * curand_uniform(&state);
    double phi = maxPhi * curand_uniform(&state);

    double vx = v * sin(theta) * cos(phi);
    double vy = v * sin(theta) * sin(phi);
    double vz = v * cos(theta);

    particles[idx].id = idx;
    particles[idx].type = type;
    particles[idx].x = x;
    particles[idx].y = y;
    particles[idx].z = z;
    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;

    double mass = ParticleUtils::getMassFromType(static_cast<ParticleType>(type));
    particles[idx].energy = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
}

START_PARTICLE_VECTOR ParticleGenerator::byVelocityModule(size_t count, ParticleType type,
                                                          double x, double y, double z,
                                                          double v, double maxTheta, double maxPhi)
{
    if (count == 0)
        throw std::logic_error("Cannot generate 0 particles");

    ParticleDeviceArray_t deviceParticles;
    deviceParticles.resize(count);

    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    generateParticlesWithVelocityModuleKernel<<<blocksPerGrid, threadsPerBlock>>>(
        deviceParticles.begin(), count, static_cast<int>(type), x, y, z, v, maxTheta, maxPhi, 123'456'789ull);

    cuda_utils::check_cuda_err(cudaGetLastError(), "Error during kernel execution: generateParticlesWithVelocityModuleKernel");
    cuda_utils::check_cuda_err(cudaDeviceSynchronize(), "Failed to synchronize device after particle generation");

    return deviceParticles;
}

__global__ void generateParticlesFromPointSourceKernel(ParticleDevice_t *particles, size_t count,
                                                       double3 position, double energy, int type,
                                                       double expansionAngle, double phi, double theta,
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

    // Set particle energy and direction based on angles
    double vx = energy * sin(theta) * cos(phi);
    double vy = energy * sin(theta) * sin(phi);
    double vz = energy * cos(theta);

    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;
    particles[idx].energy = energy;
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

    ParticleDeviceArray_t deviceParticles;
    deviceParticles.resize(totalParticles);

    size_t offset = 0;
    for (const auto &sourceData : source)
    {
        double3 position = {sourceData.baseCoordinates[0], sourceData.baseCoordinates[1], sourceData.baseCoordinates[2]};
        int type = static_cast<int>(util::getParticleTypeFromStrRepresentation(sourceData.type));
        size_t count = sourceData.count;
        double energy = sourceData.energy;
        unsigned long long seed = 123'456'789ull;

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        generateParticlesFromPointSourceKernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceParticles.begin() + offset, count, position, energy, type,
            sourceData.expansionAngle, sourceData.phi, sourceData.theta, seed);

        cuda_utils::check_cuda_err(cudaGetLastError(), "Error during point source particle generation");
        offset += count;
    }

    cuda_utils::check_cuda_err(cudaDeviceSynchronize(), "Failed to synchronize after point source generation");

    return deviceParticles;
}

__global__ void generateParticlesFromSurfaceSourceKernel(ParticleDevice_t *particles, size_t count,
                                                         double3 *cellCenters, double3 *normals,
                                                         double energy, int type, size_t numCells,
                                                         unsigned long long seed)
{
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

    double vx = energy * sin(theta) * cos(phi);
    double vy = energy * sin(theta) * sin(phi);
    double vz = energy * cos(theta);

    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;
    particles[idx].energy = energy;
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

    ParticleDeviceArray_t deviceParticles;
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
        double energy = sourceData.energy;
        unsigned long long seed = 123'456'789ull;

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        generateParticlesFromSurfaceSourceKernel<<<blocksPerGrid, threadsPerBlock>>>(
            deviceParticles.begin() + particleOffset, count, d_cellCenters, d_normals,
            energy, static_cast<int>(util::getParticleTypeFromStrRepresentation(sourceData.type)), numCells, seed);

        cuda_utils::check_cuda_err(cudaGetLastError(), "Error during surface source particle generation");
        particleOffset += count;
    }

    cuda_utils::check_cuda_err(cudaDeviceSynchronize(), "Failed to synchronize after surface source generation");

    cudaFree(d_cellCenters);
    cudaFree(d_normals);

    return deviceParticles;
}

#endif // !USE_CUDA
