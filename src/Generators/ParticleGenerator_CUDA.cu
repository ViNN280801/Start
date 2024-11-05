#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Generators/ParticleGenerator.hpp"
#include "Particle/ParticleDevice_CUDA.hpp"
#include "Particle/ParticleMemoryConverter_CUDA.hpp"
#include "Particle/ParticleUtils.hpp"

__global__ void generateParticlesKernel(ParticleDevice *particles, size_t count, int type,
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

    double x = minx + (maxx - minx) * curand_uniform_double(&state);
    double y = miny + (maxy - miny) * curand_uniform_double(&state);
    double z = minz + (maxz - minz) * curand_uniform_double(&state);

    double vx = minvx + (maxvx - minvx) * curand_uniform_double(&state);
    double vy = minvy + (maxvy - minvy) * curand_uniform_double(&state);
    double vz = minvz + (maxvz - minvz) * curand_uniform_double(&state);

    size_t id = idx;

    particles[idx].id = id;
    particles[idx].type = type;
    particles[idx].x = x;
    particles[idx].y = y;
    particles[idx].z = z;
    particles[idx].vx = vx;
    particles[idx].vy = vy;
    particles[idx].vz = vz;

    double mass = ParticleUtils::getMassFromType(static_cast<ParticleType>(type));
    double energy = 0.5 * mass * (vx * vx + vy * vy + vz * vz); // E = 1/2 * mv^2

    particles[idx].energy = energy;
}

ParticleVector ParticleGenerator::byVelocities(size_t count, ParticleType type,
                                               double minx, double miny, double minz,
                                               double maxx, double maxy, double maxz,
                                               double minvx, double minvy, double minvz,
                                               double maxvx, double maxvy, double maxvz)
{
    if (count == 0)
        throw std::logic_error("Cannot generate 0 particles");

    ParticleDevice *d_particles;
    size_t bytes = count * sizeof(ParticleDevice);
    cudaMalloc(&d_particles, bytes);

    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long long seed = seed = seed = 123'456'789ull; // Example seed for reproducibility

    generateParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles, count, static_cast<int>(type), minx, miny, minz,
        maxx, maxy, maxz, minvx, minvy, minvz, maxvx, maxvy, maxvz, seed);

    cudaDeviceSynchronize();

    ParticleDevice *h_particles = new ParticleDevice[count];
    cudaMemcpy(h_particles, d_particles, bytes, cudaMemcpyDeviceToHost);

    ParticleVector particles(count);
    for (size_t i = 0; i < count; ++i)
    {
        particles[i] = DeviceToParticle(h_particles[i]);
    }

    cudaFree(d_particles);
    delete[] h_particles;

    return particles;
}

__global__ void generateFixedParticlesKernel(ParticleDevice *particles, size_t count, int type,
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

ParticleVector ParticleGenerator::byVelocities(size_t count, ParticleType type,
                                               double x, double y, double z,
                                               double vx, double vy, double vz)
{
    if (count == 0)
        throw std::logic_error("Cannot generate 0 particles");

    ParticleDevice *d_particles;
    size_t bytes = count * sizeof(ParticleDevice);
    cudaMalloc(&d_particles, bytes);

    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    generateFixedParticlesKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles, count, static_cast<int>(type), x, y, z, vx, vy, vz);

    cudaDeviceSynchronize();

    ParticleDevice *h_particles = new ParticleDevice[count];
    cudaMemcpy(h_particles, d_particles, bytes, cudaMemcpyDeviceToHost);

    ParticleVector particles(count);
    for (size_t i = 0; i < count; ++i)
    {
        particles[i] = DeviceToParticle(h_particles[i]);
    }

    cudaFree(d_particles);
    delete[] h_particles;

    return particles;
}

__global__ void generateParticlesWithVelocityModuleKernel(ParticleDevice *particles, size_t count, int type,
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

ParticleVector ParticleGenerator::byVelocityModule(size_t count, ParticleType type,
                                                   double x, double y, double z,
                                                   double v, double maxTheta, double maxPhi)
{
    if (count == 0)
        throw std::logic_error("Cannot generate 0 particles");

    ParticleDevice *d_particles;
    size_t bytes = count * sizeof(ParticleDevice);
    cudaMalloc(&d_particles, bytes);

    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;
    unsigned long long seed = seed = seed = 123'456'789ull;

    generateParticlesWithVelocityModuleKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_particles, count, static_cast<int>(type), x, y, z, v, maxTheta, maxPhi, seed);

    cudaDeviceSynchronize();

    ParticleDevice *h_particles = new ParticleDevice[count];
    cudaMemcpy(h_particles, d_particles, bytes, cudaMemcpyDeviceToHost);

    ParticleVector particles(count);
    for (size_t i = 0; i < count; ++i)
    {
        particles[i] = DeviceToParticle(h_particles[i]);
    }

    cudaFree(d_particles);
    delete[] h_particles;

    return particles;
}

__global__ void generateParticlesFromPointSourceKernel(ParticleDevice *particles, size_t count,
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

ParticleVector ParticleGenerator::fromPointSource(std::vector<point_source_t> const &source)
{
    ParticleVector particles;

    for (const auto &sourceData : source)
    {
        ParticleType type = util::getParticleTypeFromStrRepresentation(sourceData.type);
        if (type == ParticleType::Unknown)
            throw std::invalid_argument("Unknown particle type received");

        if (sourceData.count == 0)
            throw std::logic_error("Cannot generate 0 particles");

        double3 position{sourceData.baseCoordinates[0], sourceData.baseCoordinates[1], sourceData.baseCoordinates[2]};
        size_t count = sourceData.count;
        double energy = sourceData.energy;
        unsigned long long seed = seed = seed = 123'456'789ull;

        ParticleDevice *d_particles;
        size_t bytes = count * sizeof(ParticleDevice);
        cudaMalloc(&d_particles, bytes);

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        generateParticlesFromPointSourceKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_particles, count, position, energy, static_cast<int>(type),
            sourceData.expansionAngle, sourceData.phi, sourceData.theta, seed);

        cudaDeviceSynchronize();

        ParticleDevice *h_particles = new ParticleDevice[count];
        cudaMemcpy(h_particles, d_particles, bytes, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < count; ++i)
        {
            particles.emplace_back(type, Point(h_particles[i].x, h_particles[i].y, h_particles[i].z),
                                   h_particles[i].energy, std::array<double, 3>{sourceData.expansionAngle, sourceData.phi, sourceData.theta});
        }

        cudaFree(d_particles);
        delete[] h_particles;
    }

    return particles;
}

__global__ void generateParticlesFromSurfaceSourceKernel(ParticleDevice *particles, size_t count,
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

ParticleVector ParticleGenerator::fromSurfaceSource(std::vector<surface_source_t> const &source)
{
    ParticleVector particles;
    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto &sourceData : source)
    {
        ParticleType type = util::getParticleTypeFromStrRepresentation(sourceData.type);
        if (type == ParticleType::Unknown)
            throw std::invalid_argument("Unknown particle type received");

        if (sourceData.count == 0)
            throw std::logic_error("Cannot generate 0 particles");

        size_t numCells = sourceData.baseCoordinates.size();
        size_t count = sourceData.count;
        double energy = sourceData.energy;
        unsigned long long seed = seed = seed = 123'456'789ull;

        double3 *d_cellCenters, *d_normals;
        ParticleDevice *d_particles;

        size_t cellBytes = numCells * sizeof(double3);
        cudaMalloc(&d_cellCenters, cellBytes);
        cudaMalloc(&d_normals, cellBytes);
        cudaMalloc(&d_particles, count * sizeof(ParticleDevice));

        // Prepare cell centers and normals on the device
        std::vector<double3> cellCenters(numCells), normals(numCells);
        size_t i = 0;
        for (const auto &item : sourceData.baseCoordinates)
        {
            // Parse cell center coordinates
            std::istringstream iss(item.first);
            iss >> cellCenters[i].x >> cellCenters[i].y >> cellCenters[i].z;

            // Set normal vector
            normals[i] = {item.second[0], item.second[1], item.second[2]};
            ++i;
        }
        cudaMemcpy(d_cellCenters, cellCenters.data(), cellBytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_normals, normals.data(), cellBytes, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

        generateParticlesFromSurfaceSourceKernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_particles, count, d_cellCenters, d_normals, energy, static_cast<int>(type), numCells, seed);

        cudaDeviceSynchronize();

        ParticleDevice *h_particles = new ParticleDevice[count];
        cudaMemcpy(h_particles, d_particles, count * sizeof(ParticleDevice), cudaMemcpyDeviceToHost);

        for (size_t j = 0; j < count; ++j)
        {
            size_t cellIdx = j % numCells;
            particles.emplace_back(type, Point(h_particles[j].x, h_particles[j].y, h_particles[j].z),
                                   h_particles[j].energy, std::array<double, 3>{0, atan2(normals[cellIdx].y, normals[cellIdx].x), acos(normals[cellIdx].z)});
        }

        cudaFree(d_particles);
        cudaFree(d_cellCenters);
        cudaFree(d_normals);
        delete[] h_particles;
    }

    return particles;
}

#endif // !USE_CUDA
