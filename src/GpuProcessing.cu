#include <algorithm>
#include <cuda_runtime.h>
#include <stdexcept>

#include "GpuProcessing.hpp"

__global__ void processParticlesKernel(Particle *d_particles, size_t num_particles, CubicGrid *d_cubicGrid,
                                       double *d_nodeChargeDensityMap, double timeStep)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles)
        return;

    Particle &particle = d_particles[idx];

    // Example of processing - update particle position or charge.
    // This is a simplified version; you can add more logic as per your needs.
    auto meshParams = d_cubicGrid->getTetrahedronsByGridIndex(d_cubicGrid->getGridIndexByPosition(particle.getCentre()));

    for (auto const &meshParam : meshParams)
    {
        if (Mesh::isPointInsideTetrahedron(particle.getCentre(), meshParam.tetrahedron))
        {
            // Update particle charge density or position
            // Use atomic operations to safely update shared variables
            atomicAdd(&d_nodeChargeDensityMap[meshParam.globalTetraId], particle.getCharge());
        }
    }

    // Update particle position
    particle.updatePosition(timeStep);
}

void processParticleTrackerOnGPU(std::vector<Particle> &particles, size_t start_index, size_t end_index,
                                 std::shared_ptr<CubicGrid> cubicGrid,
                                 std::shared_ptr<GSMAssemblier> assemblier,
                                 std::map<size_t, double> &nodeChargeDensityMap, double time)
{
    size_t num_particles = end_index - start_index;
    if (num_particles == 0)
        return;

    // Allocate memory on GPU for particles
    Particle *d_particles;
    cudaMalloc(&d_particles, num_particles * sizeof(Particle));

    // Copy particles to the GPU
    cudaMemcpy(d_particles, particles.data() + start_index, num_particles * sizeof(Particle), cudaMemcpyHostToDevice);

    // Copy cubicGrid to the GPU
    CubicGrid *d_cubicGrid;
    cudaMalloc(&d_cubicGrid, sizeof(CubicGrid));
    cudaMemcpy(d_cubicGrid, cubicGrid.get(), sizeof(CubicGrid), cudaMemcpyHostToDevice);

    // Allocate and initialize memory for nodeChargeDensityMap on the GPU
    double *d_nodeChargeDensityMap;
    cudaMalloc(&d_nodeChargeDensityMap, nodeChargeDensityMap.size() * sizeof(double));
    cudaMemset(d_nodeChargeDensityMap, 0, nodeChargeDensityMap.size() * sizeof(double));

    // Define grid and block dimensions for CUDA
    int blockSize = 256;
    int numBlocks = (num_particles + blockSize - 1) / blockSize;

    // Launch the kernel on GPU
    processParticlesKernel<<<numBlocks, blockSize>>>(d_particles, num_particles, d_cubicGrid, d_nodeChargeDensityMap, time);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy back the results from the GPU
    std::vector<double> temp_nodeChargeDensityMap(nodeChargeDensityMap.size());
    cudaMemcpy(temp_nodeChargeDensityMap.data(), d_nodeChargeDensityMap, nodeChargeDensityMap.size() * sizeof(double), cudaMemcpyDeviceToHost);

    // Transfer the GPU results to the host nodeChargeDensityMap
    for (size_t i = 0; i < nodeChargeDensityMap.size(); ++i)
    {
        nodeChargeDensityMap[i] = temp_nodeChargeDensityMap[i];
    }

    // Free GPU memory
    cudaFree(d_particles);
    cudaFree(d_cubicGrid);
    cudaFree(d_nodeChargeDensityMap);
}