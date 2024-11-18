# src/Particle/CUDA/CMakeLists.txt

set(CUDA_PARTICLES_SOURCES
    ParticleKernels.cu
    ParticleDeviceMemoryConverter.cu
    ParticleKernels.cu
)

list(APPEND PARTICLES_SOURCES ${CUDA_PARTICLES_SOURCES})
