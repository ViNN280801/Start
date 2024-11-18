# src/Particle/CMakeLists.txt

set(PARTICLES_SOURCES
    Particle.cpp
)

if(GPU_CUDA_ON)
    set(CUDA_PARTICLES_SOURCES
        CUDA/ParticleDevice.cu
        CUDA/ParticleDeviceMemoryConverter.cu
        CUDA/ParticleKernels.cu
    )
endif()

set(LIB_PARTICLE Particle)
add_library(${LIB_PARTICLE} ${PARTICLES_SOURCES})

target_link_libraries(${LIB_PARTICLE} PRIVATE
    ${Boost_LIBRARIES}
    Generators
)

target_include_directories(${LIB_PARTICLE} PRIVATE
    ${Boost_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_sources(${LIB_PARTICLE} PRIVATE ${CUDA_PARTICLES_SOURCES})
endif()
