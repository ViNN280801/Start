set(PARTICLES_SOURCES
    Particle.cpp
)

if(GPU_CUDA_ON)
    set(CUDA_PARTICLES_SOURCES
        ParticleKernels.cu
    )
    list(APPEND PARTICLES_SOURCES ${CUDA_PARTICLES_SOURCES})
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
