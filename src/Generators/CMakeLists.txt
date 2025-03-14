# src/Generators/CMakeLists.txt

set(LIB_GENERATORS Generators)
set(GENERATORS_SOURCES
    Host/ParticleGeneratorHost.cpp
    Host/RealNumberGeneratorHost.cpp
)

if (START_CUDA_ON)
    set(CUDA_GENERATORS_SOURCES 
        CUDA/ParticleGeneratorDevice.cu
    )
endif()

add_library(${LIB_GENERATORS} ${GENERATORS_SOURCES})

if (START_CUDA_ON)
    target_sources(${LIB_GENERATORS} PRIVATE ${CUDA_GENERATORS_SOURCES})
endif()
