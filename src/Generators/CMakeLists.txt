# src/Generators/CMakeLists.txt

set(GENERATORS_SOURCES
    ParticleGenerator.cpp
    RealNumberGenerator.cpp
)

if (GPU_CUDA_ON)
    set(CUDA_GENERATORS_SOURCES 
        CUDA/ParticleGenerator.cu
    )
endif()

set(LIB_GENERATORS Generators)
add_library(${LIB_GENERATORS} ${GENERATORS_SOURCES})

target_link_libraries(${LIB_GENERATORS} PRIVATE
    ${Boost_LIBRARIES}
)

target_include_directories(${LIB_GENERATORS} PRIVATE
    ${Boost_INCLUDE_DIRS}
)

if (GPU_CUDA_ON)
    target_sources(${LIB_GENERATORS} PRIVATE ${CUDA_GENERATORS_SOURCES})
endif()
