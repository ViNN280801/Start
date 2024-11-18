set(GENERATORS_SOURCES
    ParticleGenerator.cpp
    RealNumberGenerator.cpp
)

if (GPU_CUDA_ON)
    set(CUDA_GENERATORS_SOURCES
        ParticleGenerator.cu
    )
    list(APPEND GENERATORS_SOURCES ${CUDA_GENERATORS_SOURCES})
endif()

set(LIB_GENERATORS Generators)
add_library(${LIB_GENERATORS} ${GENERATORS_SOURCES})

target_link_libraries(${LIB_GENERATORS} PRIVATE
    ${Boost_LIBRARIES}
)

target_include_directories(${LIB_GENERATORS} PRIVATE
    ${Boost_INCLUDE_DIRS}
)
