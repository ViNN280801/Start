set(GENERATORS_SOURCES
    ParticleGenerator.cpp
    RealNumberGenerator.cpp
)

if (GPU_CUDA_ON)
    list(APPEND GENERATORS_SOURCES ParticleGenerator_CUDA.cu)
endif()

add_library(Generators ${GENERATORS_SOURCES})

target_link_libraries(Generators PRIVATE
    ${Boost_LIBRARIES}
)

target_include_directories(Generators PRIVATE
    ${Boost_INCLUDE_DIRS}
)
