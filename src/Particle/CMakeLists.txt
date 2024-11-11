set(PARTICLES_SOURCES
    Particle.cpp
)

set(LIB_PARTICLE Particle)
add_library(${LIB_PARTICLE} ${PARTICLES_SOURCES})

target_link_libraries(${LIB_PARTICLE} PRIVATE
    ${Boost_LIBRARIES}
    Generators
)

target_include_directories(${LIB_PARTICLE} PRIVATE
    ${Boost_INCLUDE_DIRS}
)
