set(PARTICLES_SOURCES
    Particle.cpp
)

add_library(Particle ${PARTICLES_SOURCES})

target_link_libraries(Particle PRIVATE
    ${Boost_LIBRARIES}
    Generators
)

target_include_directories(Particle PRIVATE
    ${Boost_INCLUDE_DIRS}
)
