set(GENERATORS_SOURCES
    ParticleGenerator.cpp
    RealNumberGenerator.cpp
)

add_library(Generators ${GENERATORS_SOURCES})

target_link_libraries(Generators PRIVATE
    ${Boost_LIBRARIES}
)

target_include_directories(Generators PRIVATE
    ${Boost_INCLUDE_DIRS}
)
target_compile_features(DataHandling PRIVATE cxx_std_20)
