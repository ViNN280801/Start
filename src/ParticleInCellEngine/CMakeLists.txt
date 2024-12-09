# src/ParticleInCellEngine/CMakeLists.txt

add_subdirectory(ParticleDynamicsProcessor)

set(PARTICLE_IN_CELL_ENGINE_SOURCES
    ChargeDensityEquationSolver.cpp
    NodeChargeDensityProcessor.cpp
)

set(LIB_PARTICLE_IN_CELL_ENGINE ParticleInCellEngine)
add_library(${LIB_PARTICLE_IN_CELL_ENGINE} ${PARTICLE_IN_CELL_ENGINE_SOURCES})
