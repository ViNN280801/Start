# src/ParticleInCellEngine/ParticleDynamicsProcessor/CMakeLists.txt

set(PICE_DYNAMIC_SOLVER_SOURCES
    ParticleDynamicsProcessor.cpp
    ParticleMovementTracker.cpp
    ParticlePhysicsUpdater.cpp
    ParticleSettler.cpp
    ParticleSurfaceCollisionHandler.cpp
)

set(LIB_PICE_DYNAMIC_SOLVER ParticleInCellEngine_ParticleDynamicsProcessor)
add_library(${LIB_PICE_DYNAMIC_SOLVER} ${PICE_DYNAMIC_SOLVER_SOURCES})
