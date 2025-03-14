# src/ParticleInCellEngine/CMakeLists.txt

set(PARTICLE_DYNAMICS_PROCESSOR_DIR ParticleDynamicsProcessor)
set(PARTICLE_IN_CELL_ENGINE_SOURCES
    ${PARTICLE_DYNAMICS_PROCESSOR_DIR}/ParticleDynamicsProcessor.cpp
    ${PARTICLE_DYNAMICS_PROCESSOR_DIR}/ParticleMovementTracker.cpp
    ${PARTICLE_DYNAMICS_PROCESSOR_DIR}/ParticlePhysicsUpdater.cpp
    ${PARTICLE_DYNAMICS_PROCESSOR_DIR}/ParticleSettler.cpp
    ${PARTICLE_DYNAMICS_PROCESSOR_DIR}/ParticleSurfaceCollisionHandler.cpp
    ${PARTICLE_DYNAMICS_PROCESSOR_DIR}/StopModelingObserver.cpp

    ChargeDensityEquationSolver.cpp
    NodeChargeDensityProcessor.cpp
)

set(LIB_PARTICLE_IN_CELL_ENGINE ParticleInCellEngine)
add_library(${LIB_PARTICLE_IN_CELL_ENGINE} ${PARTICLE_IN_CELL_ENGINE_SOURCES})
