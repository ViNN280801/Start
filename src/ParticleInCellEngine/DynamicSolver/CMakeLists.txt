# src/ParticleInCellEngine/DynamicSolver/CMakeLists.txt

set(PICE_DYNAMIC_SOLVER_SOURCES
    DynamicSolver.cpp
    ParticleMovementTracker.cpp
    ParticlePhysicsUpdater.cpp
    ParticleSettler.cpp
    SurfaceCollisionHandler.cpp
)

set(LIB_PICE_DYNAMIC_SOLVER ParticleInCellEngine_DynamicSolver)
add_library(${LIB_PICE_DYNAMIC_SOLVER} ${PICE_DYNAMIC_SOLVER_SOURCES})
