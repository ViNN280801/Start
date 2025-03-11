# src/ParticleInCellEngine/ParticleDynamicsProcessor/CMakeLists.txt

set(PICE_DYNAMIC_SOLVER_SOURCES
    ParticleDynamicsProcessor.cpp
    ParticleMovementTracker.cpp
    ParticlePhysicsUpdater.cpp
    ParticleSettler.cpp
    ParticleSurfaceCollisionHandler.cpp
    StopModelingObserver.cpp
)

if (GPU_CUDA_ON)
    set(CUDA_PICE_DYNAMIC_SOLVER_SOURCES
        CUDA/ParticleDynamicsProcessorCUDA.cu
    )
    list(APPEND PICE_DYNAMIC_SOLVER_SOURCES ${CUDA_PICE_DYNAMIC_SOLVER_SOURCES})
endif()

set(LIB_PICE_DYNAMIC_SOLVER ParticleInCellEngine_ParticleDynamicsProcessor)
add_library(${LIB_PICE_DYNAMIC_SOLVER} ${PICE_DYNAMIC_SOLVER_SOURCES})

target_link_libraries(${LIB_PICE_DYNAMIC_SOLVER} PRIVATE ${LIB_PARTICLE})
