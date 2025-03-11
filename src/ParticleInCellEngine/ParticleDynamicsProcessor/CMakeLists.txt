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
        CUDA/ParticleSurfaceCollisionHandlerCUDA.cu
    )
    list(APPEND PICE_DYNAMIC_SOLVER_SOURCES ${CUDA_PICE_DYNAMIC_SOLVER_SOURCES})

    # Enable relocatable device code for ParticleSurfaceCollisionHandlerCUDA.cu
    # and explicitly specify supported GPU architectures
    set_source_files_properties(
        CUDA/ParticleSurfaceCollisionHandlerCUDA.cu
        PROPERTIES COMPILE_FLAGS "-rdc=true ${CUDA_ARCH_FLAGS}"
    )
    
    # Explicitly disable RDC for the Trilinos file because `desul atomics` 
    # from Trilinos are not supported in RDC mode
    set_source_files_properties(
        CUDA/ParticleDynamicsProcessorCUDA.cu
        PROPERTIES COMPILE_FLAGS "-rdc=false ${CUDA_ARCH_FLAGS}"
    )
endif()

set(LIB_PICE_DYNAMIC_SOLVER ParticleInCellEngine_ParticleDynamicsProcessor)
add_library(${LIB_PICE_DYNAMIC_SOLVER} ${PICE_DYNAMIC_SOLVER_SOURCES})

# Explicitly link against Geometry for CUDA functions
if(GPU_CUDA_ON)
    # We use CUDA_SEPARABLE_COMPILATION OFF because of Trilinos desul atomics issue
    # But we manually set rdc=true for files that need it (ParticleSurfaceCollisionHandlerCUDA.cu)
    set_target_properties(${LIB_PICE_DYNAMIC_SOLVER} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_link_libraries(${LIB_PICE_DYNAMIC_SOLVER} PRIVATE 
        Geometry 
        ${LIB_PARTICLE}
        cudadevrt
    )
else()
    target_link_libraries(${LIB_PICE_DYNAMIC_SOLVER} PRIVATE 
        Geometry 
        ${LIB_PARTICLE}
    )
endif()
