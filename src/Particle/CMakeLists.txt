# src/Particle/CMakeLists.txt

set(PARTICLES_SOURCES
    Particle.cpp
)

if(START_CUDA_ON)
    set(CUDA_PARTICLES_SOURCES
        CUDA/ParticleDevice.cu
        CUDA/ParticleDeviceMemoryConverter.cu
    )
endif()

set(COLLISION_MODEL_DIR PhysicsCore/CollisionModel)
set(COLLISION_MODEL_SOURCES
    ${COLLISION_MODEL_DIR}/CollisionModelFactory.cpp
    ${COLLISION_MODEL_DIR}/HSModel.cpp
    ${COLLISION_MODEL_DIR}/VHSModel.cpp
    ${COLLISION_MODEL_DIR}/VSSModel.cpp
)

set(LIB_PARTICLE Particle)
add_library(${LIB_PARTICLE} 
    ${PARTICLES_SOURCES}
    ${COLLISION_MODEL_SOURCES}
)

if (START_CUDA_ON)
    target_sources(${LIB_PARTICLE} PRIVATE ${CUDA_PARTICLES_SOURCES})
endif()
