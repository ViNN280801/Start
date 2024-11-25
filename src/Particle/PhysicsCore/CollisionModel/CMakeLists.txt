# src/Particle/PhysicsCore/CollisionModel/CMakeLists.txt

set(PARTICLES_PHYSICS_CORE_COLLISION_MODEL_SOURCES
    CollisionModelFactory.cpp
    HSModel.cpp
    VHSModel.cpp
    VSSModel.cpp
)

set(LIB_PARTICLE_PHYSICS_CORE_COLLISION_MODEL Particle_PhysicsCore_CollisionModel)
add_library(${LIB_PARTICLE_PHYSICS_CORE_COLLISION_MODEL} ${PARTICLES_PHYSICS_CORE_COLLISION_MODEL_SOURCES})
