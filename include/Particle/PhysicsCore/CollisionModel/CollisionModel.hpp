#ifndef COLLISION_MODEL_HPP
#define COLLISION_MODEL_HPP

#include "Particle/Particle.hpp"

/// @brief Abstract base class for collision models.
class CollisionModel
{
public:
    /// @brief Virtual destructor.
    virtual ~CollisionModel() = default;

    /**
     * @brief Performs collision between two particles.
     * @param particle The particle undergoing collision.
     * @param targetType The type of the target particle with which collision occurs.
     * @param n_concentration Particle concentration.
     * @param time_step Simulation time step.
     * @return true if collision occurred, false otherwise.
     */
    virtual bool collide(Particle &particle, ParticleType targetType, double n_concentration, double time_step) = 0;
};

#endif // COLLISION_MODEL_HPP
