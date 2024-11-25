#ifndef HSMODEL_HPP
#define HSMODEL_HPP

#include "CollisionModel.hpp"

/// @brief Collision model implementing the Hard Sphere (HS) collision model.
class HSModel : public CollisionModel
{
public:
    /**
     * @brief Performs collision between two particles using the Hard Sphere model.
     * @param particle The particle undergoing collision.
     * @param targetType The type of the target particle with which collision occurs.
     * @param n_concentration Particle concentration.
     * @param time_step Simulation time step.
     * @return true if collision occurred, false otherwise.
     */
    virtual bool collide(Particle &particle, ParticleType target, double n_concentration, double time_step) override;
};

#endif // HSMODEL_HPP
