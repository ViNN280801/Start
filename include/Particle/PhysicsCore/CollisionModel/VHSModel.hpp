#ifndef VHSMODEL_HPP
#define VHSMODEL_HPP

#include "CollisionModel.hpp"

/// @brief Collision model implementing the Variable Hard Sphere (HS) collision model.
class VHSModel : public CollisionModel
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
    virtual bool collide(Particle &particle, ParticleType targetType, double n_concentration, double time_step) override;
};

#endif // VHSMODEL_HPP
