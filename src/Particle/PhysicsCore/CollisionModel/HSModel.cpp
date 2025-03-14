#include "Particle/PhysicsCore/CollisionModel/HSModel.hpp"
#include "Generators/Host/RealNumberGeneratorHost.hpp"
#include "Particle/ParticlePropertiesManager.hpp"
#include "Particle/PhysicsCore/ParticleDynamicUtils.hpp"

bool HSModel::collide(Particle_ref particle, ParticleType targetType, double n_concentration, double time_step)
{
    double p_radius{particle.getRadius()},
        p_mass{particle.getMass()},
        t_radius{ParticlePropertiesManager::getRadiusFromType(targetType)},
        t_mass{ParticlePropertiesManager::getMassFromType(targetType)},
        sigma{(START_PI_NUMBER)*std::pow(p_radius + t_radius, 2)};

    // Probability of the scattering
    double Probability{sigma * particle.getVelocityModule() * n_concentration * time_step};

    // Result of the collision: if colide -> change attributes of the particle
    RealNumberGeneratorHost rng;
    bool iscolide{rng() < Probability};
    if (iscolide)
    {
        double xi_cos{rng(-1.0, 1.0)}, xi_sin{sqrt(1 - xi_cos * xi_cos)},
            phi{rng(0, 2 * START_PI_NUMBER)};

        double x{xi_sin * cos(phi)}, y{xi_sin * sin(phi)}, z{xi_cos},
            mass_cp{p_mass / (t_mass + p_mass)},
            mass_ct{t_mass / (t_mass + p_mass)};

        auto velocity{particle.getVelocityVector()};
        VelocityVector cm_vel(velocity * mass_cp), p_vec(mass_ct * velocity);
        double mp{p_vec.module()};
        VelocityVector dir_vector(x * mp, y * mp, z * mp);

        particle.setVelocity(dir_vector + cm_vel);

        // Updating energy after updating velocity:
        double energy;
        ParticleDynamicUtils::calculateEnergyJFromVelocity(energy, p_mass, velocity);
        particle.setEnergy_J(energy);
    }
    return iscolide;
}
