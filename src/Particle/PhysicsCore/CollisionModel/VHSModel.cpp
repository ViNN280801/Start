#include "Particle/PhysicsCore/CollisionModel/VHSModel.hpp"
#include "Generators/Host/RealNumberGeneratorHost.hpp"
#include "Particle/ParticlePropertiesManager.hpp"
#include "Particle/PhysicsCore/ParticleDynamicUtils.hpp"

bool VHSModel::collide(Particle_ref particle, ParticleType targetType, double n_concentration, double time_step)
{
    double p_radius{particle.getRadius()},
        p_mass{particle.getMass()},
        t_radius{ParticlePropertiesManager::getRadiusFromType(targetType)},
        t_mass{ParticlePropertiesManager::getMassFromType(targetType)};

    double omega{ParticlePropertiesManager::getViscosityTemperatureIndexFromType(targetType)},
        d_reference{(p_radius + t_radius)},
        mass_constant{p_mass * t_mass / (p_mass + t_mass)},
        Exponent{omega - 1. / 2.};

    double p_velocity_module{particle.getVelocityModule()},
        d_vhs_2{(std::pow(d_reference, 2) / std::tgamma(5. / 2. - omega)) *
                std::pow(2 * KT_reference /
                             (mass_constant * p_velocity_module * p_velocity_module),
                         Exponent)};

    double sigma{START_PI_NUMBER * d_vhs_2},
        Probability{sigma * p_velocity_module * n_concentration * time_step};

    RealNumberGeneratorHost rng;
    bool iscolide{rng() < Probability};
    if (iscolide)
    {
        double xi_cos{rng()}, xi_sin{sqrt(1 - xi_cos * xi_cos)},
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
