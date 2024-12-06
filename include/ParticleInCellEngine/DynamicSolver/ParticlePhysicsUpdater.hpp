#ifndef PARTICLEPHYSICSUPDATER_HPP
#define PARTICLEPHYSICSUPDATER_HPP

#include <memory>

#include "FiniteElementMethod/GSMAssembler.hpp"
#include "Particle/Particle.hpp"
#include "Particle/PhysicsCore/CollisionModel/CollisionModelFactory.hpp"
#include "Utilities/ConfigParser.hpp"
#include "Utilities/Utilities.hpp"

/**
 * @class ParticlePhysicsUpdater
 * @brief Updates the physical state of particles, including electromagnetic forces and collisions.
 *
 * This class handles particle interactions with electromagnetic fields and gas molecules.
 * It updates particle positions and velocities during the simulation.
 */
class ParticlePhysicsUpdater
{
private:
    ConfigParser m_config;     ///< Configuration parser for simulation settings.
    double m_gasConcentration; ///< Concentration of gas molecules in the simulation.

public:
    /**
     * @brief Constructs a ParticlePhysicsUpdater object.
     * @param config_filename Path to the configuration file.
     */
    ParticlePhysicsUpdater(std::string_view config_filename);

    /**
     * @brief Applies electromagnetic forces to a particle.
     * @param particle Reference to the particle being updated.
     * @param gsmAssembler Shared pointer to the GSM assembler for accessing mesh data.
     * @param tetrahedronId The ID of the tetrahedron containing the particle.
     */
    void doElectroMagneticPush(Particle &particle, std::shared_ptr<GSMAssembler> gsmAssembler, size_t tetrahedronId);

    /**
     * @brief Simulates collisions between a particle and gas molecules.
     * @param particle Reference to the particle undergoing collision.
     */
    void collideWithGas(Particle &particle);

    /**
     * @brief Updates the position of a particle based on its velocity and the simulation timestep.
     * @param particle Reference to the particle being updated.
     */
    void updatePosition(Particle &particle);
};

#endif // PARTICLEPHYSICSUPDATER_HPP
