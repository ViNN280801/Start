#include "ParticleInCellEngine/DynamicSolver/ParticlePhysicsUpdater.hpp"

ParticlePhysicsUpdater::ParticlePhysicsUpdater(std::string_view config_filename)
    : m_config(config_filename),
      m_gasConcentration(util::calculateConcentration_w(config_filename)) {}

void ParticlePhysicsUpdater::doElectroMagneticPush(Particle &particle, std::shared_ptr<GSMAssembler> gsmAssembler, size_t tetrahedronId)
{
    if (auto tetrahedron = gsmAssembler->getMeshManager().getMeshDataByTetrahedronId(tetrahedronId))
    {
        if (tetrahedron->electricField.has_value())
        {
            particle.electroMagneticPush(
                MagneticInduction{}, // Assuming zero magnetic field
                ElectricField(tetrahedron->electricField->x(), tetrahedron->electricField->y(), tetrahedron->electricField->z()),
                m_config.getTimeStep());
        }
    }
}

void ParticlePhysicsUpdater::collideWithGas(Particle &particle)
{
    auto collisionModel{CollisionModelFactory::create(m_config.getScatteringModel())};
    collisionModel->collide(particle, m_config.getGas(), m_gasConcentration, m_config.getTimeStep());
}

void ParticlePhysicsUpdater::updatePosition(Particle &particle) { particle.updatePosition(m_config.getTimeStep()); }
