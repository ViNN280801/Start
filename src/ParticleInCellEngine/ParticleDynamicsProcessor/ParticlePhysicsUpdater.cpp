#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticlePhysicsUpdater.hpp"
#include "Particle/PhysicsCore/CollisionModel/CollisionModelFactory.hpp"

void ParticlePhysicsUpdater::doElectroMagneticPush(Particle &particle, std::shared_ptr<GSMAssembler> gsmAssembler, size_t tetrahedronId, double timeStep) noexcept
{
    if (auto tetrahedron{gsmAssembler->getMeshManager().getMeshDataByTetrahedronId(tetrahedronId)})
    {
        if (tetrahedron->m_electricField.has_value())
        {
            particle.electroMagneticPush(
                MagneticInduction{}, // Assuming zero magnetic field
                ElectricField(tetrahedron->m_electricField->x(),
                              tetrahedron->m_electricField->y(),
                              tetrahedron->m_electricField->z()),
                timeStep);
        }
    }
}

void ParticlePhysicsUpdater::collideWithGas(Particle &particle, std::string_view scatteringModel, std::string_view gasName, double gasConcentration, double timeStep)
{
    auto collisionModel{CollisionModelFactory::create(scatteringModel)};
    collisionModel->collide(particle, util::getParticleTypeFromStrRepresentation(gasName), gasConcentration, timeStep);
}
