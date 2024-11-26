#include <algorithm>

#include "Particle/PhysicsCore/CollisionModel/CollisionModelFactory.hpp"
#include "Particle/PhysicsCore/CollisionModel/HSModel.hpp"
#include "Particle/PhysicsCore/CollisionModel/VHSModel.hpp"
#include "Particle/PhysicsCore/CollisionModel/VSSModel.hpp"

std::unique_ptr<CollisionModel> CollisionModelFactory::create(CollisionModelType model_type)
{
    switch (model_type)
    {
    case CollisionModelType::HS:
        return std::make_unique<HSModel>();
    case CollisionModelType::VHS:
        return std::make_unique<VHSModel>();
    case CollisionModelType::VSS:
        return std::make_unique<VSSModel>();
    default:
        throw std::invalid_argument("Unknown collision model type.");
    }
}

std::unique_ptr<CollisionModel> CollisionModelFactory::create(std::string_view model_name)
{
    std::string model_str(model_name);
    std::transform(model_str.begin(), model_str.end(), model_str.begin(), ::toupper);

    if (model_str == "HS")
        return create(CollisionModelType::HS);
    else if (model_str == "VHS")
        return create(CollisionModelType::VHS);
    else if (model_str == "VSS")
        return create(CollisionModelType::VSS);
    else
        throw std::invalid_argument("Unknown collision model name.");
}
