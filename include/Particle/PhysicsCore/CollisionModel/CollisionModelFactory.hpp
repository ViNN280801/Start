#ifndef COLLISIONMODELFACTORY_HPP
#define COLLISIONMODELFACTORY_HPP

#include <memory>
#include <string_view>

#include "CollisionModel.hpp"
#include "CollisionModelType.hpp"

/// @brief Factory class for creating collision model instances.
class CollisionModelFactory
{
public:
    /**
     * @brief Creates a collision model based on the specified type.
     * @param model_type The type of collision model to create.
     * @return A unique_ptr to the created CollisionModel.
     */
    static std::unique_ptr<CollisionModel> create(CollisionModelType model_type);

    /**
     * @brief Creates a collision model based on the specified model name.
     * @param model_name The name of the collision model ("HS", "VHS", "VSS").
     * @return A unique_ptr to the created CollisionModel.
     */
    static std::unique_ptr<CollisionModel> create(std::string_view model_name);
};

#endif // COLLISIONMODELFACTORY_HPP
