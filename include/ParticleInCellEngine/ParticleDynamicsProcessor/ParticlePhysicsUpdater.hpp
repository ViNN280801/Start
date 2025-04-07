#ifndef PARTICLE_PHYSICS_UPDATER_HPP
#define PARTICLE_PHYSICS_UPDATER_HPP

#include <memory>

#include "FiniteElementMethod/Assemblers/GSMAssembler.hpp"
#include "Particle/Particle.hpp"
#include "Utilities/Utilities.hpp"

/**
 * @class ParticlePhysicsUpdater
 * @brief Handles the physical updates of particles, including electromagnetic interactions and gas collisions.
 *
 * The `ParticlePhysicsUpdater` class provides static methods to apply **electromagnetic forces**
 * and **particle-gas collisions** to particles in a particle-in-cell (PIC) simulation.
 *
 * @details
 * - **Electromagnetic Push (`doElectroMagneticPush`)**:
 *   - Updates particle velocity based on the **electric field** in a given tetrahedron.
 *   - Assumes a **zero magnetic field** unless explicitly provided.
 *   - Uses data retrieved from the **`GSMAssembler` mesh manager**.
 *
 * - **Collisions with Gas (`collideWithGas`)**:
 *   - Simulates collisions between the particle and a background gas using a **specified collision model**.
 *   - Supports **three scattering models**: Hard Sphere (HS), Variable Hard Sphere (VHS), and Variable Soft Sphere (VSS).
 *   - Uses **gas concentration** to determine collision probability.
 *
 * @note This class provides **static utility functions** and does not store any instance variables.
 *
 * @thread_safety
 * - This class does not contain **shared state** and can be used in parallel processing.
 *
 * **Usage Example:**
 * @code
 * std::shared_ptr<GSMAssembler> gsmAssembler = std::make_shared<GSMAssembler>();
 * Particle particle;
 * double timeStep = 1e-6;
 * size_t tetrahedronId = 42;
 *
 * // Apply electromagnetic forces
 * ParticlePhysicsUpdater::doElectroMagneticPush(particle, gsmAssembler, tetrahedronId, timeStep);
 *
 * // Handle gas collisions
 * std::string_view model = "VSS";
 * std::string_view gas = "Ar";
 * double concentration = 1e25; // in particles/m³
 * ParticlePhysicsUpdater::collideWithGas(particle, model, gas, concentration, timeStep);
 * @endcode
 */
class ParticlePhysicsUpdater
{
public:
    /**
     * @brief Applies electromagnetic forces to a particle using electric field data from a given tetrahedron.
     *
     * This method retrieves the **electric field** from the finite element mesh and applies
     * **Lorentz force-based motion** to update the particle's velocity and position.
     * It assumes that the **magnetic field is zero** unless explicitly defined elsewhere.
     *
     * @param[in,out] particle The `Particle` object to be updated.
     * @param[in] gsmAssembler A shared pointer to the `GSMAssembler`, which provides mesh data.
     * @param[in] tetrahedronId The ID of the tetrahedron containing the particle.
     * @param[in] timeStep The time step used for updating the particle motion.
     *
     * @details
     * **Algorithm:**
     * 1. Retrieve the **electric field vector** from the tetrahedron in `GSMAssembler`.
     * 2. Apply **Lorentz force calculations** to update the particle's velocity.
     * 3. Assume a **zero magnetic field** in the calculations unless otherwise specified.
     *
     * **Example Usage:**
     * @code
     * ParticlePhysicsUpdater::doElectroMagneticPush(particle, gsmAssembler, tetrahedronId, timeStep);
     * @endcode
     *
     * @note If the tetrahedron does not contain an electric field, the function **does nothing**.
     * @warning This function assumes that `gsmAssembler` contains **valid mesh data**.
     */
    static void doElectroMagneticPush(Particle &particle, std::shared_ptr<GSMAssembler> gsmAssembler, size_t tetrahedronId, double timeStep) noexcept;

    /**
     * @brief Simulates a collision between a particle and background gas using a selected collision model.
     *
     * This method computes the probability of a collision occurring and, if successful,
     * updates the particle's velocity according to the **selected gas-scattering model**.
     *
     * @param[in,out] particle The `Particle` object undergoing a collision.
     * @param[in] scatteringModel The scattering model to use (`"HS"`, `"VHS"`, or `"VSS"`).
     * @param[in] gasName The type of background gas (`"Ar"`, `"Ne"`, `"He"`, etc.).
     * @param[in] gasConcentration The concentration of the gas in **particles/m³**.
     * @param[in] timeStep The time step used for computing collision probability.
     *
     * @details
     * **Supported Scattering Models:**
     * - **Hard Sphere (HS)**: Assumes constant collision cross-section.
     * - **Variable Hard Sphere (VHS)**: Uses velocity-dependent cross-section scaling.
     * - **Variable Soft Sphere (VSS)**: Introduces additional angular scattering effects.
     *
     * **Algorithm:**
     * 1. Retrieve the **gas properties** from `ParticlePropertiesManager`.
     * 2. Compute the **collision probability** based on the gas concentration and time step.
     * 3. If a collision occurs:
     *    - Update the particle's velocity using the selected collision model.
     *    - Modify the **energy distribution** of the particle accordingly.
     *
     * **Example Usage:**
     * @code
     * ParticlePhysicsUpdater::collideWithGas(particle, "VHS", "Ar", 1e25, 1e-6);
     * @endcode
     *
     * @throws std::runtime_error If the provided `scatteringModel` is not `"HS"`, `"VHS"`, or `"VSS"`.
     * @note This method **modifies** the velocity of the particle upon collision.
     * @warning The accuracy of the simulation depends on the correct **gas properties** being defined.
     */
    static void collideWithGas(Particle &particle, std::string_view scatteringModel, std::string_view gasName, double gasConcentration, double timeStep);
};

#endif // !PARTICLE_PHYSICS_UPDATER_HPP
