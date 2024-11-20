#ifndef PARTICLEUTILS_HPP
#define PARTICLEUTILS_HPP

#include <atomic>

#include "Generators/CUDA/RealNumberGeneratorDevice.cuh"
#include "Generators/RealNumberGenerator.hpp"
#include "Geometry/MathVector.hpp"
#include "Utilities/CUDA/DeviceUtils.cuh"
#include "Utilities/Utilities.hpp"

using namespace constants;
using namespace particle_types;
using namespace physical_constants;
using namespace viscosity_temperature_index;
using namespace VSS_deflection_parameter;

/**
 * @class ParticleUtils
 * @brief Utility class providing various particle property retrieval functions.
 *
 * The `ParticleUtils` class encapsulates static methods for retrieving specific
 * physical properties of particles based on their type. This class supports fetching
 * essential particle attributes like radius, mass, viscosity temperature index,
 * VSS deflection parameter, and charge. By using enums for particle types, the class
 * provides a structured and efficient approach to querying these properties.
 *
 * This utility class is highly efficient and lightweight, offering only static methods
 * without requiring instantiation. It is suitable for CUDA applications, as indicated
 * by the `START_CUDA_HOST_DEVICE` decorator, ensuring compatibility with both
 * host and device functions for GPU-based computations.
 *
 * ### Key Features
 * - **Radius Retrieval**: Returns the radius (in meters) of a particle based on its type.
 * - **Mass Retrieval**: Returns the mass (in kilograms) of a particle.
 * - **Viscosity Temperature Index**: Provides the viscosity temperature index,
 *   significant in simulation scenarios involving variable viscosities.
 * - **VSS Deflection Parameter**: Useful in VSS (Variable Soft Sphere) models, this
 *   parameter aids in defining particle collision behaviors.
 * - **Charge Data**: Retrieves the electric charge of a particle in Coulombs, critical
 *   for simulations involving ionized particles.
 *
 * ### Design Considerations
 * - The use of an `enum` for particle types ensures type safety, reducing potential
 *   errors when requesting particle properties.
 * - Implements defensive checks for undefined particle types, providing default values
 *   (typically 0 or 0.0) and optional warnings when not running in CUDA.
 * - Compatible with GPU computing; methods are marked for CUDA device compatibility.
 *
 * ### Error Handling
 * - For unsupported particle types, properties are set to default values.
 * - Warning messages (wrapped in `WARNINGMSG`) are conditionally emitted when not
 *   compiled with CUDA support, alerting the user to potential misconfigurations.
 *
 * ### Usage
 * This class is primarily intended for use in physics simulations where particle-specific
 * properties are frequently required. By encapsulating these property retrievals, it
 * abstracts away the need for direct access to constants and minimizes the risk of errors.
 *
 * ### Example
 * ```
 * auto radius = ParticleUtils::getRadiusFromType(ParticleType::Ar);
 * auto mass = ParticleUtils::getMassFromType(ParticleType::Ti);
 * ```
 *
 * @note This class relies on several namespaces such as `constants`, `particle_types`,
 *       and others, which must be included for full functionality.
 */
class ParticleUtils
{
public:
    /**
     * @brief Gets radius from the specified type of the particle.
     * @param type Type of the particle represented as enum.
     * @return Radius of the particle [m].
     */
    START_CUDA_HOST_DEVICE static double getRadiusFromType(ParticleType type)
    {
        switch (type)
        {
        case ParticleType::Ar:
            return Ar_radius;
        case ParticleType::Ne:
            return Ne_radius;
        case ParticleType::He:
            return He_radius;
        case ParticleType::Ti:
            return Ti_radius;
        case ParticleType::Al:
            return Al_radius;
        case ParticleType::Sn:
            return Sn_radius;
        case ParticleType::W:
            return W_radius;
        case ParticleType::Au:
            return Au_radius;
        case ParticleType::Cu:
            return Cu_radius;
        case ParticleType::Ni:
            return Ni_radius;
        case ParticleType::Ag:
            return Ag_radius;
        default:
            return 0;
        }
    }

    /**
     * @brief Gets mass from the specified type of the particle.
     * @param type Type of the particle represented as enum.
     * @return Mass of the particle [kg].
     */
    START_CUDA_HOST_DEVICE static double getMassFromType(ParticleType type)
    {
        switch (type)
        {
        case ParticleType::Ar:
            return Ar_mass;
        case ParticleType::Ne:
            return Ne_mass;
        case ParticleType::He:
            return He_mass;
        case ParticleType::Ti:
            return Ti_mass;
        case ParticleType::Al:
            return Al_mass;
        case ParticleType::Sn:
            return Sn_mass;
        case ParticleType::W:
            return W_mass;
        case ParticleType::Au:
            return Au_mass;
        case ParticleType::Cu:
            return Cu_mass;
        case ParticleType::Ni:
            return Ni_mass;
        case ParticleType::Ag:
            return Ag_mass;
        default:
            return 0;
        }
    }

    /**
     * @brief Gets viscosity temperature index from the specified type of the particle.
     * @param type Type of the particle represented as enum.
     * @return viscosity temperature index of the particle [no measure units].
     */
    START_CUDA_HOST_DEVICE static double getViscosityTemperatureIndexFromType(ParticleType type)
    {
        switch (type)
        {
        case ParticleType::Ar:
            return Ar_VTI;
        case ParticleType::Ne:
            return Ne_VTI;
        case ParticleType::He:
            return He_VTI;
        default:
#ifndef USE_CUDA
            WARNINGMSG("Viscosity temperature index is 0 - it means smth went wrong while simulation with VHS or VSS, or you passed wrong particle type");
#endif
            return 0.0;
        }
    }

    /**
     * @brief Gets VSS deflection parameter from the specified type of the particle.
     * @param type Type of the particle represented as enum.
     * @return VSS deflection parameter of the particle [no measure units].
     */
    START_CUDA_HOST_DEVICE static double getVSSDeflectionParameterFromType(ParticleType type)
    {
        switch (type)
        {
        case ParticleType::Ar:
            return Ar_VSS_TI;
        case ParticleType::Ne:
            return Ne_VSS_TI;
        case ParticleType::He:
            return He_VSS_TI;
        default:
#ifndef USE_CUDA
            WARNINGMSG("VSS deflection parameter is 0 - it means smth went wrong while simulation with VHS or VSS, or you passed wrong particle type");
#endif
            return 0.0;
        }
    }

    /**
     * @brief Gets charge from the specified type of the particle.
     * @param type Type of the particle represented as enum.
     * @return Charge of the particle [C - columbs].
     */
    START_CUDA_HOST_DEVICE static double getChargeFromType(ParticleType type)
    {
        switch (type)
        {
        case ParticleType::Ti:
            return ion_charges_coulombs::Ti_2plus; // By default returning 2 ion Ti.
        case ParticleType::Al:
            return ion_charges_coulombs::Al_3plus;
        case ParticleType::Sn:
            return ion_charges_coulombs::Sn_2plus; // By default returning 2 ion Sn.
        case ParticleType::W:
            return ion_charges_coulombs::W_6plus;
        case ParticleType::Au:
            return ion_charges_coulombs::Au_3plus; // By default returning 3 ion Au.
        case ParticleType::Cu:
            return ion_charges_coulombs::Cu_1plus; // By defaule returning 1 ion Cu.
        case ParticleType::Ni:
            return ion_charges_coulombs::Ni_2plus;
        case ParticleType::Ag:
            return ion_charges_coulombs::Ag_1plus;
        default:
#ifndef USE_CUDA
            WARNINGMSG("Charge of the atom is 0 - it means smth went wrong or you passed unknown particle type, or it's a noble gas");
#endif
            return 0.0;
        }
    }

    /**
     * @brief Gets charge in count of ions from the specified type of the particle.
     * @param type Type of the particle represented as enum.
     * @return Charge of the particle [C - columbs].
     */
    START_CUDA_HOST_DEVICE static double getChargeInIonsFromType(ParticleType type)
    {
        switch (type)
        {
        case ParticleType::Ti:
            return ion_charges::Ti_2plus; // By default returning 2 ion Ti.
        case ParticleType::Al:
            return ion_charges::Al_3plus;
        case ParticleType::Sn:
            return ion_charges::Sn_2plus; // By default returning 2 ion Sn.
        case ParticleType::W:
            return ion_charges::W_6plus;
        case ParticleType::Au:
            return ion_charges::Au_3plus; // By default returning 3 ion Au.
        case ParticleType::Cu:
            return ion_charges::Cu_1plus; // By defaule returning 1 ion Cu.
        case ParticleType::Ni:
            return ion_charges::Ni_2plus;
        case ParticleType::Ag:
            return ion_charges::Ag_1plus;
        default:
#ifndef USE_CUDA
            WARNINGMSG("Charge of the atom is 0 - it means smth went wrong or you passed unknown particle type, or it's a noble gas");
#endif
            return 0.0;
        }
    }

    /**
     * @brief Calculates the velocity vector based on energy (in eV) and directional angles.
     *
     * This function performs the following steps:
     * - Converts the energy from electron volts (eV) to joules (J).
     * - Calculates the velocity magnitude using the formula:
     *   \f$ v = \sqrt{\dfrac{2E}{m}} \f$,
     *   where \f$ E \f$ is the energy in joules and \f$ m \f$ is the mass of the particle.
     * - Generates a random deviation in the polar angle based on the user's expansion angle.
     * - Computes the velocity components (\f$ v_x, v_y, v_z \f$) using spherical coordinates:
     *   \f{eqnarray*}{
     *   v_x &=& v \cdot \sin(\theta) \cdot \cos(\phi), \\
     *   v_y &=& v \cdot \sin(\theta) \cdot \sin(\phi), \\
     *   v_z &=& v \cdot \cos(\theta).
     *   \f}
     *
     * @param[in,out] energy_eV The energy of the particle in eV. This value will be converted to joules (J) in place.
     * @param[in] mass The mass of the particle in kilograms (kg).
     * @param[in] thetaPhi A std::array containing:
     *   - \f$ \theta_{\text{Users}} \f$: User-provided expansion angle.
     *   - \f$ \phi_{\text{Calculated}} \f$: Calculated azimuthal angle.
     *   - \f$ \theta_{\text{Calculated}} \f$: Calculated polar angle.
     * @return The computed `VelocityVector`.
     *
     * @note The energy value `energy_eV` is modified in place, converted to joules.
     */
    START_CUDA_HOST_DEVICE inline static VelocityVector calculateVelocityFromEnergy_eV(
        double &energy_eV,
        double mass,
        std::array<double, 3> const &thetaPhi,
        curandState_t *state = nullptr)
    {
        // GUI sends energy in eV, so, we need to convert it from eV to J:
        util::convert_energy_eV_to_energy_J_inplace(energy_eV);

        double randomValue;
#ifdef __CUDA_ARCH__
        // Device-side random number generation
        if (state != nullptr)
        {
            RealNumberGeneratorDevice rng(*state);
            randomValue = rng.generate(-1, 1);
        }
        else
        {
            // Fallback for device without state
            randomValue = 0.0; // Default or deterministic fallback value
        }
#else
        // Host-side random number generation
        static thread_local RealNumberGenerator host_rng(-1, 1);
        randomValue = host_rng();
#endif

        auto [thetaUsers, phiCalculated, thetaCalculated] = thetaPhi;
        double theta;

        // Assuming that expansion angle (`thetaUsers`) in surface source is 0.
        // Suppressing excessive calculations if `thetaUsers` is 0, but adding additional if-else statement.
        if (thetaUsers != 0.0)
            theta = thetaCalculated + randomValue * thetaUsers;
        else
            theta = thetaCalculated;
        
        double v = std::sqrt(2 * energy_eV / mass);
        double vx = v * std::sin(theta) * std::cos(phiCalculated);
        double vy = v * std::sin(theta) * std::sin(phiCalculated);
        double vz = v * std::cos(theta);

        return VelocityVector(vx, vy, vz);
    }

    /**
     * @brief Calculates the kinetic energy of a particle based on velocity components.
     *
     * The method computes kinetic energy using the formula:
     * \f$ E = \frac{1}{2} m \cdot |v|^2 \f$, where:
     * - \f$ E \f$ is kinetic energy in joules.
     * - \f$ m \f$ is the mass of the particle in kilograms.
     * - \f$ |v| \f$ is the velocity magnitude, calculated as:
     * \f$ |v| = \sqrt{v_x^2 + v_y^2 + v_z^2} \f$.
     *
     * @param[in,out] energy Reference to the energy to update with the calculated value.
     * @param[in] mass Mass of the particle in kilograms.
     * @param[in] vx X-component of velocity in \f$ m/s \f$.
     * @param[in] vy Y-component of velocity in \f$ m/s \f$.
     * @param[in] vz Z-component of velocity in \f$ m/s \f$.
     * @return void
     */
    START_CUDA_HOST_DEVICE static inline void calculateEnergyJFromVelocity(double &energy, double mass, double vx, double vy, double vz) noexcept
    {
        // Compute the velocity magnitude
        double velocityMagnitudeSquared = vx * vx + vy * vy + vz * vz;

        // Calculate and update energy
        energy = 0.5 * mass * velocityMagnitudeSquared;
    }
};

#endif
