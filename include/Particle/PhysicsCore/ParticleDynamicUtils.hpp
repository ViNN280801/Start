#ifndef PARTICLEDYNAMICUTILS_HPP
#define PARTICLEDYNAMICUTILS_HPP

#include "Generators/Host/RealNumberGeneratorHost.hpp"
#include "Geometry/MathVector.hpp"
#include "Utilities/Utilities.hpp"

#ifdef USE_CUDA
#include "Generators/CUDA/RealNumberGeneratorDevice.cuh"
#endif

/**
 * @class ParticleDynamicUtils
 * @brief Utility class for performing calculations related to particle dynamics.
 *
 * This class provides static methods for performing common particle dynamics calculations such as:
 * - Computing the velocity vector of a particle based on its energy and directional angles.
 * - Determining the kinetic energy of a particle from its velocity components.
 *
 * The class leverages various physical formulas and conventions to facilitate accurate and efficient computations
 * for simulation purposes. It also integrates seamlessly with the project utilities for energy conversion and random number generation.
 */
class ParticleDynamicUtils
{
public:
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
    START_CUDA_DEVICE static VelocityVector calculateVelocityFromEnergy_eV(
        double &energy_eV,
        double mass,
        std::array<double, 3> const &thetaPhi
#if defined(USE_CUDA) && defined(__CUDA_ARCH__)
        ,
        curandState_t *state = nullptr
#endif
    )
    {
        // GUI sends energy in eV, so, we need to convert it from eV to J:
        util::convert_energy_eV_to_energy_J_inplace(energy_eV);

        auto [thetaUsers, phiCalculated, thetaCalculated] = thetaPhi;
        double randomValue;

#if defined(USE_CUDA) && defined(__CUDA_ARCH__)
        if (state != nullptr) // Device-side random number generation
        {
            RealNumberGeneratorDevice rng(*state);
            randomValue = rng.generate(-1, 1);
        }
        else
        {
            // Fallback for device without state
            randomValue = 1.0; // Default or deterministic fallback value
        }
#else
        static thread_local RealNumberGeneratorHost rng;
        randomValue = rng(-1.0, 1.0);
#endif
        // Assuming that expansion angle (`thetaUsers`) in surface source is 0.
        // Suppressing excessive calculations if `thetaUsers` is 0, but adding additional if-else statement.
        double theta;
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
     * @param[in] vz Z-component of velocity in \f$ m/s \f$.\
     */
    START_CUDA_DEVICE static void calculateEnergyJFromVelocity(double &energy, double mass, double vx, double vy, double vz) noexcept
    {
        double velocityMagnitudeSquared = vx * vx + vy * vy + vz * vz;
        energy = 0.5 * mass * velocityMagnitudeSquared;
    }
};

#endif
