#ifndef PARTICLEUTILS_HPP
#define PARTICLEUTILS_HPP

#include "Utilities/Utilities.hpp"

using namespace constants;
using namespace particle_types;
using namespace physical_constants;
using namespace viscosity_temperature_index;
using namespace VSS_deflection_parameter;

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
};

#endif
