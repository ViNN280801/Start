#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

#include "PreprocessorUtils.hpp"

namespace constants
{
    namespace physical_constants
    {
        static STARTCONSTINIT const double R{8.314};                           ///< [J/k*mol].
        static STARTCONSTINIT const double T{300};                             ///< [K].
        static STARTCONSTEXPR const double KT_reference{297.0 * 1.380658e-23}; ///< KT referevnce value [J/kg].
        static STARTCONSTINIT const double N_av{6.22e23};                      ///< Avogadro number.
        static STARTCONSTEXPR const double e_charge{1.602176565e-19};          ///< Charge of the electron.
        static STARTCONSTINIT const double eV_J{1.602176565e-19};              ///< Conversion factor of eV to J (1 eV is ...  J).
        static STARTCONSTINIT const double J_eV{6.242e+18};                    ///< Conversion factor of J to eV (1 J  is ... eV).

        /*** Weight of particles in [kg]. ***/
        static STARTCONSTINIT const double O2_mass{53.1e-27}; // Book: The DSMC method G. A. Bird Version 1.2; 2013. 286 p. (Table A.1) - omega
        static STARTCONSTINIT const double Ar_mass{6.6335209e-26};
        static STARTCONSTINIT const double Ne_mass{3.3509177e-26};
        static STARTCONSTINIT const double He_mass{6.6464731e-27};
        static STARTCONSTINIT const double Ti_mass{7.9485017e-26};
        static STARTCONSTINIT const double Al_mass{4.4803831e-26};
        static STARTCONSTINIT const double Sn_mass{1.9712258e-25};
        static STARTCONSTINIT const double W_mass{8.4590343e-26};
        static STARTCONSTINIT const double Au_mass{3.2707137e-25};
        static STARTCONSTINIT const double Cu_mass{1.0552061e-25};
        static STARTCONSTINIT const double Ni_mass{9.7462675e-26};
        static STARTCONSTINIT const double Ag_mass{1.7911901e-25};

        /*** Radii (empirical) of particles in [m]. ***/
        /// @link https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
        static STARTCONSTINIT const double O2_radius{2.035e-10}; // Book: The DSMC method G. A. Bird Version 1.2; 2013. 286 p. (Table A.1) - omega
        static STARTCONSTINIT const double Ar_radius{71e-12};
        static STARTCONSTINIT const double Ne_radius{160e-12};
        static STARTCONSTINIT const double He_radius{120e-12};
        static STARTCONSTINIT const double Ti_radius{140e-12};
        static STARTCONSTINIT const double Al_radius{125e-12};
        static STARTCONSTINIT const double Sn_radius{145e-12};
        static STARTCONSTINIT const double W_radius{135e-12};
        static STARTCONSTINIT const double Au_radius{135e-12};
        static STARTCONSTINIT const double Cu_radius{135e-12};
        static STARTCONSTINIT const double Ni_radius{135e-12};
        static STARTCONSTINIT const double Ag_radius{160e-12};
    }

    // Book: The DSMC method G. A. Bird Version 1.2; 2013. 286 p. (Table A.1) - omega
    namespace viscosity_temperature_index
    {
        static STARTCONSTINIT const float O2_VTI{0.77};
        static STARTCONSTINIT const float Ar_VTI{0.81};
        static STARTCONSTINIT const float Ne_VTI{0.66};
        static STARTCONSTINIT const float He_VTI{0.66};
    }

    // Book: The DSMC method G. A. Bird Version 1.2; 2013. 286 p. (Table A.2) - alpha
    namespace VSS_deflection_parameter
    {
        static STARTCONSTINIT const float O2_VSS_TI{1.40};
        static STARTCONSTINIT const float Ar_VSS_TI{1.40};
        static STARTCONSTINIT const float Ne_VSS_TI{1.31};
        static STARTCONSTINIT const float He_VSS_TI{1.26};
    }

    namespace particle_types
    {
        enum ParticleType
        {
            O2,
            Ar,
            Ne,
            He,
            Ti,
            Al,
            Sn,
            W,
            Au,
            Cu,
            Ni,
            Ag,
            Unknown
        };
    }

    namespace ion_charges
    {
        static STARTCONSTINIT const int O2_charge{-2}; ///< For O2^2- ion.
        static STARTCONSTINIT const int Ar_charge{0};  ///< Argon is a noble gas and generally does not form ions.
        static STARTCONSTINIT const int Ne_charge{0};  ///< Neon is a noble gas and generally does not form ions.
        static STARTCONSTINIT const int He_charge{0};  ///< Helium is a noble gas and generally does not form ions.

        static STARTCONSTINIT const int Ti_2plus{2}; ///< Titanium typically forms a +2 ion.
        static STARTCONSTINIT const int Ti_4plus{4}; ///< Titanium can also form a +4 ion.
        static STARTCONSTINIT const int Al_3plus{3}; ///< Aluminum typically forms a +3 ion.
        static STARTCONSTINIT const int Sn_2plus{2}; ///< Tin typically forms a +2 ion.
        static STARTCONSTINIT const int Sn_4plus{4}; ///< Tin can also form a +4 ion.
        static STARTCONSTINIT const int W_6plus{6};  ///< Tungsten typically forms a +6 ion.
        static STARTCONSTINIT const int Au_1plus{1}; ///< Gold can also form a +1 ion.
        static STARTCONSTINIT const int Au_3plus{3}; ///< Gold typically forms a +3 ion.
        static STARTCONSTINIT const int Cu_1plus{1}; ///< Copper typically forms a +1 ion.
        static STARTCONSTINIT const int Cu_2plus{2}; ///< Copper can also form a +2 ion.
        static STARTCONSTINIT const int Ni_2plus{2}; ///< Nickel typically forms a +2 ion.
        static STARTCONSTINIT const int Ag_1plus{1}; ///< Silver typically forms a +1 ion.
    }

    namespace ion_charges_coulombs
    {
        static STARTCONSTINIT const double O2_charge{-2 * physical_constants::e_charge};
        static STARTCONSTINIT const double Ar_charge{0.0};
        static STARTCONSTINIT const double Ne_charge{0.0};
        static STARTCONSTINIT const double He_charge{0.0};

        static STARTCONSTEXPR const double Ti_2plus{2 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Ti_4plus{4 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Al_3plus{3 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Sn_2plus{2 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Sn_4plus{4 * physical_constants::e_charge};
        static STARTCONSTEXPR const double W_6plus{6 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Au_1plus{1 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Au_3plus{3 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Cu_1plus{1 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Cu_2plus{2 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Ni_2plus{2 * physical_constants::e_charge};
        static STARTCONSTEXPR const double Ag_1plus{1 * physical_constants::e_charge};
    }

    static STARTCONSTINIT const double gasConcentrationMinimalValue{1e18};
}

#endif // !CONSTANTS_HPP
