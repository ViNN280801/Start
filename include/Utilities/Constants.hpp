#ifndef CONSTANTS_HPP
#define CONSTANTS_HPP

namespace constants
{
    namespace physical_constants
    {
        static constinit const double R{8.314};                           // [J/k*mol].
        static constinit const double T{300};                             // [K].
        static constexpr const double KT_reference{297.0 * 1.380658e-23}; // KT referevnce value [J/kg].
        static constinit const double N_av{6.22e23};                      // Avogadro number.
        static constinit const double eV_J{1.602176565e-19};              // Conversion factor of eV to J (1 eV is ...  J).
        static constinit const double J_eV{6.242e+18};                    // Conversion factor of J to eV (1 J  is ... eV).

        /*** Weight of particles in [kg]. ***/
        static constinit const double Ar_mass{6.6335209e-26};
        static constinit const double Ne_mass{3.3509177e-26};
        static constinit const double He_mass{6.6464731e-27};
        static constinit const double Ti_mass{7.9485017e-26};
        static constinit const double Al_mass{4.4803831e-26};
        static constinit const double Sn_mass{1.9712258e-25};
        static constinit const double W_mass{8.4590343e-26};
        static constinit const double Au_mass{3.2707137e-25};
        static constinit const double Cu_mass{1.0552061e-25};
        static constinit const double Ni_mass{9.7462675e-26};
        static constinit const double Ag_mass{1.7911901e-25};

        /*** Radii (empirical) of particles in [m]. ***/
        /// @link https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
        static constinit const double Ar_radius{71e-12};
        static constinit const double Ne_radius{160e-12};
        static constinit const double He_radius{120e-12};
        static constinit const double Ti_radius{140e-12};
        static constinit const double Al_radius{125e-12};
        static constinit const double Sn_radius{145e-12};
        static constinit const double W_radius{135e-12};
        static constinit const double Au_radius{135e-12};
        static constinit const double Cu_radius{135e-12};
        static constinit const double Ni_radius{135e-12};
        static constinit const double Ag_radius{160e-12};
    }

    // Book: The DSMC method G. A. Bird Version 1.2; 2013. 286 p. (Table A.1) - omega
    namespace viscosity_temperature_index
    {
        static constinit const float Ar_VTI{0.81};
        static constinit const float Ne_VTI{0.66};
        static constinit const float He_VTI{0.66};
    }

    // Book: The DSMC method G. A. Bird Version 1.2; 2013. 286 p. (Table A.2) - alpha
    namespace VSS_deflection_parameter
    {
        static constinit const float Ar_VSS_TI{1.40};
        static constinit const float Ne_VSS_TI{1.31};
        static constinit const float He_VSS_TI{1.26};
    }

    namespace particle_types
    {
        enum ParticleType
        {
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

    static constinit const double gasConcentrationMinimalValue{1e18};
}

#endif // !CONSTANTS_HPP
