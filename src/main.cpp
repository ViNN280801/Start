#include "../include/ParticleInCell.hpp"

int main(int argc, char *argv[])
{
    // Initializing global MPI session and Kokkos.
    Teuchos::GlobalMPISession mpiSession(std::addressof(argc), std::addressof(argv));
    Kokkos::initialize(argc, argv);

    if (argc != 2)
    {
        ERRMSG(util::stringify("Usage: ", argv[0], " <config_file>"));
        return EXIT_FAILURE;
    }
    ParticleInCell pic(argv[1]);
    pic.startSimulation();

    Kokkos::finalize();
    return EXIT_SUCCESS;
}
