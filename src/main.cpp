#include "ModelingMainDriver.hpp"

#include "SputteringModel/SputteringModel.hpp"

int main(int argc, char *argv[])
{
    // // Initialize MPI and Kokkos using Tpetra::ScopeGuard.
    // Tpetra::ScopeGuard tpetraScope(std::addressof(argc), std::addressof(argv));

    // if (argc != 2)
    // {
    //     ERRMSG(util::stringify("Usage: ", argv[0], " <config_file>"));
    //     return EXIT_FAILURE;
    // }

    // {
    //     // All Tpetra and Kokkos objects must be created and destroyed within this scope.
    //     ModelingMainDriver modeling(argv[1]);
    //     modeling.startModeling();
    // }

    // // Kokkos and MPI are finalized automatically when tpetraScope goes out of scope.
    // return EXIT_SUCCESS;

    SputteringModel sm;
    sm.start();

    return EXIT_SUCCESS;
}
