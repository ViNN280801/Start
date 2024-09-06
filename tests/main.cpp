#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "FiniteElementMethod/FEMTypes.hpp"

int main(int argc, char **argv)
{
    Teuchos::GlobalMPISession mpiSession(std::addressof(argc), std::addressof(argv));
    if (!Kokkos::is_initialized())
        Kokkos::initialize(argc, argv);

    ::testing::InitGoogleTest(std::addressof(argc), argv);

    int result{RUN_ALL_TESTS()};

    if (Kokkos::is_initialized())
        Kokkos::finalize();
    
    return result;
}
