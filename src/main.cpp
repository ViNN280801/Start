#include "ModelingMainDriver.hpp"
#include "SputteringModel/TwoPlatesCreator.hpp"

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

    [[maybe_unused]] std::string material("Ti"), gas("Ar"), scattering_model("VSS");
    [[maybe_unused]] double density{4500.0}, molar_mass{0.048}, cell_size = 0.0, mm = 1.0;
    [[maybe_unused]] int grid_type, particle_weight{10};

    std::cout << "Введите тип сетки (1 - равномерная, 2 - неравномерная): ";
    std::cin >> grid_type;
    MeshType mesh_type{grid_type == 1 ? MeshType::Uniform : MeshType::Adaptive};

    std::cout << "Введите желаемый размер стороны ячейки сетки: ";
    std::cin >> cell_size;

    TwoPlatesCreator tpc(mesh_type, cell_size, mm);
    tpc.show();

    // Kokkos and MPI are finalized automatically when tpetraScope goes out of scope.
    return EXIT_SUCCESS;
}
