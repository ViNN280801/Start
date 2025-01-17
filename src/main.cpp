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

    // // Kokkos and MPI are finalized automatically when tpetraScope goes out of scope.
    // return EXIT_SUCCESS;

    [[maybe_unused]] std::string material("Ti"), gas("Ar"), scattering_model("VSS");
    [[maybe_unused]] double density{4500.0}, molar_mass{0.048}, cell_size = 0.0, mm = 1.0;
    [[maybe_unused]] int grid_type, particle_weight{10};

    std::cout << "Введите тип сетки (1 - равномерная, 2 - неравномерная): ";
    std::cin >> grid_type;
    MeshType mesh_type{grid_type == 1 ? MeshType::Uniform : MeshType::Adaptive};

    if (mesh_type == MeshType::Uniform)
    {
        std::cout << "Введите желаемый размер стороны ячейки сетки: ";
        std::cin >> cell_size;
    }

    // 1. Creation of 2 plates.
    TwoPlatesCreator tpc(mesh_type, cell_size, mm);
    tpc.addPlates();

    // 2. Assigning materials to these plates.
    tpc.assignMaterials("Ti", "Ni");
    std::cout << "Материал подложки (большой пластинки): " << tpc.getMaterial(1) << '\n';
    std::cout << "Материал мишени (маленькой пластинки): " << tpc.getMaterial(2) << '\n';

    // 3. Generating mesh to the .msh file.
    tpc.generateMeshfile();

    // 4. Applying AABB tree to the surface mesh (with triangle cells).
    auto surfaceMeshParams{Mesh::getMeshParams(tpc.getMeshFilename())};
    auto surfaceMeshAABB{constructAABBTreeFromMeshParams(surfaceMeshParams)};
    if (!surfaceMeshAABB)
    {
        ERRMSG("Failed to create AABB Tree for the surface mesh");
    }
    else
    {
        SUCCESSMSG("Created AABB Tree for the surface mesh");
    }

    tpc.show(argc, argv);

    return EXIT_SUCCESS;
}
