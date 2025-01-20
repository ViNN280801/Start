#include "ModelingMainDriver.hpp"

#include "Generators/ParticleGenerator.hpp"
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

    [[maybe_unused]] std::string sputteringMaterialName("Ti"), gasName("Ar"), scatteringModel("VSS");
    [[maybe_unused]] double targetMaterialDensity{4500.0}, targetMaterialMolarMass{0.048}, cellSize = 0.0, mm = 1.0;
    [[maybe_unused]] int meshTypeInt, particleWeight{12};

    std::cout << "Введите тип сетки (1 - равномерная, 2 - неравномерная): ";
    std::cin >> meshTypeInt;
    MeshType meshType{meshTypeInt == 1 ? MeshType::Uniform : MeshType::Adaptive};

    if (meshType == MeshType::Uniform)
    {
        std::cout << "Введите желаемый размер стороны ячейки сетки: ";
        std::cin >> cellSize;
    }

    // 1. Creation of 2 plates.
    TwoPlatesCreator tpc(meshType, cellSize, mm);
    tpc.addPlates();

    // 2. Assigning materials to these plates.
    tpc.assignMaterials("Ti", "Ni");
    std::cout << "Материал подложки (большой пластинки): " << tpc.getMaterial(1) << '\n';
    std::cout << "Материал мишени (маленькой пластинки): " << tpc.getMaterial(2) << '\n';

    // 3. Specify target
    tpc.setTargetSurface();

    // 4. Generating mesh to the .msh file.
    tpc.generateMeshfile();

    // 5. Applying AABB tree to the surface mesh (with triangle cells).
    auto surfaceMeshParams{Mesh::getMeshParams(tpc.getMeshFilename())};
    auto surfaceMeshAABB{constructAABBTreeFromMeshParams(surfaceMeshParams)};
    if (!surfaceMeshAABB)
    {
        ERRMSG("Failed to create AABB Tree for the surface mesh");
    }

    tpc.setTargetSurface();

    std::cout << "Введите плотность материала мишени [кг/м3]: ";
    std::cin >> targetMaterialDensity;

    std::cout << "Введите молярную массу атома материала, из которого состоит мишень [кг/моль]: ";
    std::cin >> targetMaterialMolarMass;

    double S{tpc.calculateTargetSurfaceArea()};
    std::cout << "Площадь плоскости мишени: " << S << " [м2]\n";
    std::cout << "Радиус атома Ti = " << constants::physical_constants::Ti_mass << " [м]\n";
    double N{tpc.calculateCountOfParticlesOnTargetSurface(targetMaterialDensity, targetMaterialMolarMass)};
    std::cout << "Количество атомов на поверхности мишени: " << N << '\n';

    std::cout << "Введите вес 1-й моделируемой частицы: ";
    std::cin >> particleWeight;

    double N_model{N / std::pow(10, particleWeight)};
    std::cout << "Таким образом " << N << " реальных частиц = " << N_model << " модельных частиц\n";

    double t{};
    std::cout << "Введите время симуляции [с]: ";
    std::cin >> t;

    double J_model{N_model / (S * t)};
    std::cout << "Поток модельных частиц: " << J_model << " [N/(м2⋅c)]\n";

    double energy_eV{};
    std::cout << "Введите энергию, вылетающих частиц (не всех, а каждой) [эВ]: ";
    std::cin >> energy_eV;

    auto surfaceSource{tpc.prepareDataForSpawnParticles(N_model, energy_eV)};

    double expansionAngle{};
    std::cout << "Введите угол рассеяния [в градусах]: ";
    std::cin >> expansionAngle;

    // Convert from degrees to radians:
    expansionAngle *= START_PI_NUMBER / 180.0;
    auto particles{ParticleGenerator::fromSurfaceSource({surfaceSource}, expansionAngle)};

    if (particles.empty())
    {
        ERRMSG("Ошибка генерации частиц на поверхности.");
    }
    else
    {
        SUCCESSMSG(util::stringify("Было сгенерировано ", particles.size(), " частиц на поверхности мишени."));
    }

    return EXIT_SUCCESS;
}
