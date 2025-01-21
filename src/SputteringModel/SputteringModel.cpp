#include "SputteringModel/SputteringModel.hpp"
#include "Generators/ParticleGenerator.hpp"
#include "Geometry/Mesh.hpp"
#include "SputteringModel/TwoPlatesCreator.hpp"

void SputteringModel::start()
{
    [[maybe_unused]] std::string sputteringMaterialName("Ti"), gasName("Ar"), scatteringModel("VSS");
    [[maybe_unused]] double targetMaterialDensity{4500.0}, targetMaterialMolarMass{0.048}, cellSize = 0.0, mm = 1.0;
    [[maybe_unused]] int meshTypeInt, particleWeight{12};

    // ===================================
    // 1. Generating 3D-model of 2 plates.
    // ===================================
    std::cout << "Введите тип сетки (1 - равномерная, 2 - неравномерная): ";
    std::cin >> meshTypeInt;
    MeshType meshType{meshTypeInt == 1 ? MeshType::Uniform : MeshType::Adaptive};

    if (meshType == MeshType::Uniform)
    {
        std::cout << "Введите желаемый размер стороны ячейки сетки: ";
        std::cin >> cellSize;
    }

    // 1.1 Creation of 2 plates.
    TwoPlatesCreator tpc(meshType, cellSize, mm);
    tpc.addPlates();

    // 1.2. Assigning materials to these plates.
    tpc.assignMaterials("Ti", "Ni");

    // 1.3. Specify target and substrate.
    tpc.setTargetSurface();
    tpc.setSubstrateSurface();

    // 1.4. Generating mesh to the .msh file.
    tpc.generateMeshfile();

    // 1.5. Constructing AABB tree to the surface mesh (with triangle cells).
    auto surfaceMeshParams{Mesh::getMeshParams(tpc.getMeshFilename())};
    auto surfaceMeshAABB{constructAABBTreeFromMeshParams(surfaceMeshParams)};
    if (!surfaceMeshAABB)
    {
        ERRMSG("Failed to create AABB Tree for the surface mesh");
    }

    // ==========================================
    // 2. Spawn particles on the "Target" surface
    // ==========================================
    std::cout << "Введите плотность материала мишени [кг/м3]: ";
    std::cin >> targetMaterialDensity;

    std::cout << "Введите молярную массу атома материала, из которого состоит мишень [кг/моль]: ";
    std::cin >> targetMaterialMolarMass;

    // 2.2. Calculating surface area.
    double S{tpc.calculateTargetSurfaceArea()};
    std::cout << "Площадь плоскости мишени: " << S << " [м2]\n";
    std::cout << "Радиус атома Ti = " << constants::physical_constants::Ti_mass << " [м]\n";

    // 2.3. Calculating real count of particles on this surface.
    double N{tpc.calculateCountOfParticlesOnTargetSurface(targetMaterialDensity, targetMaterialMolarMass)};
    std::cout << "Количество атомов на поверхности мишени: " << N << '\n';

    std::cout << "Введите вес 1-й моделируемой частицы: ";
    std::cin >> particleWeight;

    // 2.4. Calculating model count of particles on this surface.
    double N_model{std::ceil(N / std::pow(10, particleWeight))};
    std::cout << "Таким образом " << N << " реальных частиц = " << N_model << " модельных частиц\n";

    double t{};
    std::cout << "Введите время симуляции [с]: ";
    std::cin >> t;

    // 2.5. Calculating flux of the particles from this surface.
    double J_model{N_model / (S * t)};
    std::cout << "Поток модельных частиц: " << J_model << " [N/(м2⋅c)]\n";

    double energy_eV{};
    std::cout << "Введите энергию, вылетающих частиц (не всех, а каждой) [эВ]: ";
    std::cin >> energy_eV;

    // 2.6. Collecting cell centers and forcing direction [0,0,-1] from target plate to substrate
    //      and building surface source data for properly generating particles.
    auto surfaceSource{tpc.prepareDataForSpawnParticles(N_model, energy_eV)};

    double expansionAngle{};
    std::cout << "Введите угол рассеяния [в градусах]: ";
    std::cin >> expansionAngle;
    expansionAngle *= START_PI_NUMBER / 180.0;

    // 2.7. Generating particles on the target surface.
    auto particles{ParticleGenerator::fromSurfaceSource({surfaceSource}, expansionAngle)};
    if (particles.empty())
    {
        ERRMSG("Ошибка генерации частиц на поверхности.");
    }
    else
    {
        SUCCESSMSG(util::stringify("Было сгенерировано ", particles.size(), " частиц на поверхности мишени."));
    }
}
