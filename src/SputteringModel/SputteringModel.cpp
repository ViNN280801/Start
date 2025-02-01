#include <hdf5.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "DataHandling/TriangleMeshHdf5Manager.hpp"
#include "Generators/ParticleGenerator.hpp"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDynamicsProcessor.hpp"
#include "SputteringModel/SputteringModel.hpp"
#include "SputteringModel/TwoPlatesCreator.hpp"
#include "Utilities/GmshUtilities/GmshUtils.hpp"

std::mutex SputteringModel::m_particlesMovementMutex;
std::shared_mutex SputteringModel::m_settledParticlesMutex;
std::atomic_flag SputteringModel::m_stop_processing = ATOMIC_FLAG_INIT;

enum InputMode
{
    Manual,
    Auto
};

void SputteringModel::_initializeObservers()
{
    m_stopObserver = std::make_shared<StopFlagObserver>(m_stop_processing);
    addObserver(m_stopObserver);
}

void SputteringModel::_ginitialize() { _initializeObservers(); }

void SputteringModel::_updateSurfaceMesh()
{
    // Updating hdf5file to know how many particles settled on certain triangle from the surface mesh.
    std::string hdf5filename(std::string(m_mesh_filename.substr(0ul, m_mesh_filename.find("."))));
    hdf5filename += ".hdf5";
    TriangleMeshHdf5Manager hdf5handler(hdf5filename);
    hdf5handler.saveMeshToHDF5(m_surfaceMesh.getTriangleCellMap());
}

void _readParticlePositionsHdf5(const std::string &filepath = "results/particles_movements.hdf5", int particleID = -1)
{
    try
    {
        // Open the HDF5 file
        hid_t file_id = H5Fopen(filepath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0)
        {
            ERRMSG("Failed to open HDF5 file: " + filepath);
            return;
        }

        // Get the list of all datasets in the file
        hsize_t num_objs;
        H5Gget_num_objs(file_id, &num_objs);

        bool particleFound = false;

        for (hsize_t i = 0; i < num_objs; ++i)
        {
            char dataset_name[1024];
            H5Gget_objname_by_idx(file_id, i, dataset_name, sizeof(dataset_name));

            // If particleID is -1, print all particles
            if (particleID == -1 || dataset_name == ("Particle_" + std::to_string(particleID)))
            {
                particleFound = true;

                // Open the dataset
                hid_t dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
                if (dataset_id < 0)
                {
                    ERRMSG("Failed to open dataset: " + std::string(dataset_name));
                    continue;
                }

                // Get the dataspace and dimensions
                hid_t dataspace_id = H5Dget_space(dataset_id);
                hsize_t dims[2];
                H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

                // Read the data
                std::vector<double> positions(dims[0] * dims[1]);
                H5Dread(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions.data());

                // Print the particle's positions
                std::cout << dataset_name << ":\n";
                for (hsize_t j = 0; j < dims[0]; ++j)
                {
                    std::cout << "  " << j + 1 << ". [" << positions[j * 3] << ", "
                              << positions[j * 3 + 1] << ", " << positions[j * 3 + 2] << "]\n";
                }

                // Close the dataset and dataspace
                H5Dclose(dataset_id);
                H5Sclose(dataspace_id);

                // If a specific particle ID was requested, exit the loop after finding it
                if (particleID != -1)
                    break;
            }
        }

        if (!particleFound)
        {
            if (particleID == -1)
            {
                ERRMSG("No particles found in the file.");
            }
            else
            {
                ERRMSG("Particle with ID " + std::to_string(particleID) + " not found.");
            }
        }

        // Close the file
        H5Fclose(file_id);
    }
    catch (...)
    {
        ERRMSG("An error occurred while reading particle positions from HDF5.");
    }
}

void _saveParticleMovementsHdf5(ParticleMovementMap const &particlesMovement)
{
    try
    {
        if (particlesMovement.empty())
        {
            WARNINGMSG("Warning: Particle movements map is empty, no data to save");
            return;
        }

        // Create or open the HDF5 file
        std::string filepath("results/particles_movements.hdf5");
        hid_t file_id = H5Fcreate(filepath.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

        for (auto const &[id, movements] : particlesMovement)
        {
            if (movements.empty())
                continue;

            // Prepare the data for this particle
            std::vector<double> positions;
            for (auto const &point : movements)
            {
                positions.emplace_back(point.x());
                positions.emplace_back(point.y());
                positions.emplace_back(point.z());
            }

            // Create dataspace
            hsize_t dims[2] = {movements.size(), 3}; // Nx3 array
            hid_t dataspace_id = H5Screate_simple(2, dims, NULL);

            // Create dataset
            std::string dataset_name = "Particle_" + std::to_string(id);
            hid_t dataset_id = H5Dcreate2(file_id, dataset_name.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

            // Write data to the dataset
            H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions.data());

            // Close dataset and dataspace
            H5Dclose(dataset_id);
            H5Sclose(dataspace_id);
        }

        // Close the file
        H5Fclose(file_id);

        LOGMSG("Successfully written particle movements to the HDF5 file.");
    }
    catch (...)
    {
        ERRMSG("An error occurred while saving particle movements to HDF5.");
    }
}

void SputteringModel::_gfinalize()
{
    _updateSurfaceMesh();
    _saveParticleMovementsHdf5(m_particlesMovement);
}

SputteringModel::SputteringModel(std::string_view mesh_filename, std::string_view physicalGroupName)
    : m_mesh_filename(mesh_filename), m_surfaceMesh(mesh_filename) { _ginitialize(); }

SputteringModel::~SputteringModel() { _gfinalize(); }

void SputteringModel::startModeling(unsigned int numThreads)
{
    [[maybe_unused]] std::string sputteringMaterialName("Ti"), gasName("Ar"), scatteringModel("VSS");
    [[maybe_unused]] double targetMaterialDensity{4500.0}, targetMaterialMolarMass{0.048}, cellSize = 0.0, mm = 1.0;
    [[maybe_unused]] int meshTypeInt, particleWeight{12};

    int choiceInt{};
    std::cout << "Ручной ввод (1)/Автоматический (2): ";
    std::cin >> choiceInt;

    InputMode inputMode{choiceInt == 1 ? InputMode::Manual : InputMode::Auto};

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

    // 1.5. Constructing AABB tree to the surface mesh (with triangle cells). It construct inside SputteringModel instance.

    // ==========================================
    // 2. Spawn particles on the "Target" surface
    // ==========================================
    if (inputMode == InputMode::Manual)
    {
        std::cout << "Введите плотность материала мишени [кг/м3]: ";
        std::cin >> targetMaterialDensity;

        std::cout << "Введите молярную массу атома материала, из которого состоит мишень [кг/моль]: ";
        std::cin >> targetMaterialMolarMass;
    }

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

    double totalTime{};
    std::cout << "Введите время симуляции [с]: ";
    std::cin >> totalTime;

    double timeStep{};
    std::cout << "Введите шаг по времени [с] (желательно, маленький, например, 0.0001): ";
    std::cin >> timeStep;

    // 2.5. Calculating flux of the particles from this surface.
    double J_model{N_model / (S * totalTime)};
    std::cout << "Поток модельных частиц: " << J_model << " [N/(м2⋅c)]\n";

    double energy_eV{1};
    if (inputMode == InputMode::Manual)
    {
        std::cout << "Введите энергию, вылетающих частиц (не всех, а каждой) [эВ]: ";
        std::cin >> energy_eV;
    }

    // 2.6. Collecting cell centers and forcing direction [0,0,-1] from target plate to substrate
    //      and building surface source data for properly generating particles.
    auto surfaceSource{tpc.prepareDataForSpawnParticles(N_model, energy_eV)};
    double expansionAngle{30};
    if (inputMode == InputMode::Manual)
    {
        std::cout << "Введите угол рассеяния [в градусах]: ";
        std::cin >> expansionAngle;
    }
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

    // =============================================
    // === 3. Starting modeling in the main loop ===
    // =============================================
    double pressure{1}, temperature{273};
    if (inputMode == InputMode::Manual)
    {
        std::cout << "Введите давление [Па]: ";
        std::cin >> pressure;
        std::cout << "Введите температуру [К]: ";
        std::cin >> temperature;
    }

    double gasConcentration{(pressure / (constants::physical_constants::R * temperature)) * constants::physical_constants::N_av};
    if (gasConcentration < constants::gasConcentrationMinimalValue)
    {
        WARNINGMSG(util::stringify("Something wrong with the concentration of the gas. Its value is ", gasConcentration, ". Simulation might considerably slows down"));
    }

    for (double timeMoment{}; timeMoment <= totalTime && !m_stop_processing.test(); timeMoment += timeStep)
    {
        try
        {
            omp_set_num_threads(numThreads);

#pragma omp parallel for schedule(dynamic)
            for (size_t i = 0ul; i < particles.size(); ++i)
            {
                auto &particle = particles[i];

                // 1. Check if particle is settled.
                if (ParticleSettler::isSettled(particle.getId(), m_settledParticlesIds, m_settledParticlesMutex))
                    continue;

                // 2. Electromagnetic push (Here is no need to do EM-push).
                // @remark In the sputtering model there is no any field, so, there is no need to do EM-push.
                // m_physicsUpdater.doElectroMagneticPush(particle, gsmAssembler, *tetrahedronId);

                // 3. Record previous position.
                Point prev{particle.getCentre()};
                ParticleMovementTracker::recordMovement(m_particlesMovement, m_particlesMovementMutex, particle.getId(), prev);

                // 4. Update position.
                particle.updatePosition(timeStep);
                Ray ray(prev, particle.getCentre());
                if (ray.is_degenerate())
                    continue;

                // 5. Gas collision.
                ParticlePhysicsUpdater::collideWithGas(particle, scatteringModel, gasName, gasConcentration, timeStep);

                // 6. Skip surface collision checks if t == 0.
                if (timeMoment == 0.0)
                    continue;

                // 7. Surface collision.
                ParticleSurfaceCollisionHandler::handle(particle,
                                                        ray,
                                                        particles.size(),
                                                        m_surfaceMesh,
                                                        m_settledParticlesMutex,
                                                        m_particlesMovementMutex,
                                                        m_particlesMovement,
                                                        m_settledParticlesIds,
                                                        *this);
            }
        }
        catch (std::exception const &ex)
        {
            ERRMSG(util::stringify("Can't finish detecting particles collisions with surfaces (OMP): ", ex.what()));
        }
        catch (...)
        {
            ERRMSG("Some error occured while detecting particles collisions with surfaces (OMP)");
        }

        if (m_stop_processing.test())
        {
            SUCCESSMSG(util::stringify("All particles are settled. Stop requested by observers, terminating the simulation loop. ",
                                       "Last time moment is: ", timeMoment, "s."));
        }
        SUCCESSMSG(util::stringify("Time = ", timeMoment, "s. Totally settled: ", m_settledParticlesIds.size(), "/", particles.size(), " particles."));
    }
}
