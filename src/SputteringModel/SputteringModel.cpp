#include <hdf5.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "DataHandling/SettledParticleHDF5Writer.hpp"
#include "DataHandling/TriangleMeshHdf5Manager.hpp"
#include "Generators/ParticleGenerator.hpp"
#include "Geometry/Utils/Overlaps/TriangleOverlapCalculator.hpp"
#include "Particle/CUDA/ParticleDeviceMemoryConverter.cuh"
#include "ParticleInCellEngine/ParticleDynamicsProcessor/ParticleDistributor.hpp"
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

void _readParticlePositionsHdf5(std::string const &filepath = "results/particles_movements.hdf5", int particleID = -1)
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

void SputteringModel::_writeHistogramToFile(std::string_view filepath)
{
    // =============================================
    // === 4. Writing the histogram to the file  ===
    // =============================================
    // Coordinates of the central axis (X=50, Y=0→20, Z=1)
    constexpr double axis_center{50.0};
    constexpr double epsilon{2.0}; // Width of the band ±2 cm from the axis

    // Bounds of the area
    constexpr double x_min{axis_center - epsilon};
    constexpr double x_max{axis_center + epsilon};

    auto const &triangleMap{m_surfaceMesh.getTriangleCellMap()};
    std::unordered_map<size_t, double> filtered_counts;

    for (auto const &[id, cell] : triangleMap)
    {
        // Get the coordinates of the centroid
        Point centroid{TriangleCell::compute_centroid(cell.triangle)};

        // Check if the centroid belongs to the band
        if (centroid.x() < x_min || centroid.x() > x_max)
            continue;

        // Calculate the ratio of the area in the band
        double overlap_ratio{TriangleOverlapCalculator::calculateOverlapRatio(cell.triangle, x_min, x_max)};

        // Consider only if >75% of the area is in the band
        if (overlap_ratio >= 0.75)
            filtered_counts[id] = cell.count * overlap_ratio;
    }

    // Grouping by Y-coordinate with a step of 1 cm
    constexpr double bin_size{1.0};
    std::map<double, double> y_bins;
    for (auto const &[id, count] : filtered_counts)
    {
        Point centroid{TriangleCell::compute_centroid(triangleMap.at(id).triangle)};

        // Calculate the bin by taking the floor of (Y / bin_size) and then multiplying back.
        double bin{std::floor(centroid.y() / bin_size) * bin_size};
        y_bins[bin] += count;
    }

    // Writing the histogram to the file in format: y_bin, count
    std::ofstream hist("results/histogram.dat");
    for (auto const &[y, total] : y_bins)
        hist << y << " " << total << "\n";

    std::cout << "Triangles in the band [" << x_min << ", " << x_max << "]:\n";
    std::cout << std::setw(10) << "ID"
              << std::setw(12) << "Centroid X"
              << std::setw(12) << "Overlap"
              << std::setw(12) << "Count\n";

    for (auto const &[id, cell] : m_surfaceMesh.getTriangleCellMap())
    {
        Point centroid{TriangleCell::compute_centroid(cell.triangle)};
        double overlap{TriangleOverlapCalculator::calculateOverlapRatio(cell.triangle, x_min, x_max)};

        // Condition for output: 75% of the area of the triangle is in the band
        if (overlap > 0.75)
        {
            std::cout << std::setw(10) << id
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << centroid.x()
                      << std::setw(12) << overlap
                      << std::setw(12) << cell.count << "\n";
        }
    }
}

void SputteringModel::_writeKDEToFile(std::string_view filepath)
{
    std::ofstream hist(filepath.data());
    if (!hist.is_open())
    {
        ERRMSG("Failed to open histogram.dat for writing.");
        return;
    }
    hist << "# X-coordinate(cm) Y-coordinate(cm) Count\n";

    for (auto const &[id, cell] : m_surfaceMesh.getTriangleCellMap())
    {
        if (cell.count > 0)
        {
            Point centroid{TriangleCell::compute_centroid(cell.triangle)};
            hist << centroid.x() << " " << centroid.y() << " " << cell.count << "\n";
        }
    }
    hist.close();
}

void SputteringModel::_gfinalize()
{
    _updateSurfaceMesh();
    _saveParticleMovementsHdf5(m_particlesMovement);
    if (m_particleWeight != 1)
        _distributeSettledParticles();
    _writeHistogramToFile();
    _writeKDEToFile();
}

SputteringModel::SputteringModel(std::string_view mesh_filename, std::string_view physicalGroupName)
    : m_mesh_filename(mesh_filename) { _ginitialize(); }

SputteringModel::~SputteringModel() { _gfinalize(); }

void SputteringModel::startModeling(unsigned int numThreads)
{
    [[maybe_unused]] std::string sputteringMaterialName("Ti"), gasName("Ar"), scatteringModel("VSS");
    [[maybe_unused]] double targetMaterialDensity{4500.0},
        targetMaterialMolarMass{0.048},
        cellSize{},
        mm{1};
    [[maybe_unused]] int meshTypeInt{};

    int choiceInt{};
    std::cout << "Manual input (1)/Automatic (2): ";
    std::cin >> choiceInt;

    InputMode inputMode{choiceInt == 1 ? InputMode::Manual : InputMode::Auto};

    // ===================================
    // 1. Generating 3D-model of 2 plates.
    // ===================================
    std::cout << "Enter the type of mesh (1 - uniform, 2 - adaptive): ";
    std::cin >> meshTypeInt;
    MeshType meshType{meshTypeInt == 1 ? MeshType::Uniform : MeshType::Adaptive};

    if (meshType == MeshType::Uniform)
    {
        std::cout << "Enter the desired size of the cell side: ";
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
    tpc.generateMeshfile(3);

    // 1.5. Constructing AABB tree to the surface mesh (with triangle cells). It construct inside SputteringModel instance.
    m_surfaceMesh = SurfaceMesh(tpc.kdefault_mesh_filename, tpc.kdefault_substrate_name);

    // ==========================================
    // 2. Spawn particles on the "Target" surface
    // ==========================================
    if (inputMode == InputMode::Manual)
    {
        std::cout << "Enter the density of the target material [kg/m3]: ";
        std::cin >> targetMaterialDensity;

        std::cout << "Enter the molar mass of the atom of the material that makes up the target [kg/mol]: ";
        std::cin >> targetMaterialMolarMass;
    }

    // 2.2. Calculating surface area.
    double S{tpc.calculateTargetSurfaceArea()};
    std::cout << "The surface area of the target: " << S << " [m2]\n";
    std::cout << "The radius of the Ti atom = " << constants::physical_constants::Ti_mass << " [m]\n";

    // 2.3. Calculating real count of particles on this surface.
    double N{tpc.calculateCountOfParticlesOnTargetSurface(targetMaterialDensity, targetMaterialMolarMass)};
    std::cout << "The number of atoms on the target surface: " << N << '\n';

    std::cout << "Do you want to change particle count? (1 - yes, 0 - no): ";
    std::cin >> choiceInt;
    if (choiceInt == 1)
    {
        std::cout << "Enter the new particle count: ";
        std::cin >> N;
        std::cout << "Now particle count is " << N << '\n';
    }

    std::cout << "Enter the weight of the one modeling particle (1 modeling particle = 10^m_particleWeight real particles): ";
    std::cin >> m_particleWeight;
    if (m_particleWeight < 0)
    {
        ERRMSG("The weight of the particle cannot be less than 0");
        return;
    }

    // 2.4. Calculating model count of particles on this surface.
    double N_model{std::ceil(N / std::pow(10, m_particleWeight))};
    std::cout << "Thus, " << N << " real particles = " << N_model << " simulated particles\n";

    double totalTime{};
    std::cout << "Enter the simulation time [s]: ";
    std::cin >> totalTime;

    double timeStep{};
    std::cout << "Enter the time step [s] (preferably, small, e.g. 0.0001): ";
    std::cin >> timeStep;

    // 2.5. Calculating flux of the particles from this surface.
    double J_model{N_model / (S * totalTime)};
    std::cout << "The flux of the simulated particles: " << J_model << " [N/(m2⋅c)]\n";

    double energy_eV{};
    std::cout << "Enter the energy of the particles that leave the surface [eV]: ";
    std::cin >> energy_eV;

    // 2.6. Collecting cell centers and forcing direction [0,0,-1] from target plate to substrate
    //      and building surface source data for properly generating particles.
    auto surfaceSource{tpc.prepareDataForSpawnParticles(N_model, energy_eV)};
    double expansionAngle{30};
    if (inputMode == InputMode::Manual)
    {
        std::cout << "Enter the scattering angle [in degrees]: ";
        std::cin >> expansionAngle;
    }
    expansionAngle *= START_PI_NUMBER / 180.0;

    // 2.7. Generating particles on the target surface.
    ParticleVector particles{ParticleGenerator::fromSurfaceSource({surfaceSource}, expansionAngle)};
    if (particles.empty())
    {
        ERRMSG("Error generating particles on the surface.");
    }
    else
    {
        SUCCESSMSG(util::stringify(" ", particles.size(), " particles were generated on the target surface."));
    }

    // =============================================
    // === 3. Starting modeling in the main loop ===
    // =============================================
    double pressure{1}, temperature{273};
    if (inputMode == InputMode::Manual)
    {
        std::cout << "Enter the pressure [Pa]: ";
        std::cin >> pressure;
        std::cout << "Enter the temperature [K]: ";
        std::cin >> temperature;
    }

    double gasConcentration{util::calculateConcentration(pressure, temperature)};

#if __cplusplus >= 202002L
    for (double timeMoment{}; timeMoment <= totalTime && !m_stop_processing.test(); timeMoment += timeStep)
#else
    for (double timeMoment{}; timeMoment <= totalTime && !m_stop_processing.test_and_set(); timeMoment += timeStep)
#endif
    {
#if __cplusplus <= 201703L
        // When we use test_and_set(), we need to reset the flag after each iteration.
        m_stop_processing.clear();
#endif

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
                Segment segment(prev, particle.getCentre());
                if (segment.is_degenerate())
                    continue;

                // 5. Gas collision.
                ParticlePhysicsUpdater::collideWithGas(particle, scatteringModel, gasName, gasConcentration, timeStep);

                // 6. Skip surface collision checks if t == 0.
                if (timeMoment == 0.0)
                    continue;

                // 7. Surface collision.
                ParticleSurfaceCollisionHandler::handle(particle,
                                                        segment,
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

#if __cplusplus >= 202002L
        if (m_stop_processing.test())
#else
        if (m_stop_processing.test_and_set())
#endif
        {
            SUCCESSMSG(util::stringify("All particles are settled. Stop requested by observers, terminating the simulation loop. ",
                                       "Last time moment is: ", timeMoment, "s."));
        }
        SUCCESSMSG(util::stringify("Time = ", timeMoment, "s. Totally settled: ",
                                   m_settledParticlesIds.size(), "/",
                                   particles.size(), " particles."));
    }
}

void SputteringModel::_distributeSettledParticles(std::string_view filepath)
{
    try
    {
        // 1. Ask user about distribution type
        int distributionTypeInt{};
        std::cout << "Which of the following options do you want to use for particle distribution? (1 - uniform, 2 - gaussian): ";
        std::cin >> distributionTypeInt;

        auto distributionType = static_cast<ParticleDistributor::DistributionType>(distributionTypeInt);

        // 2. Distribute particles across the mesh
        auto distributionResult = ParticleDistributor::distributeParticles(
            m_surfaceMesh,
            m_particleWeight,
            distributionType);

        // 4. Save the distributed particles to HDF5 file
        std::string outputFilePath = filepath.empty() ? "results/settled_particles.hdf5" : std::string(filepath);
        bool saveResult = SettledParticleHDF5Writer::saveParticlesToHDF5(
            distributionResult.positions,
            outputFilePath);

        if (!saveResult)
        {
            WARNINGMSG("Failed to save settled particles to HDF5 file");
        }
    }
    catch (const std::exception &e)
    {
        ERRMSG(util::stringify("Error in particle distribution: ", e.what()));
    }
    catch (...)
    {
        ERRMSG("Unknown error during particle distribution");
    }
}

int main()
{
    try
    {
        SputteringModel sm("meshes/TwoPlates.msh", "Substrate");
        sm.startModeling(10);
    }
    catch (std::exception const &ex)
    {
        ERRMSG(util::stringify("Error: ", ex.what()));
    }
    catch (...)
    {
        ERRMSG("Unknown error during sputtering modeling");
    }
    return EXIT_SUCCESS;
}
