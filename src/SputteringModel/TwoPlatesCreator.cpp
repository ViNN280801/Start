#include <algorithm>
#include <set>

#include "SputteringModel/TwoPlatesCreator.hpp"

TwoPlatesCreator::TwoPlatesCreator(MeshType mesh_type, double cell_size, double unit)
    : m_gmsh_mesher(mesh_type, unit, cell_size), m_mesh_type(mesh_type), m_cell_size(cell_size), m_unit(unit)
{
    static_assert(std::is_floating_point_v<decltype(cell_size)>, "cell_size must be a floating-point type.");

    gmsh::model::add("TwoPlates");
    gmsh::option::setNumber("Mesh.SaveAll", 1.0); // Forcing Gmsh to save all elements (not only physical groups).
}

void TwoPlatesCreator::addPlates()
{
    m_volume_tags[0] = gmsh::model::occ::addBox(0, 0, 0, 100 * m_unit, 20 * m_unit, 1 * m_unit);
    m_volume_tags[1] = gmsh::model::occ::addBox(20, 5, 100 * m_unit, 60 * m_unit, 10 * m_unit, 1 * m_unit);
    gmsh::model::occ::synchronize();
}

void TwoPlatesCreator::assignMaterials(std::string_view material1_name, std::string_view material2_name)
{
    if (m_volume_tags.size() != 2)
        throw std::runtime_error("Expected exactly two plates to assign materials.");
    if (material1_name.length() < 1 || material1_name.length() > 2)
        throw std::runtime_error("Expected material name with length (1 <= len <= 2) for the 'material1_name' param.");
    if (material2_name.length() < 1 || material2_name.length() > 2)
        throw std::runtime_error("Expected material name with length (1 <= len <= 2) for the 'material2_name' param.");

    int material1_group{gmsh::model::addPhysicalGroup(3, {m_volume_tags.front()})};
    gmsh::model::setPhysicalName(3, material1_group, std::string(material1_name));

    int material2_group{gmsh::model::addPhysicalGroup(3, {m_volume_tags.back()})};
    gmsh::model::setPhysicalName(3, material2_group, std::string(material2_name));
}

void TwoPlatesCreator::generateMeshfile(int dimension)
{
    try
    {
        m_gmsh_mesher.applyMesh(m_mesh_type);
        m_gmsh_mesher.generateMeshfile(kdefault_mesh_filename, dimension);

        SUCCESSMSG(util::stringify("Model successfully created and saved to the file '", kdefault_mesh_filename));
    }
    catch (const std::exception &ex)
    {
        ERRMSG(util::stringify("Error: ", ex.what()));
    }
    catch (...)
    {
        ERRMSG("An unknown error occurred.");
    }
}

void TwoPlatesCreator::setTargetSurface() const
{
    int targetSurfaceTag{GmshUtils::findSurfaceTagByCoords<double>(kdefault_target_point_coords)};
    int targetGroupTag{gmsh::model::addPhysicalGroup(2, {targetSurfaceTag})};
    gmsh::model::setPhysicalName(2, targetGroupTag, kdefault_target_name);
}

void TwoPlatesCreator::setSubstrateSurface() const
{
    int targetSurfaceTag{GmshUtils::findSurfaceTagByCoords<double>(kdefault_substrate_point_coords)};
    int targetGroupTag{gmsh::model::addPhysicalGroup(2, {targetSurfaceTag})};
    gmsh::model::setPhysicalName(2, targetGroupTag, kdefault_substrate_name);
}

surface_source_t TwoPlatesCreator::prepareDataForSpawnParticles(size_t N_model, double energy_eV) const
{
    auto const &triangleCentersMap{GmshUtils::getCellCentersByPhysicalGroupName(kdefault_target_name)};

    GmshUtils::getCellsByPhysicalGroupName(kdefault_target_name);

    // Step 5: Populate the `surface_source_t` structure.
    surface_source_t source;
    source.type = "Ti";        // Particle type.
    source.count = N_model;    // Number of particles.
    source.energy = energy_eV; // Particle energy.

    for (const auto &[triangleTag, centroid] : triangleCentersMap)
    {
        // Convert the centroid to a string key.
        std::string centroidKey = std::to_string(centroid[0]) + ", " +
                                  std::to_string(centroid[1]) + ", " +
                                  std::to_string(centroid[2]);

        // Assign a normal vector [0, 0, -1] to each centroid.
        source.baseCoordinates[centroidKey] = {0.0, 0.0, -1.0};
    }

    if (source.baseCoordinates.empty())
    {
        ERRMSG("Error while filling sources - base coords are empty.");
        return {};
    }

    return source;
}

void TwoPlatesCreator::show(int argc, char *argv[]) { m_gmsh_session.runGmsh(argc, argv); }
