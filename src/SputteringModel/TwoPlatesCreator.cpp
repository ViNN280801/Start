#include <algorithm>
#include <set>

#include <gmsh.h>

#include "SputteringModel/TwoPlatesCreator.hpp"

std::vector<int> TwoPlatesCreator::_getAllBoundaryTags()
{
    std::vector<int> all_boundary_tags;
    std::vector<std::pair<int, int>> volumes;
    gmsh::model::getEntities(volumes, 3);

    for (auto const &volume : volumes)
    {
        std::vector<std::pair<int, int>> boundaries;
        gmsh::model::getBoundary({volume}, boundaries, true, false, false);
        for (auto const &boundary : boundaries)
        {
            if (boundary.first == 2)
                all_boundary_tags.push_back(boundary.second);
        }
    }
    return all_boundary_tags;
}

void TwoPlatesCreator::_applyMesh()
{
    if (m_mesh_type == MeshType::Uniform)
    {
        gmsh::option::setNumber("Mesh.MeshSizeMin", m_cell_size);
        gmsh::option::setNumber("Mesh.MeshSizeMax", m_cell_size);
    }
    else if (m_mesh_type == MeshType::Adaptive)
    {
        auto boundary_tags{_getAllBoundaryTags()};
        _applyAdaptiveMesh(boundary_tags);
    }
}

void TwoPlatesCreator::_applyAdaptiveMesh(const std::vector<int> &boundary_tags)
{
    std::vector<double> boundary_tags_double(boundary_tags.begin(), boundary_tags.end());

    gmsh::model::mesh::field::add("Distance", 1);
    gmsh::model::mesh::field::setNumbers(1, "SurfacesList", boundary_tags_double);

    gmsh::model::mesh::field::add("Threshold", 2);
    gmsh::model::mesh::field::setNumber(2, "InField", 1.0);
    gmsh::model::mesh::field::setNumber(2, "SizeMin", 1.0 * m_unit);
    gmsh::model::mesh::field::setNumber(2, "SizeMax", 100.0 * m_unit);
    gmsh::model::mesh::field::setNumber(2, "DistMin", 5.0 * m_unit);
    gmsh::model::mesh::field::setNumber(2, "DistMax", 50.0 * m_unit);

    gmsh::model::mesh::field::setAsBackgroundMesh(2);
}

TwoPlatesCreator::TwoPlatesCreator(MeshType mesh_type, double cell_size, double unit)
    : m_mesh_type(mesh_type), m_cell_size(cell_size), m_unit(unit)
{
    static_assert(std::is_floating_point_v<decltype(cell_size)>, "cell_size must be a floating-point type.");
    if (m_mesh_type == MeshType::Uniform && m_cell_size <= 0)
        throw std::invalid_argument("Cell size must be greater than zero for uniform mesh.");
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

void TwoPlatesCreator::generateMeshfile()
{
    try
    {
        _applyMesh();
        gmsh::model::mesh::generate(2);
        gmsh::write(kdefault_mesh_filename);

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
    std::vector<std::pair<int, int>> surfaces;
    gmsh::model::getEntities(surfaces, 2);

    int targetSurfaceTag{-1};
    for (auto const &surface : surfaces)
    {
        int currentSurfaceTag{surface.second};
        std::vector<size_t> nodeTags;
        std::vector<double> coords, parametricCoords;
        gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords, 2, currentSurfaceTag, true, true);

        std::set<std::array<double, 3ul>> surfaceNodes;
        for (size_t i{}; i < coords.size(); i += 3)
            surfaceNodes.insert({coords.at(i), coords.at(i + 1), coords.at(i + 2)});

        bool isTargetSurface{true};
        for (auto const &targetCoord : kdefault_target_point_coords)
        {
            if (surfaceNodes.find(targetCoord) == surfaceNodes.cend())
            {
                isTargetSurface = false;
                break;
            }
        }

        if (isTargetSurface)
            targetSurfaceTag = currentSurfaceTag;
    }

    if (targetSurfaceTag == -1)
        ERRMSG("Error finding taget surface.");

    int target_group{gmsh::model::addPhysicalGroup(2, {targetSurfaceTag})};
    gmsh::model::setPhysicalName(2, target_group, "Target");
}

surface_source_t TwoPlatesCreator::prepareDataForSpawnParticles(size_t N_model, double energy_eV) const
{
    // Step 1: Find the physical group with the name "Target"
    std::vector<std::pair<int, int>> physicalGroups;
    gmsh::model::getPhysicalGroups(physicalGroups, 2);

    int targetGroupTag{-1};
    for (const auto &[groupDim, groupTag] : physicalGroups)
    {
        std::string name;
        gmsh::model::getPhysicalName(2, groupTag, name);
        if (name == "Target")
        {
            targetGroupTag = groupTag;
            break;
        }
    }

    if (targetGroupTag == -1)
    {
        ERRMSG("Error: Physical group 'Target' not found.");
        return {};
    }

    // Step 2: Get the nodes associated with the "Target" physical group
    std::vector<size_t> targetNodeTags;
    std::vector<double> targetCoords;
    gmsh::model::mesh::getNodesForPhysicalGroup(2, targetGroupTag, targetNodeTags, targetCoords);

    // Store the target node tags in a set for quick lookup
    std::set<size_t> targetNodeTagSet(targetNodeTags.cbegin(), targetNodeTags.cend());

    // Step 3: Identify triangles associated with the target surface
    std::unordered_map<size_t, std::vector<size_t>> triangleNodeTagsMap;
    std::vector<size_t> triangleTags, triangleNodeTags;
    gmsh::model::mesh::getElementsByType(2, triangleTags, triangleNodeTags);

    short nodesPerTriangle{3};
    for (size_t i{}; i < triangleTags.size(); ++i)
    {
        size_t triangleTag{triangleTags.at(i)};
        std::vector<size_t> triangleNodes;
        bool allNodesInTarget{true};

        for (short j{}; j < nodesPerTriangle; ++j)
        {
            size_t nodeTag{triangleNodeTags.at(i * nodesPerTriangle + j)};
            if (targetNodeTagSet.find(nodeTag) == targetNodeTagSet.cend())
            {
                allNodesInTarget = false;
                break;
            }
            triangleNodes.emplace_back(nodeTag);
        }

        if (allNodesInTarget)
            triangleNodeTagsMap[triangleTag] = triangleNodes;
    }

    // Step 4: Calculate centroids of triangles
    std::unordered_map<size_t, std::array<double, 3ul>> triangleCentersMap;
    for (const auto &[cellTag, cellNodeTags] : triangleNodeTagsMap)
    {
        if (cellNodeTags.size() != 3ul)
        {
            ERRMSG(util::stringify("Error: Triangle with tag ", cellTag, " does not have exactly 3 nodes."));
            continue;
        }

        std::array<double, 3ul> centroid{0.0, 0.0, 0.0};
        for (size_t cellNodeTag : cellNodeTags)
        {
            std::vector<double> coord, paramCoord;
            int dim{}, tag{};
            gmsh::model::mesh::getNode(cellNodeTag, coord, paramCoord, dim, tag);

            if (coord.size() != 3)
            {
                ERRMSG(util::stringify("Error: Node ", cellNodeTag, " does not have 3D coordinates."));
                continue;
            }

            centroid[0] += coord[0];
            centroid[1] += coord[1];
            centroid[2] += coord[2];
        }

        // Average the coordinates to get the centroid
        centroid[0] /= 3.0;
        centroid[1] /= 3.0;
        centroid[2] /= 3.0;

        triangleCentersMap[cellTag] = centroid;
    }

    // Step 5: Populate the `surface_source_t` structure
    surface_source_t source;
    source.type = "Ti";        // Particle type
    source.count = N_model;    // Number of particles
    source.energy = energy_eV; // Particle energy

    for (const auto &[triangleTag, centroid] : triangleCentersMap)
    {
        // Convert the centroid to a string key
        std::string centroidKey = std::to_string(centroid[0]) + ", " +
                                  std::to_string(centroid[1]) + ", " +
                                  std::to_string(centroid[2]);

        // Assign a normal vector [0, 0, -1] to each centroid
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

std::string TwoPlatesCreator::getMaterial(int plate_number) const
{
    std::string material_name;
    if (plate_number == 1)
        gmsh::model::getPhysicalName(3, m_volume_tags.front(), material_name);
    else if (plate_number == 2)
        gmsh::model::getPhysicalName(3, m_volume_tags.back(), material_name);
    else
        throw std::runtime_error("Expected 'plate_number' == 1 or 'plate_number == 2'");
    return material_name;
}

std::string TwoPlatesCreator::getMeshFilename() const { return kdefault_mesh_filename; }
