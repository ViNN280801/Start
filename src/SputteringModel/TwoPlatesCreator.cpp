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
}

void TwoPlatesCreator::addPlates()
{
    m_volume_tags[0] = gmsh::model::occ::addBox(0, 0, 0, 100 * m_unit, 20 * m_unit, 1 * m_unit);
    m_volume_tags[1] = gmsh::model::occ::addBox(20, 5, 100 * m_unit, 60 * m_unit, 10 * m_unit, 1 * m_unit);
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
        // Synchronize the OpenCASCADE model.
        gmsh::model::occ::synchronize();
        _applyMesh();
        gmsh::model::mesh::generate(2);
        gmsh::write(kdefault_mesh_filename);

        std::cout << "Model successfully created and saved to the file '" << kdefault_mesh_filename << "'\n";
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Error: " << ex.what() << std::endl;
    }
    catch (...)
    {
        std::cerr << "An unknown error occurred." << std::endl;
    }
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
