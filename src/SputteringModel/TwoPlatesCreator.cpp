#include "TwoPlatesCreator.hpp"

void TwoPlatesCreator::_initializeGmsh()
{
    if (!gmsh::isInitialized())
        gmsh::initialize();
    gmsh::model::add("TwoPlates");
}

void TwoPlatesCreator::_addPlates()
{
    gmsh::model::occ::addBox(0, 0, 0, 100 * m_unit, 20 * m_unit, 1 * m_unit);
    gmsh::model::occ::addBox(0, 0, 100 * m_unit, 50 * m_unit, 20 * m_unit, 1 * m_unit);
    gmsh::model::occ::synchronize();
}

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
        auto boundary_tags = _getAllBoundaryTags();
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
    gmsh::model::mesh::field::setNumber(2, "DistMin", 0.5 * m_unit);
    gmsh::model::mesh::field::setNumber(2, "DistMax", 10.0 * m_unit);

    gmsh::model::mesh::field::setAsBackgroundMesh(2);
}

void TwoPlatesCreator::_generateMesh()
{
    try
    {
        _applyMesh();
        gmsh::model::mesh::generate(2);
        gmsh::write("TwoPlates.msh");

        std::cout << "Model successfully created and saved.\n";
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

TwoPlatesCreator::TwoPlatesCreator(MeshType mesh_type, double cell_size, double unit)
    : m_mesh_type(mesh_type), m_cell_size(cell_size), m_unit(unit)
{
    static_assert(std::is_floating_point_v<decltype(cell_size)>, "cell_size must be a floating-point type.");
    if (m_mesh_type == MeshType::Uniform && m_cell_size <= 0)
    {
        throw std::invalid_argument("Cell size must be greater than zero for uniform mesh.");
    }
    _initializeGmsh();
    _addPlates();
    _generateMesh();
}

TwoPlatesCreator::~TwoPlatesCreator()
{
    if (gmsh::isInitialized())
        gmsh::finalize();
}

void TwoPlatesCreator::show() { gmsh::fltk::run(); }
