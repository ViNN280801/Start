#include <iostream>
#include <stdexcept>

#include <gmsh.h>

#include "Utilities/GmshUtilities/GmshMesher.hpp"

GmshMesher::GmshMesher(std::string_view mesh_filename, MeshType mesh_type, double unit, double cell_size)
    : m_mesh_filename(mesh_filename), m_unit(unit), m_cell_size(cell_size)
{
    if (mesh_filename.empty())
        throw std::invalid_argument("Filename of the mesh can't be empty.");
    std::string tmp(mesh_filename);
    if (!tmp.ends_with(".msh"))
    {
        tmp += ".msh";
        WARNINGMSG(util::stringify("Missed '.msh' extension in passed filename: ",
                                   mesh_filename, ", adding it manually to get: ", tmp));
    }
    m_mesh_filename = tmp;

    if (mesh_type == MeshType::Uniform && (unit <= 0 || cell_size <= 0))
        throw std::invalid_argument("Unit and desired cell size must be positive values and not equal to 0.");
    GmshUtils::gmshInitializeCheck();
}

void GmshMesher::applyMesh(MeshType meshType,
                           std::vector<int> const &boundaryTags,
                           int inFieldTag,
                           double sizeMin,
                           double sizeMax,
                           double distMin,
                           double distMax) const
{
    GmshUtils::gmshInitializeCheck();
    switch (meshType)
    {
    case MeshType::Uniform:
        applyUniformMesh();
        break;
    case MeshType::Adaptive:
        applyAdaptiveMesh(boundaryTags, inFieldTag, sizeMin, sizeMax, distMin, distMax);
        break;
    default:
        throw std::invalid_argument("Wrong value of the 'meshTypeVariable'.");
        break;
    }
}

void GmshMesher::applyUniformMesh() const
{
    GmshUtils::gmshInitializeCheck();
    gmsh::option::setNumber("Mesh.MeshSizeMin", m_cell_size);
    gmsh::option::setNumber("Mesh.MeshSizeMax", m_cell_size);
}

void GmshMesher::applyAdaptiveMesh(std::vector<int> const &boundaryTags,
                                   int inFieldTag,
                                   double sizeMin,
                                   double sizeMax,
                                   double distMin,
                                   double distMax) const
{
    if (boundaryTags.empty())
        throw std::invalid_argument("Boundary tags cannot be empty.");
    if (sizeMin <= 0 || sizeMax <= 0 || distMin <= 0 || distMax <= 0)
        throw std::invalid_argument("Mesh sizes and distances must be positive values.");
    if (sizeMin > sizeMax || distMin > distMax)
        throw std::invalid_argument("Invalid range for size or distance parameters.");

    GmshUtils::gmshInitializeCheck();
    gmsh::model::mesh::field::add("Distance", inFieldTag);
    gmsh::model::mesh::field::setNumbers(inFieldTag, "SurfacesList",
                                         std::vector<double>(boundaryTags.cbegin(), boundaryTags.cend()));

    int thresholdFieldTag{inFieldTag + 1};
    gmsh::model::mesh::field::add("Threshold", thresholdFieldTag);
    gmsh::model::mesh::field::setNumber(thresholdFieldTag, "InField", static_cast<double>(inFieldTag));
    gmsh::model::mesh::field::setNumber(thresholdFieldTag, "SizeMin", sizeMin * m_unit);
    gmsh::model::mesh::field::setNumber(thresholdFieldTag, "SizeMax", sizeMax * m_unit);
    gmsh::model::mesh::field::setNumber(thresholdFieldTag, "DistMin", distMin * m_unit);
    gmsh::model::mesh::field::setNumber(thresholdFieldTag, "DistMax", distMax * m_unit);

    gmsh::model::mesh::field::setAsBackgroundMesh(thresholdFieldTag);
}

void GmshMesher::generateMeshfile(int dimension) const
{
    GmshUtils::gmshInitializeCheck();

    if (dimension < 2 || dimension > 3)
        throw std::invalid_argument(util::stringify("Supported dimensions for this method: 2D or 3D, but passed: ", dimension));

    try
    {
        gmsh::model::mesh::generate(dimension);
        gmsh::write(m_mesh_filename.data());

        LOGMSG(util::stringify("Model successfully created and saved to the file '", m_mesh_filename));
    }
    catch (const std::exception &ex)
    {
        ERRMSG(util::stringify("Error during mesh generation: ", ex.what()));
        throw ex;
    }
    catch (...)
    {
        ERRMSG("An unknown error occurred during mesh generation.");
        throw;
    }
}
