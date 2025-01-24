#include <gmsh.h>
#include <limits>

#include "Geometry/Mesh.hpp"
#include "Utilities/Utilities.hpp"

std::optional<AABB_Tree_Triangle> constructAABBTreeFromMeshParams(TriangleCellMap const &meshParams)
{
    if (meshParams.empty())
    {
        ERRMSG("Can't construct AABB for triangle mesh -> mesh is empty");
        return std::nullopt;
    }

    TriangleVector triangles;
    for (auto const &[id, triangleCell] : meshParams)
    {
        if (triangleCell.triangle.is_degenerate())
        {
            WARNINGMSG(util::stringify("Triangle with ID ", id, " is degenerate, skipping it..."));
            continue;
        }
        triangles.emplace_back(triangleCell.triangle);
    }

    if (triangles.empty())
    {
        ERRMSG("Can't create AABB for triangle mesh -> triangles from the mesh are invalid (all degenerate)");
        return std::nullopt;
    }

    return AABB_Tree_Triangle(std::cbegin(triangles), std::cend(triangles));
}

void Mesh::setMeshSize(double meshSizeFactor) { gmsh::option::setNumber("Mesh.MeshSizeFactor", meshSizeFactor); }

TriangleCellMap Mesh::getMeshParams(std::string_view msh_filename)
{
    TriangleCellMap result;
    try
    {
        gmsh::open(msh_filename.data());

        std::vector<size_t> nodeTags;
        std::vector<double> coords, parametricCoords;
        gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords, -1, -1);

        std::vector<double> xyz;
        for (size_t i{}; i < coords.size(); i += 3)
        {
            xyz.emplace_back(coords[i]);
            xyz.emplace_back(coords[i + 1]);
            xyz.emplace_back(coords[i + 2]);
        }

        std::vector<size_t> elTags, nodeTagsByEl;
        gmsh::model::mesh::getElementsByType(2, elTags, nodeTagsByEl, -1);

        std::vector<std::vector<size_t>> nodeTagsPerEl;
        for (size_t i{}; i < nodeTagsByEl.size(); i += 3)
            nodeTagsPerEl.push_back({nodeTagsByEl[i], nodeTagsByEl[i + 1], nodeTagsByEl[i + 2]});

        for (size_t i{}; i < elTags.size(); ++i)
        {
            size_t triangleId{elTags[i]};
            std::vector<size_t> nodes = nodeTagsPerEl[i];

            std::vector<double> xyz1{{xyz[(nodes[0] - 1) * 3], xyz[(nodes[0] - 1) * 3 + 1], xyz[(nodes[0] - 1) * 3 + 2]}},
                xyz2{{xyz[(nodes[1] - 1) * 3], xyz[(nodes[1] - 1) * 3 + 1], xyz[(nodes[1] - 1) * 3 + 2]}},
                xyz3{{xyz[(nodes[2] - 1) * 3], xyz[(nodes[2] - 1) * 3 + 1], xyz[(nodes[2] - 1) * 3 + 2]}};

            Point p1(xyz1[0], xyz1[1], xyz1[2]),
                p2(xyz2[0], xyz2[1], xyz2[2]),
                p3(xyz3[0], xyz3[1], xyz3[2]);
            Triangle triangle(p1, p2, p3);
            TriangleCell cell{triangle, TriangleCell::compute_area(triangle), 0}; // At the initial time moment 'counter' is 0.
            result[triangleId] = TriangleCell(cell);
        }
    }
    catch (std::exception const &e)
    {
        ERRMSG(e.what());
    }
    catch (...)
    {
        ERRMSG("Something went wrong");
    }

    if (result.empty())
        WARNINGMSG("By some reason 'result' data ('TriangleCellMap') is empty, check the input...");

    return result;
}

TetrahedronCellMap Mesh::getTetrahedronMeshParams(std::string_view msh_filename)
{
    TetrahedronCellMap result;
    try
    {
        gmsh::open(msh_filename.data());

        std::vector<size_t> nodeTags;
        std::vector<double> coords;
        std::vector<double> parametricCoords;
        gmsh::model::mesh::getNodes(nodeTags, coords, parametricCoords);

        std::vector<double> xyz;
        for (size_t i{}; i < coords.size(); i += 3)
        {
            xyz.emplace_back(coords[i]);
            xyz.emplace_back(coords[i + 1]);
            xyz.emplace_back(coords[i + 2]);
        }

        std::vector<size_t> elTags, nodeTagsByEl;
        gmsh::model::mesh::getElementsByType(4, elTags, nodeTagsByEl, -1);

        std::vector<std::vector<size_t>> nodeTagsPerEl;
        for (size_t i{}; i < nodeTagsByEl.size(); i += 4)
            nodeTagsPerEl.push_back({nodeTagsByEl[i], nodeTagsByEl[i + 1], nodeTagsByEl[i + 2], nodeTagsByEl[i + 3]});

        for (size_t i{}; i < elTags.size(); ++i)
        {
            size_t tetrahedronID{elTags[i]};
            std::vector<size_t> nodes = nodeTagsPerEl[i];

            std::array<std::vector<double>, 4> vertices;
            for (int j{}; j < 4; ++j)
                vertices[j] = {xyz[(nodes[j] - 1) * 3], xyz[(nodes[j] - 1) * 3 + 1], xyz[(nodes[j] - 1) * 3 + 2]};

            Tetrahedron tetrahedron(Point(vertices[0][0], vertices[0][1], vertices[0][2]),
                                    Point(vertices[1][0], vertices[1][1], vertices[1][2]),
                                    Point(vertices[2][0], vertices[2][1], vertices[2][2]),
                                    Point(vertices[3][0], vertices[3][1], vertices[3][2]));
            result[tetrahedronID] = TetrahedronCell(tetrahedron);
        }
    }
    catch (std::exception const &e)
    {
        ERRMSG(e.what());
    }
    catch (...)
    {
        ERRMSG("Something went wrong");
    }

    if (result.empty())
        WARNINGMSG("By some reason 'result' data ('TetrahedronCellMap') is empty, check the input...");

    return result;
}

bool Mesh::isPointInsideTetrahedron(Point const &point, Tetrahedron const &tetrahedron)
{
    CGAL::Oriented_side oriented_side{tetrahedron.oriented_side(point)};
    if (oriented_side == CGAL::ON_POSITIVE_SIDE)
        return true;
    else if (oriented_side == CGAL::ON_NEGATIVE_SIDE)
        return false;
    else
        // TODO: Correctly handle case when particle is on boundary of tetrahedron.
        return true;
}

double Mesh::getVolumeFromTetrahedronMesh(std::string_view msh_filename)
{
    double totalVolume{};
    auto tetrahedronMesh{getTetrahedronMeshParams(msh_filename)};
    for (auto const &[id, tetrahedronCell] : tetrahedronMesh)
        totalVolume += tetrahedronCell.tetrahedron.volume();
    return totalVolume;
}

std::map<size_t, std::vector<size_t>> Mesh::getTetrahedronNodesMap(std::string_view msh_filename)
{
    std::map<size_t, std::vector<size_t>> tetrahedronNodesMap;
    try
    {
        gmsh::open(msh_filename.data());

        std::vector<size_t> elTags, nodeTagsByEl;
        gmsh::model::mesh::getElementsByType(4, elTags, nodeTagsByEl, -1);

        for (size_t i{}; i < elTags.size(); ++i)
        {
            size_t tetrahedronID{elTags[i]};
            std::vector<size_t> nodes = {
                nodeTagsByEl[i * 4 + 0],
                nodeTagsByEl[i * 4 + 1],
                nodeTagsByEl[i * 4 + 2],
                nodeTagsByEl[i * 4 + 3]};
            tetrahedronNodesMap[tetrahedronID] = nodes;
        }
    }
    catch (std::exception const &e)
    {
        ERRMSG(e.what());
    }
    catch (...)
    {
        ERRMSG("An unknown error occurred.");
    }
    return tetrahedronNodesMap;
}

std::map<size_t, std::vector<size_t>> Mesh::getNodeTetrahedronsMap(std::string_view msh_filename)
{
    std::map<size_t, std::vector<size_t>> nodeTetrahedronMap;
    try
    {
        gmsh::open(msh_filename.data());

        std::vector<size_t> elTags, nodeTagsByEl;
        gmsh::model::mesh::getElementsByType(4, elTags, nodeTagsByEl, -1);

        constexpr size_t const nodesPerTetrahedron{4ul};
        for (size_t i{}; i < elTags.size(); ++i)
        {
            size_t tetrahedronID{elTags[i]};
            for (size_t j{}; j < nodesPerTetrahedron; ++j)
            {
                size_t nodeID{nodeTagsByEl[i * nodesPerTetrahedron + j]};
                nodeTetrahedronMap[nodeID].emplace_back(tetrahedronID);
            }
        }
    }
    catch (std::exception const &e)
    {
        ERRMSG(e.what());
    }
    catch (...)
    {
        ERRMSG("An unknown error occurred.");
    }
    return nodeTetrahedronMap;
}

std::map<size_t, std::array<double, 3>> Mesh::getTetrahedronNodeCoordinates(std::string_view msh_filename)
{
    std::map<size_t, std::array<double, 3>> nodeCoordinatesMap;
    try
    {
        gmsh::open(msh_filename.data());

        std::vector<std::size_t> nodeTags;
        std::vector<double> coord, parametricCoord;
        gmsh::model::mesh::getNodes(nodeTags, coord, parametricCoord, -1, -1, false, false);

        for (size_t i{}; i < nodeTags.size(); ++i)
        {
            size_t nodeID = nodeTags[i];
            std::array<double, 3> coords = {coord[i * 3 + 0], coord[i * 3 + 1], coord[i * 3 + 2]};
            nodeCoordinatesMap[nodeID] = coords;
        }
    }
    catch (const std::exception &e)
    {
        ERRMSG(e.what());
    }
    catch (...)
    {
        ERRMSG("An unknown error occurred.");
    }
    return nodeCoordinatesMap;
}

std::vector<size_t> Mesh::getTetrahedronMeshBoundaryNodes(std::string_view msh_filename)
{
    std::vector<size_t> boundaryNodeTags;
    try
    {
        gmsh::open(msh_filename.data());

        std::vector<size_t> nodeTags;
        std::vector<double> coords, parametricCoords;
        gmsh::model::mesh::getNodesByElementType(2, nodeTags, coords, parametricCoords);

        std::set<size_t> uniqueNodes(nodeTags.cbegin(), nodeTags.cend());
        boundaryNodeTags.assign(uniqueNodes.cbegin(), uniqueNodes.cend());
    }
    catch (const std::exception &e)
    {
        ERRMSG(e.what());
    }
    catch (...)
    {
        ERRMSG("An unknown error occurred.");
    }
    return boundaryNodeTags;
}

std::map<size_t, std::array<double, 3>> Mesh::getTetrahedronCenters(std::string_view msh_filename)
{
    std::map<size_t, std::array<double, 3>> tetrahedronCenters;
    try
    {
        auto nodeMap{getTetrahedronNodesMap(msh_filename)};           // map<TetraID, vector<NodeID>>.
        auto nodeCoords{getTetrahedronNodeCoordinates(msh_filename)}; // map<NodeID, array<double, 3>>.

        for (auto const &[tetraId, nodeIds] : nodeMap)
        {
            std::array<double, 3> tetraCentre = {0.0, 0.0, 0.0};

            for (size_t nodeId : nodeIds)
            {
                auto coords{nodeCoords.at(nodeId)};
                tetraCentre[0] += coords[0];
                tetraCentre[1] += coords[1];
                tetraCentre[2] += coords[2];
            }

            tetraCentre[0] /= 4.;
            tetraCentre[1] /= 4.;
            tetraCentre[2] /= 4.;

            tetrahedronCenters[tetraId] = tetraCentre;
        }
        return tetrahedronCenters;
    }
    catch (const std::exception &e)
    {
        ERRMSG(e.what());
    }
    catch (...)
    {
        ERRMSG("An unknown error occurred.");
    }
    WARNINGMSG("Returning an empty map for tetrahedron centers");
    return tetrahedronCenters;
}
