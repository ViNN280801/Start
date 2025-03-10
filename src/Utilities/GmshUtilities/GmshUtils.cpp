#include <set>

#include "Utilities/GmshUtilities/GmshUtils.hpp"

void GmshUtils::gmshInitializeCheck()
{
    if (!gmsh::isInitialized())
        throw std::runtime_error("This method works only when Gmsh was initialized.");
}

void GmshUtils::checkAndOpenMesh(std::string_view meshFilename)
{
    GmshUtils::gmshInitializeCheck();
    util::check_gmsh_mesh_file(meshFilename);

    gmsh::option::setNumber("General.Terminal", 0); // Turn off logging to the terminal.
    gmsh::open(meshFilename.data());
    gmsh::option::setNumber("General.Terminal", 1); // Turn on logging to the terminal.
}

std::vector<int> GmshUtils::getAllBoundaryTags()
{
    std::vector<std::pair<int, int>> volumes;
    gmsh::model::getEntities(volumes, 3);

    if (volumes.empty())
        throw std::runtime_error("There is no volume entities.");

    std::vector<int> allBoundaryTags;
    for (auto const &volume : volumes)
    {
        std::vector<std::pair<int, int>> boundaries;
        gmsh::model::getBoundary({volume}, boundaries, true, false, false);
        for (auto const &boundary : boundaries)
            allBoundaryTags.push_back(boundary.second);
    }

    if (allBoundaryTags.empty())
        throw std::runtime_error("There is no boundary tags.");

    return allBoundaryTags;
}

std::vector<int> GmshUtils::getAllBoundaryTags(std::string_view meshFilename)
{
    GmshUtils::checkAndOpenMesh(meshFilename);
    return GmshUtils::getAllBoundaryTags();
}

int GmshUtils::getPhysicalGroupTagByName(std::string_view physicalGroupName, std::string_view meshFilename)
{
    GmshUtils::checkAndOpenMesh(meshFilename);

    std::vector<std::pair<int, int>> physicalGroups;
    gmsh::model::getPhysicalGroups(physicalGroups, 2);

    if (physicalGroups.empty())
        throw std::runtime_error("There is no physical groups");

    int targetGroupTag{-1};
    for (const auto &[groupDim, groupTag] : physicalGroups)
    {
        std::string name;
        gmsh::model::getPhysicalName(2, groupTag, name);
        if (name == physicalGroupName)
        {
            targetGroupTag = groupTag;
            break;
        }
    }

    if (targetGroupTag == -1)
        throw std::runtime_error(util::stringify("Error: Physical group '", physicalGroupName, "' not found."));

    return targetGroupTag;
}

TriangleCellCentersMap GmshUtils::getCellCentersByPhysicalGroupName(std::string_view physicalGroupName, std::string_view meshFilename)
{
    GmshUtils::checkAndOpenMesh(meshFilename);

    // Step 1: Find the physical group with the specified name.
    int targetGroupTag{GmshUtils::getPhysicalGroupTagByName(physicalGroupName, meshFilename)};

    // Step 2: Get the nodes associated with the searching surface physical group.
    std::vector<size_t> targetNodeTags;
    std::vector<double> targetCoords;
    gmsh::model::mesh::getNodesForPhysicalGroup(2, targetGroupTag, targetNodeTags, targetCoords);

    if (targetNodeTags.empty())
        throw std::runtime_error(util::stringify("There is no node tags for physical group with name: ", physicalGroupName));

    // Step 3: Identify triangles associated with the target surface.
    std::unordered_map<size_t, std::vector<size_t>> triangleNodeTagsMap;
    std::vector<size_t> triangleTags, triangleNodeTags;
    gmsh::model::mesh::getElementsByType(2, triangleTags, triangleNodeTags);

    if (triangleTags.empty() || triangleNodeTags.empty())
        throw std::runtime_error(util::stringify("There is no triangle cells for physical group with name: ", physicalGroupName));

    // Store the target node tags in a set for quick lookup.
    std::set<size_t> targetNodeTagSet(targetNodeTags.cbegin(), targetNodeTags.cend());
    for (size_t i{}; i < triangleTags.size(); ++i)
    {
        size_t triangleTag{triangleTags.at(i)};
        std::vector<size_t> triangleNodes;
        bool allNodesInTarget{true};

        for (short j{}; j < 3; ++j)
        {
            size_t nodeTag{triangleNodeTags.at(i * 3 + j)};
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

    if (triangleNodeTagsMap.empty())
        throw std::runtime_error("Failed to fill 'triangleNodeTagsMap' variable.");

    // Step 4: Calculate centroids of triangles.
    std::unordered_map<size_t, std::array<double, 3ul>> triangleCentersMap;
    for (const auto &[cellTag, cellNodeTags] : triangleNodeTagsMap)
    {
        if (cellNodeTags.size() != 3ul)
        {
            WARNINGMSG(util::stringify("Triangle with tag ", cellTag, " does not have exactly 3 nodes."));
            continue;
        }

        std::array<double, 3ul> centroid{0.0, 0.0, 0.0};
        for (size_t cellNodeTag : cellNodeTags)
        {
            int dim{}, tag{};
            std::vector<double> coord, paramCoord;
            gmsh::model::mesh::getNode(cellNodeTag, coord, paramCoord, dim, tag);

            if (coord.size() != 3)
            {
                WARNINGMSG(util::stringify("Node ", cellNodeTag, " does not have 3D coordinates."));
                continue;
            }

            centroid[0] += coord[0];
            centroid[1] += coord[1];
            centroid[2] += coord[2];
        }

        // Average the coordinates to get the centroid.
        centroid[0] /= 3.0;
        centroid[1] /= 3.0;
        centroid[2] /= 3.0;

        triangleCentersMap[cellTag] = centroid;
    }

    if (triangleCentersMap.empty())
        throw std::runtime_error("Failed to fill 'triangleCentersMap' variable.");

    return triangleCentersMap;
}

TriangleCellMap GmshUtils::getCellsByPhysicalGroupName(std::string_view physicalGroupName, std::string_view meshFilename)
{
    GmshUtils::checkAndOpenMesh(meshFilename);

    // Step 1: Find the physical group with the specified name.
    int targetGroupTag{GmshUtils::getPhysicalGroupTagByName(physicalGroupName, meshFilename)};

    // Step 2: Get the nodes associated with the searching surface physical group.
    std::vector<size_t> targetNodeTags;
    std::vector<double> targetCoords;
    gmsh::model::mesh::getNodesForPhysicalGroup(2, targetGroupTag, targetNodeTags, targetCoords);

    if (targetNodeTags.empty())
        throw std::runtime_error(util::stringify("There is no node tags for physical group with name: ", physicalGroupName));

    // Step 3: Identify triangles associated with the target surface.
    std::unordered_map<size_t, std::vector<size_t>> triangleNodeTagsMap;
    std::vector<size_t> triangleTags, triangleNodeTags;
    gmsh::model::mesh::getElementsByType(2, triangleTags, triangleNodeTags);

    if (triangleTags.empty() || triangleNodeTags.empty())
        throw std::runtime_error(util::stringify("There is no triangle cells for physical group with name: ", physicalGroupName));

    // Store the target node tags in a set for quick lookup.
    std::set<size_t> targetNodeTagSet(targetNodeTags.cbegin(), targetNodeTags.cend());
    for (size_t i{}; i < triangleTags.size(); ++i)
    {
        size_t triangleTag{triangleTags.at(i)};
        std::vector<size_t> triangleNodes;
        bool allNodesInTarget{true};

        for (short j{}; j < 3; ++j)
        {
            size_t nodeTag{triangleNodeTags.at(i * 3 + j)};
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

    if (triangleNodeTagsMap.empty())
        throw std::runtime_error("Failed to fill 'triangleNodeTagsMap' variable.");

    // Step 4: Calculate centroids of triangles.
    TriangleCellMap cellsMap;
    for (auto const &[triangleTag, triangleNodeTags] : triangleNodeTagsMap)
    {
        if (triangleNodeTags.size() != 3ul)
        {
            WARNINGMSG(util::stringify("Triangle with tag ", triangleTag, " does not have exactly 3 nodes."));
            continue;
        }

        std::array<Point, 3> trianglePoints;
        for (short i{}; i < 3; ++i)
        {
            int dim{}, tag{};
            std::vector<double> coord, paramCoord;
            gmsh::model::mesh::getNode(triangleNodeTags.at(i), coord, paramCoord, dim, tag);

            if (coord.size() != 3ul)
            {
                WARNINGMSG(util::stringify("Node ", triangleNodeTags.at(i), " does not have 3D coordinates. Supported only 3D."));
                continue;
            }

            trianglePoints[i] = Point(coord.at(0), coord.at(1), coord.at(2));
        }

        // Create a CGAL Triangle object from the points and add it to the map.
        try
        {
            Triangle triangle(trianglePoints.at(0), trianglePoints.at(1), trianglePoints.at(2));
            if (triangle.is_degenerate())
            {
                WARNINGMSG(util::stringify("Triangle with ID ", triangleTag, " is degenerate, skipping it..."));
                continue;
            }

            // Assuming that in initial time moment (t=0) there is no any settled particles (3rd param is 0).
            cellsMap[triangleTag] = TriangleCell(triangle, TriangleCell::compute_area(triangle), 0);
        }
        catch (std::exception const &ex)
        {
            throw std::runtime_error(util::stringify("Failed to create triangle for tag ", triangleTag, ": ", ex.what()));
        }
        catch (...)
        {
            throw std::runtime_error(util::stringify("Failed to create trignale for tag ", triangleTag, " by unknown reason."));
        }
    }

    if (cellsMap.empty())
        throw std::runtime_error("Failed to fill 'cellsMap' variable.");

    return cellsMap;
}

std::vector<std::tuple<int, int, std::string>> GmshUtils::getAllPhysicalGroups(std::string_view meshFilename)
{
    GmshUtils::checkAndOpenMesh(meshFilename);

    std::vector<std::pair<int, int>> physicalGroupsDimTags;
    gmsh::model::getPhysicalGroups(physicalGroupsDimTags);

    if (physicalGroupsDimTags.empty())
        throw std::runtime_error("No physical groups found in the Gmsh model.");

    std::vector<std::tuple<int, int, std::string>> result;
    for (auto const &[dim, tag] : physicalGroupsDimTags)
    {
        std::string name;
        gmsh::model::getPhysicalName(dim, tag, name);
        result.emplace_back(dim, tag, name);
    }

    if (result.empty())
        throw std::runtime_error("Resulting vector of tuples 'std::vector<std::tuple<int, int, std::string>>' with physical groups is empty.");

    return result;
}

bool GmshUtils::hasPhysicalGroup(std::string_view physicalGroupName, std::string_view meshFilename)
{
    checkAndOpenMesh(meshFilename);
    auto physicalGroups{GmshUtils::getAllPhysicalGroups(meshFilename)};

#if __cplusplus >= 202002L
    return std::ranges::find_if(physicalGroups, [physicalGroupName](auto const &physicalGroup)
                                {
            auto const &[dim, tag, name]{physicalGroup};
            return name == physicalGroupName; }) != physicalGroups.cend();
#else
    return std::find_if(physicalGroups.cbegin(), physicalGroups.cend(), [physicalGroupName](auto const &physicalGroup)
                        {
            auto const &[dim, tag, name]{physicalGroup};
            return name == physicalGroupName; }) != physicalGroups.cend();
#endif
}
