#include <set>

#include "Utilities/GmshUtilities/GmshUtils.hpp"

void GmshUtils::gmshInitializeCheck()
{
    if (!gmsh::isInitialized())
        START_THROW_EXCEPTION(GmshUtilsGmshNotInitializedException,
                              "This method works only when Gmsh was initialized.");
}

void GmshUtils::checkGmshMeshFile(std::string_view mesh_filename)
{
    if (!std::filesystem::exists(mesh_filename))
        START_THROW_EXCEPTION(GmshUtilsFileDoesNotExistException,
                              "File does not exist: " + std::string(mesh_filename));
    if (std::filesystem::is_directory(mesh_filename))
        START_THROW_EXCEPTION(GmshUtilsFileIsDirectoryException,
                              "Provided path is a directory, not a file: " + std::string(mesh_filename));
    if (std::filesystem::path(mesh_filename).extension() != ".msh")
        START_THROW_EXCEPTION(GmshUtilsFileExtensionIsNotMshException,
                              "File extension is not .msh: " + std::string(mesh_filename));
    if (std::filesystem::file_size(mesh_filename) == 0)
        START_THROW_EXCEPTION(GmshUtilsFileIsEmptyException,
                              "File is empty: " + std::string(mesh_filename));
}

void GmshUtils::checkAndOpenMesh(std::string_view meshFilename)
{
    GmshUtils::gmshInitializeCheck();
    GmshUtils::checkGmshMeshFile(meshFilename);

    gmsh::option::setNumber("General.Terminal", 0); // Turn off logging to the terminal.
    gmsh::open(meshFilename.data());
    gmsh::option::setNumber("General.Terminal", 1); // Turn on logging to the terminal.
}

std::vector<int> GmshUtils::getAllBoundaryTags()
{
    std::vector<std::pair<int, int>> volumes;
    gmsh::model::getEntities(volumes, 3);

    if (volumes.empty())
        START_THROW_EXCEPTION(GmshUtilsNoVolumeEntitiesException, "There is no volume entities.");

    std::vector<int> allBoundaryTags;
    for (auto const &volume : volumes)
    {
        std::vector<std::pair<int, int>> boundaries;
        gmsh::model::getBoundary({volume}, boundaries, true, false, false);
        for (auto const &boundary : boundaries)
            allBoundaryTags.push_back(boundary.second);
    }

    if (allBoundaryTags.empty())
        START_THROW_EXCEPTION(GmshUtilsNoBoundaryTagsException, "There is no boundary tags.");

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
        START_THROW_EXCEPTION(GmshUtilsNoPhysicalGroupsException, "There is no physical groups.");

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
        START_THROW_EXCEPTION(GmshUtilsPhysicalGroupNotFoundException,
                              util::stringify("Error: Physical group '", physicalGroupName, "' not found."));

    return targetGroupTag;
}

TriangleCellMap GmshUtils::getTriangleCellsMap(std::string_view meshFilename)
{
    TriangleCellMap result;
    try
    {
        GmshUtils::checkAndOpenMesh(meshFilename);

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
        START_THROW_EXCEPTION(GmshUtilsFailedToFillTriangleCellsMapException,
                              "By some reason 'result' data ('TriangleCellMap') is empty, check the input...");

    return result;
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
        START_THROW_EXCEPTION(GmshUtilsNoNodeTagsException,
                              util::stringify("There is no node tags for physical group with name: ", physicalGroupName));

    // Step 3: Identify triangles associated with the target surface.
    std::unordered_map<size_t, std::vector<size_t>> triangleNodeTagsMap;
    std::vector<size_t> triangleTags, triangleNodeTags;
    gmsh::model::mesh::getElementsByType(2, triangleTags, triangleNodeTags);

    if (triangleTags.empty() || triangleNodeTags.empty())
        START_THROW_EXCEPTION(GmshUtilsNoTriangleCellsException,
                              util::stringify("There is no triangle cells for physical group with name: ", physicalGroupName));

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
        START_THROW_EXCEPTION(GmshUtilsFailedToFillTriangleNodeTagsMapException,
                              "Failed to fill 'triangleNodeTagsMap' variable.");

    // Step 4: Calculate centroids of triangles.
    TriangleCellCentersMap triangleCentersMap;
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

        triangleCentersMap[cellTag] = Point(centroid[0], centroid[1], centroid[2]);
    }

    if (triangleCentersMap.empty())
        START_THROW_EXCEPTION(GmshUtilsFailedToFillTriangleCentersMapException,
                              "Failed to fill 'triangleCentersMap' variable.");

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
        START_THROW_EXCEPTION(GmshUtilsNoNodeTagsException,
                              util::stringify("There is no node tags for physical group with name: ", physicalGroupName));

    // Step 3: Identify triangles associated with the target surface.
    std::unordered_map<size_t, std::vector<size_t>> triangleNodeTagsMap;
    std::vector<size_t> triangleTags, triangleNodeTags;
    gmsh::model::mesh::getElementsByType(2, triangleTags, triangleNodeTags);

    if (triangleTags.empty() || triangleNodeTags.empty())
        START_THROW_EXCEPTION(GmshUtilsNoTriangleCellsException,
                              util::stringify("There is no triangle cells for physical group with name: ", physicalGroupName));

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
        START_THROW_EXCEPTION(GmshUtilsFailedToFillTriangleNodeTagsMapException,
                              "Failed to fill 'triangleNodeTagsMap' variable.");

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
            START_THROW_EXCEPTION(GmshUtilsFailedToCreateTriangleException,
                                  util::stringify("Failed to create triangle for tag ", triangleTag, ": ", ex.what()));
        }
        catch (...)
        {
            START_THROW_EXCEPTION(GmshUtilsFailedToCreateTriangleException,
                                  util::stringify("Failed to create trignale for tag ", triangleTag, " by unknown reason."));
        }
    }

    if (cellsMap.empty())
        START_THROW_EXCEPTION(GmshUtilsFailedToFillCellsMapException,
                              "Failed to fill 'cellsMap' variable.");

    return cellsMap;
}

std::vector<std::tuple<int, int, std::string>> GmshUtils::getAllPhysicalGroups(std::string_view meshFilename)
{
    GmshUtils::checkAndOpenMesh(meshFilename);

    std::vector<std::pair<int, int>> physicalGroupsDimTags;
    gmsh::model::getPhysicalGroups(physicalGroupsDimTags);

    if (physicalGroupsDimTags.empty())
        START_THROW_EXCEPTION(GmshUtilsNoPhysicalGroupsException,
                              "No physical groups found in the Gmsh model.");

    std::vector<std::tuple<int, int, std::string>> result;
    for (auto const &[dim, tag] : physicalGroupsDimTags)
    {
        std::string name;
        gmsh::model::getPhysicalName(dim, tag, name);
        result.emplace_back(dim, tag, name);
    }

    if (result.empty())
        START_THROW_EXCEPTION(GmshUtilsNoPhysicalGroupsException,
                              "Resulting vector of tuples 'std::vector<std::tuple<int, int, std::string>>' with physical groups is empty.");

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
