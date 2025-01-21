#include <set>

#include "Utilities/GmshUtilities/GmshUtils.hpp"

int GmshUtils::gmshInitializeCheck()
{
    if (!gmsh::isInitialized())
    {
        ERRMSG("This method works only when Gmsh was initialized.");
        return -1;
    }
    else
        return 0;
}

std::vector<int> GmshUtils::getAllBoundaryTags()
{
    if (GmshUtils::gmshInitializeCheck() == -1)
        return {};

    std::vector<std::pair<int, int>> volumes;
    gmsh::model::getEntities(volumes, 3);

    if (volumes.empty())
    {
        WARNINGMSG("There is no volume entities.");
        return {};
    }

    std::vector<int> allBoundaryTags;
    for (auto const &volume : volumes)
    {
        std::vector<std::pair<int, int>> boundaries;
        gmsh::model::getBoundary({volume}, boundaries, true, false, false);
        for (auto const &boundary : boundaries)
            allBoundaryTags.push_back(boundary.second);
    }

    if (allBoundaryTags.empty())
    {
        WARNINGMSG("There is no boundary tags.");
    }

    return allBoundaryTags;
}

int GmshUtils::getPhysicalGroupTagByName(std::string_view physicalGroupName)
{
    if (GmshUtils::gmshInitializeCheck() == -1)
        return -1;

    std::vector<std::pair<int, int>> physicalGroups;
    gmsh::model::getPhysicalGroups(physicalGroups, 2);

    if (physicalGroups.empty())
    {
        WARNINGMSG("There is no physical groups");
    }

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
    {
        ERRMSG("Error: Physical group 'Target' not found.");
        return {};
    }

    return targetGroupTag;
}

std::unordered_map<size_t, std::array<double, 3ul>> GmshUtils::getCellCentersByPhysicalGroupName(std::string_view physicalGroupName)
{
    // Step 1: Find the physical group with the specified name.
    int targetGroupTag{GmshUtils::getPhysicalGroupTagByName(physicalGroupName)};

    // Step 2: Get the nodes associated with the searching surface physical group.
    std::vector<size_t> targetNodeTags;
    std::vector<double> targetCoords;
    gmsh::model::mesh::getNodesForPhysicalGroup(2, targetGroupTag, targetNodeTags, targetCoords);

    if (targetNodeTags.empty())
    {
        WARNINGMSG(util::stringify("There is no node tags for physical group with name: ", physicalGroupName));
        return {};
    }

    // Step 3: Identify triangles associated with the target surface.
    std::unordered_map<size_t, std::vector<size_t>> triangleNodeTagsMap;
    std::vector<size_t> triangleTags, triangleNodeTags;
    gmsh::model::mesh::getElementsByType(2, triangleTags, triangleNodeTags);

    if (triangleTags.empty() || triangleNodeTags.empty())
    {
        WARNINGMSG(util::stringify("There is no triangle cells for physical group with name: ", physicalGroupName));
        return {};
    }

    constexpr short const kdefault_nodes_per_triangle{3};

    // Store the target node tags in a set for quick lookup.
    std::set<size_t> targetNodeTagSet(targetNodeTags.cbegin(), targetNodeTags.cend());
    for (size_t i{}; i < triangleTags.size(); ++i)
    {
        size_t triangleTag{triangleTags.at(i)};
        std::vector<size_t> triangleNodes;
        bool allNodesInTarget{true};

        for (short j{}; j < kdefault_nodes_per_triangle; ++j)
        {
            size_t nodeTag{triangleNodeTags.at(i * kdefault_nodes_per_triangle + j)};
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
    {
        WARNINGMSG("Failed to fill 'triangleNodeTagsMap' variable.");
        return {};
    }

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
    {
        WARNINGMSG("Failed to fill 'triangleCentersMap' variable.");
        return {};
    }

    return triangleCentersMap;
}
