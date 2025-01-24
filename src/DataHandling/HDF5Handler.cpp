#include <algorithm>
#include <filesystem>
#include <stdexcept>

#if __cplusplus >= 202002L
#include <ranges>
#endif

#include "DataHandling/HDF5Handler.hpp"

void HDF5Handler::_findMinTriangleId(TriangleCellMap const &triangleCells)
{
    if (triangleCells.empty())
        throw std::invalid_argument("Input parameter 'triangleCells' is empty.");

#if __cplusplus >= 202002L
    auto minId{std::ranges::min_element(
        triangleCells,
        [](auto const &a, auto const &b)
        {
            return a.first < b.first;
        })};
#else
    auto minId{std::min_element(
        triangleCells.cbegin(), triangleCells.cend(),
        [](auto const &a, auto const &b)
        {
            return a.first < b.first;
        })};
#endif

    if (minId != triangleCells.end())
        m_firstID = minId->first;
    else
        throw std::runtime_error("Failed to find the minimum ID in 'triangleCells'.");
}

HDF5Handler::HDF5Handler(std::string_view filename)
{
    if (std::filesystem::exists(filename))
        std::filesystem::remove(filename);

    m_file_id = H5Fcreate(filename.data(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (m_file_id < 0)
        throw std::runtime_error("Failed to create HDF5 file: " + std::string(filename));
}

HDF5Handler::~HDF5Handler() { H5Fclose(m_file_id); }

void HDF5Handler::_createGroup(std::string_view groupName)
{
    hid_t grp_id{H5Gcreate2(m_file_id, groupName.data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
    if (grp_id < 0)
        throw std::runtime_error("Failed to create group: " + std::string(groupName));
    H5Gclose(grp_id);
}

void HDF5Handler::_writeDataset(std::string_view groupName, std::string_view datasetName,
                                hid_t type, void const *data, hsize_t dims)
{
    hid_t grp_id{H5Gopen2(m_file_id, groupName.data(), H5P_DEFAULT)},
        dataspace{H5Screate_simple(1, std::addressof(dims), NULL)},
        dataset{H5Dcreate2(grp_id, datasetName.data(), type, dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
    if (dataspace < 0 || dataset < 0)
    {
        H5Gclose(grp_id);
        throw std::runtime_error("Failed to create dataset " + std::string(datasetName) + " in group: " + std::string(groupName));
    }
    H5Dwrite(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dataset);
    H5Sclose(dataspace);
    H5Gclose(grp_id);
}

void HDF5Handler::_readDataset(std::string_view groupName, std::string_view datasetName,
                               hid_t type, void *data)
{
    hid_t grp_id{H5Gopen2(m_file_id, groupName.data(), H5P_DEFAULT)},
        dataset{H5Dopen2(grp_id, datasetName.data(), H5P_DEFAULT)};
    if (dataset < 0)
    {
        H5Gclose(grp_id);
        throw std::runtime_error("Failed to open dataset " + std::string(datasetName) + " in group: " + std::string(groupName));
    }
    H5Dread(dataset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    H5Dclose(dataset);
    H5Gclose(grp_id);
}

void HDF5Handler::saveMeshToHDF5(TriangleCellMap const &triangleCells)
{
    if (triangleCells.empty())
        return;

    _findMinTriangleId(triangleCells);
    for (auto const &[id, triangleCell] : triangleCells)
    {
        // Creating a group for each triangle using the triangle ID as the group name
        std::string groupName("Triangle_" + std::to_string(id));
        _createGroup(groupName);

        // Store data related to the triangle in the group
        double coordinates[9] = {
            triangleCell.triangle.vertex(0).x(),
            triangleCell.triangle.vertex(0).y(),
            triangleCell.triangle.vertex(0).z(),
            triangleCell.triangle.vertex(1).x(),
            triangleCell.triangle.vertex(1).y(),
            triangleCell.triangle.vertex(1).z(),
            triangleCell.triangle.vertex(2).x(),
            triangleCell.triangle.vertex(2).y(),
            triangleCell.triangle.vertex(2).z(),
        };
        _writeDataset(groupName, "Coordinates", H5T_NATIVE_DOUBLE, coordinates, 9);

        double area{triangleCell.area};
        _writeDataset(groupName, "Area", H5T_NATIVE_DOUBLE, std::addressof(area), 1);

        size_t count{triangleCell.count};

        // Use the appropriate HDF5 type for size_t
#if SIZE_MAX == UINT_MAX
        _writeDataset(groupName, "Counter", H5T_NATIVE_UINT, std::addressof(count), 1);
#elif SIZE_MAX == ULONG_MAX
        _writeDataset(groupName, "Counter", H5T_NATIVE_ULONG, std::addressof(count), 1);
#elif SIZE_MAX == ULLONG_MAX
        _writeDataset(groupName, "Counter", H5T_NATIVE_ULLONG, std::addressof(count), 1);
#else
        static_assert(false, "Unsupported size_t width for HDF5");
#endif
    }
}

TriangleCellMap HDF5Handler::readMeshFromHDF5()
{
    TriangleCellMap cellsMap;
    hsize_t num_objs{};
    H5Gget_num_objs(m_file_id, std::addressof(num_objs));
    m_lastID = m_firstID + num_objs;

    for (size_t id{m_firstID}; id < m_lastID; ++id)
    {
        std::string groupName("Triangle_" + std::to_string(id));

        double coordinates[9]{};
        _readDataset(groupName, "Coordinates", H5T_NATIVE_DOUBLE, coordinates);

        double area{};
        _readDataset(groupName, "Area", H5T_NATIVE_DOUBLE, std::addressof(area));

        size_t count{};
#if SIZE_MAX == UINT_MAX
        _readDataset(groupName, "Counter", H5T_NATIVE_UINT, std::addressof(count));
#elif SIZE_MAX == ULONG_MAX
        _readDataset(groupName, "Counter", H5T_NATIVE_ULONG, std::addressof(count));
#elif SIZE_MAX == ULLONG_MAX
        _readDataset(groupName, "Counter", H5T_NATIVE_ULLONG, std::addressof(count));
#else
        static_assert(false, "Unsupported size_t width for HDF5");
#endif

        // Construct the tuple and add to the mesh vector
        Triangle triangle(Point(coordinates[0], coordinates[1], coordinates[2]),
                          Point(coordinates[3], coordinates[4], coordinates[5]),
                          Point(coordinates[6], coordinates[7], coordinates[8]));
        TriangleCell triangleCell{triangle, area, count};
        cellsMap[id] = triangleCell;
    }

    return cellsMap;
}
