#ifndef TRIANGLEMESHHDF5MANAGER_HPP
#define TRIANGLEMESHHDF5MANAGER_HPP

#include <limits>
#include <string_view>
#include <unordered_map>

#include <hdf5.h>

#include "Geometry/Mesh/Surface/TriangleCell.hpp"

/**
 * @brief Handles operations related to HDF5 files for storing and managing mesh data.
 * @details This class provides functionalities to create, read, and update data in
 *          an HDF5 file. It is specifically designed to handle mesh data, including
 *          the coordinates, areas, and particle counters of each triangle in the mesh.
 */
class TriangleMeshHdf5Manager final
{
private:
    hid_t m_file_id;                                  // File id.
    size_t m_firstID{},                               // ID of the first triangle in the mesh.
        m_lastID{std::numeric_limits<size_t>::max()}; // ID of the last triangle in the mesh.

    /**
     * @brief Finds the minimum triangle ID in a vector of triangle cell maps.
     *
     * This method determines the smallest triangle ID by comparing the keys of the
     * first elements in each `TriangleCellMap` within the `TriangleCellMapVector`.
     *
     * @param triangleCells A vector of unordered maps, where each map represents
     *                      a collection of triangle cells and their associated data.
     *
     * @throws TriangleMeshHdf5ManagerInputParameterEmptyException If the input `triangleCells` vector is empty.
     *
     * @note The method assumes that all maps in the input vector contain unique IDs as keys.
     *       If a map is empty, it is ignored during the comparison.
     *
     * @see TriangleMeshHdf5Manager::readMeshFromHDF5()
     * @see TriangleMeshHdf5Manager::saveMeshToHDF5()
     */
    void _findMinTriangleId(TriangleCellMap_cref triangleCells);

    /**
     * @brief Creates a new group in the HDF5 file.
     * @details This function creates a new group in the HDF5 file with the specified name.
     *          It uses H5Gcreate2 for group creation and checks if the operation is successful.
     *          If the group creation fails, it throws an exception.
     * @param groupName The name of the group to be created.
     *                  It is a std::string_view to avoid unnecessary string copies.
     *
     * @throws TriangleMeshHdf5ManagerFailedToCreateGroupException If the group creation fails.
     */
    void _createGroup(std::string_view groupName);

    /**
     * @brief Writes data to a dataset in a specified group within the HDF5 file.
     * @details This function opens the specified group, creates a dataspace and a dataset,
     *          and then writes the provided data to the dataset. If any of these operations fail,
     *          it throws an exception. It uses H5Dwrite to write data and closes the handles
     *          for the dataset and dataspace after the operation.
     *
     * @param groupName The name of the group containing the dataset.
     * @param datasetName The name of the dataset to which data will be written.
     * @param type The HDF5 data type of the dataset.
     * @param data A pointer to the data to be written.
     * @param dims The dimension size of the dataset.
     *
     * @throws TriangleMeshHdf5ManagerFailedToCreateDatasetException If creating the dataspace or dataset fails.
     */
    void _writeDataset(std::string_view groupName, std::string_view datasetName,
                       hid_t type, void const *data, hsize_t dims);

    /**
     * @brief Reads data from a dataset within a specified group in the HDF5 file.
     * @details This function opens the specified group and dataset, and then reads
     *          the data into the provided buffer. It uses H5Dread for reading the data.
     *          If opening the group or the dataset fails, an exception is thrown.
     *          After the operation, it closes the handles for the dataset and the group.
     *
     * @param groupName The name of the group containing the dataset.
     * @param datasetName The name of the dataset from which data will be read.
     * @param type The HDF5 data type of the dataset.
     * @param data A pointer where the read data will be stored.
     *
     * @throws TriangleMeshHdf5ManagerFailedToOpenDatasetException If the dataset opening fails.
     */
    void _readDataset(std::string_view groupName, std::string_view datasetName,
                      hid_t type, void *data);

public:
    /**
     * @brief Constructs an TriangleMeshHdf5Manager object and opens or creates an HDF5 file.
     * @param filename The name of the HDF5 file to be opened or created.
     * @details The constructor opens an HDF5 file if it exists, or creates a new one if it does not.
     *          The file is opened with write access, and the file handle is stored for future operations.
     */
    explicit TriangleMeshHdf5Manager(std::string_view filename);
    ~TriangleMeshHdf5Manager();

    /**
     * @brief Saves mesh data to the HDF5 file.
     * @param triangleCells A vector of tuples representing the mesh's triangles, with each tuple containing
     *                  the triangle's ID, vertices (as PositionVector objects), and area.
     * @details This method iterates through the given vector of triangles, creating a group for each
     *          triangle in the HDF5 file. Within each group, it stores datasets for the triangle's
     *          coordinates, area, and initializes the particle counter to zero.
     * @throws TriangleMeshHdf5ManagerFailedToCreateGroupException If it fails to create a group within the HDF5 file,
     *         or if writing to the dataset fails.
     */
    void saveMeshToHDF5(TriangleCellMap_cref triangleCells);

    /**
     * @brief Reads mesh data from the HDF5 file starting from a specified object ID.
     * @return A vector of tuples representing the mesh's triangles, with each tuple containing
     *         the triangle's ID, vertices, area, and particle counter.
     * @details This method reads the HDF5 file and constructs a vector of tuples, each representing a
     *          triangle's data. It retrieves the triangle's ID, coordinates, area, and particle counter
     *          from the HDF5 file, starting from the triangle with ID `firstObjectID`.
     * @throws TriangleMeshHdf5ManagerFailedToOpenGroupException If it fails to open a group within the HDF5 file.
     */
    TriangleCellMap readMeshFromHDF5();
};

#endif // !TRIANGLEMESHHDF5MANAGER_HPP
