#ifndef SETTLED_PARTICLE_HDF5_WRITER_HPP
#define SETTLED_PARTICLE_HDF5_WRITER_HPP

#include <array>
#include <hdf5.h>
#include <string_view>
#include <vector>

#include "Utilities/PreprocessorUtils.hpp"

/**
 * @brief Class responsible for saving settled particles data to HDF5 file
 *
 * This class handles the creation of HDF5 files and writing particle position
 * data to them with proper error handling and resource management.
 */
class SettledParticleHDF5Writer
{
public:
    /**
     * @brief Saves a collection of settled particle positions to an HDF5 file
     *
     * @param positions Vector of 3D particle positions
     * @param filepath Path to the output HDF5 file
     * @param datasetName Name of the dataset in the HDF5 file (default: "settled_particles")
     * @return bool True if the operation was successful, false otherwise
     *
     * @details Algorithm:
     *  1. Create/open the HDF5 file
     *  2. Create a dataspace with dimensions [particleCount, 3]
     *  3. Create a dataset for particle positions
     *  4. Write position data to the dataset
     *  5. Close all HDF5 resources
     *
     * @exception Handles HDF5 errors internally and logs appropriate messages
     */
    static bool saveParticlesToHDF5(
        std::vector<std::array<double, 3>> const &positions,
        std::string_view filepath,
        std::string_view datasetName = "settled_particles");

private:
    /**
     * @brief Creates an HDF5 file with error checking
     *
     * @param filepath Path to the output file
     * @return hid_t HDF5 file identifier or negative value on error
     */
    static hid_t _createFile(std::string_view filepath);

    /**
     * @brief Creates a dataspace for particle positions
     *
     * @param particleCount Number of particles
     * @return hid_t HDF5 dataspace identifier or negative value on error
     */
    static hid_t _createDataspace(size_t particleCount);

    /**
     * @brief Creates a dataset in the HDF5 file
     *
     * @param fileId HDF5 file identifier
     * @param dataspaceId HDF5 dataspace identifier
     * @param datasetName Name of the dataset
     * @return hid_t HDF5 dataset identifier or negative value on error
     */
    static hid_t _createDataset(
        hid_t fileId,
        hid_t dataspaceId,
        std::string_view datasetName);
};

#endif // !SETTLED_PARTICLE_HDF5_WRITER_HPP
