#include "DataHandling/SettledParticleHDF5Writer.hpp"
#include "Utilities/LogMacros.hpp"

bool SettledParticleHDF5Writer::saveParticlesToHDF5(
    std::vector<std::array<double, 3>> const &positions,
    std::string_view filepath,
    std::string_view datasetName)
{
    try
    {
        // 1. Check if there are any particles to save
        if (positions.empty())
        {
            WARNINGMSG("No particles to save. Skipping HDF5 file creation");
            return false;
        }

        // 2. Create file with error checking
        hid_t file_id = _createFile(filepath);
        if (file_id < 0)
            return false;

        // 3. Create dataspace
        hid_t dataspace = _createDataspace(positions.size());
        if (dataspace < 0)
        {
            H5Fclose(file_id);
            return false;
        }

        // 4. Create dataset
        hid_t dataset = _createDataset(file_id, dataspace, datasetName);
        if (dataset < 0)
        {
            H5Sclose(dataspace);
            H5Fclose(file_id);
            return false;
        }

        // 5. Write data
        herr_t status = H5Dwrite(dataset,
                                 H5T_NATIVE_DOUBLE,
                                 H5S_ALL,
                                 H5S_ALL,
                                 H5P_DEFAULT,
                                 positions.data());
        if (status < 0)
        {
            H5Dclose(dataset);
            H5Sclose(dataspace);
            H5Fclose(file_id);
            ERRMSG("HDF5 data write failed");
            return false;
        }

        // 6. Correctly release resources
        H5Dclose(dataset);
        H5Sclose(dataspace);
        H5Fclose(file_id);

        SUCCESSMSG(util::stringify("Successfully saved ", positions.size(), " particles to ", filepath));
        return true;
    }
    catch (const std::exception &e)
    {
        ERRMSG(util::stringify("Error: ", e.what()));
        return false;
    }
    catch (...)
    {
        ERRMSG("Unknown error during HDF5 write operation");
        return false;
    }
}

hid_t SettledParticleHDF5Writer::_createFile(std::string_view filepath)
{
    hid_t file_id = H5Fcreate(filepath.data(),
                              H5F_ACC_TRUNC,
                              H5P_DEFAULT,
                              H5P_DEFAULT);
    if (file_id < 0)
        ERRMSG("HDF5 file creation failed");

    return file_id;
}

hid_t SettledParticleHDF5Writer::_createDataspace(size_t particleCount)
{
    hsize_t dims[2] = {particleCount, 3}; // Nx3 array for 3D positions
    hid_t dataspace = H5Screate_simple(2, dims, NULL);
    if (dataspace < 0)
        ERRMSG("HDF5 dataspace creation failed");

    return dataspace;
}

hid_t SettledParticleHDF5Writer::_createDataset(
    hid_t fileId,
    hid_t dataspaceId,
    std::string_view datasetName)
{
    hid_t dataset = H5Dcreate2(fileId,
                               datasetName.data(),
                               H5T_NATIVE_DOUBLE,
                               dataspaceId,
                               H5P_DEFAULT,
                               H5P_DEFAULT,
                               H5P_DEFAULT);
    if (dataset < 0)
        ERRMSG("HDF5 dataset creation failed");

    return dataset;
}