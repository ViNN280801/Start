# src/DataHandling/CMakeLists.txt

set(DATA_HANDLING_SOURCES
    SettledParticleHDF5Writer.cpp
    TriangleMeshHdf5Manager.cpp
)

set(LIB_DATA_HANDLING DataHandling)
add_library(${LIB_DATA_HANDLING} ${DATA_HANDLING_SOURCES})

target_link_libraries(${LIB_DATA_HANDLING} PRIVATE
    ${HDF5_LIBRARIES}
)

target_include_directories(${LIB_DATA_HANDLING} PRIVATE
    ${HDF5_INCLUDE_DIRS})
